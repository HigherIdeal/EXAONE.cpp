#!/usr/bin/env python3

import argparse
import gc
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm

from src.generate import Llama


DEFAULT_MODELS = (
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "EXAONE-4.0-1.2B",
)
EXPERIMENTS = ("MMLU-Pro", "KMMLU-Pro", "ARC-Challenge", "HellaSwag")
LETTERS = tuple("ABCDEFGHIJ")
GENERATION_EXPERIMENTS = {"MMLU-Pro", "KMMLU-Pro"}
DEFAULT_SHOTS = {
    "MMLU-Pro": 5,
    "KMMLU-Pro": 0,
    "ARC-Challenge": 25,
    "HellaSwag": 10,
}
MMLU_PRO_INSTRUCTION = (
    "The following are multiple choice questions (with answers) about {category}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)
MMLU_PRO_FALLBACK_SEED = 12345

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Example:
    prompt: str
    answer: int
    continuations: list[str] | None = None
    prompt_variants: list[str] | None = None
    answer_labels: tuple[str, ...] = ()


def parse_torch_dtype(value: str) -> torch.dtype:
    try:
        return DTYPE_MAP[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Unsupported torch dtype: {value}. Choose from: {', '.join(DTYPE_MAP)}"
        ) from exc


def sample_rows(rows: list[dict], num_samples: int, seed: int) -> list[dict]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if num_samples >= len(rows):
        return rows
    return random.Random(seed).sample(rows, num_samples)


def format_question(
    question: str,
    choices: list[str],
    labels: list[str] | tuple[str, ...],
    answer_prefix: str,
) -> str:
    lines = [f"Question: {question.strip()}"]
    lines.extend(f"{label}. {choice.strip()}" for label, choice in zip(labels, choices))
    lines.append(answer_prefix)
    return "\n".join(lines)


def format_demo(
    question: str,
    choices: list[str],
    labels: list[str] | tuple[str, ...],
    answer_index: int,
    answer_prefix: str,
) -> str:
    return (
        format_question(question, choices, labels, answer_prefix)
        + f" {labels[answer_index]}\n\n"
    )


def format_mmlu_pro_example(row: dict, including_answer: bool) -> str:
    options = [option for option in row["options"] if option != "N/A"]
    lines = ["Question:", row["question"].strip(), "Options:"]
    lines.extend(
        f"{LETTERS[index]}. {option.strip()}"
        for index, option in enumerate(options)
    )
    if including_answer:
        cot_content = row["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        lines.append(cot_content.strip())
        return "\n".join(lines) + "\n\n"

    lines.append("Answer: Let's think step by step.")
    return "\n".join(lines)


def build_mmlu_pro_prompt(
    demos: list[dict],
    row: dict,
    num_shots: int,
) -> str:
    prompt = MMLU_PRO_INSTRUCTION.format(category=row["category"]) + "\n"
    prompt += "".join(
        format_mmlu_pro_example(demo, including_answer=True)
        for demo in demos[:num_shots]
    )
    return prompt + format_mmlu_pro_example(row, including_answer=False)


def load_mmlu_pro(num_samples: int, seed: int) -> list[Example]:
    validation_rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="validation"))
    test_rows = sample_rows(
        list(load_dataset("TIGER-Lab/MMLU-Pro", split="test")),
        num_samples,
        seed,
    )

    demos_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in validation_rows:
        demos_by_category[row["category"]].append(row)

    examples = []
    for row in test_rows:
        demos = demos_by_category[row["category"]][: DEFAULT_SHOTS["MMLU-Pro"]]
        prompt_variants = [
            build_mmlu_pro_prompt(demos, row, num_shots)
            for num_shots in range(len(demos), -1, -1)
        ]
        options = [option for option in row["options"] if option != "N/A"]
        examples.append(
            Example(
                prompt=prompt_variants[0],
                answer=int(row["answer_index"]),
                prompt_variants=prompt_variants,
                answer_labels=LETTERS[: len(options)],
            )
        )
    return examples


def format_kmmlu_pro_prompt(
    question: str,
    options: list[str],
    language: str,
) -> str:
    labels = LETTERS[: len(options)]
    choices = "".join(labels)
    if language == "ko":
        instruction = (
            f"다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 {choices} 중 "
            '하나이고, "정답:" 뒤에 와야 합니다. 정답을 고르기 전에 차근차근 '
            "생각하고 추론하세요."
        )
    else:
        instruction = (
            "Answer the following multiple choice question. The last line of your "
            "response should be of the following format: 'Answer: $LETTER' "
            f"(without quotes) where LETTER is one of {choices}. Think step by step "
            "before answering."
        )

    lines = [instruction, question.strip()]
    lines.extend(
        f"{label}) {option.strip()}"
        for label, option in zip(labels, options)
    )
    return "\n".join(lines)


def load_kmmlu_pro(
    num_samples: int,
    seed: int,
    prompt_language: str,
) -> list[Example]:
    try:
        rows = list(
            load_dataset(
                "LGAI-EXAONE/KMMLU-Pro",
                "test",
                split="test",
            )
        )
    except ValueError:
        rows = list(load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test"))

    rows = sample_rows(rows, num_samples, seed)
    examples = []
    for row in rows:
        choices = list(row["options"])
        labels = LETTERS[: len(choices)]
        examples.append(
            Example(
                prompt=format_kmmlu_pro_prompt(
                    row["question"],
                    choices,
                    prompt_language,
                ),
                answer=int(row["solution"]) - 1,
                answer_labels=labels,
            )
        )
    return examples


def load_arc_challenge(num_samples: int, seed: int) -> list[Example]:
    train_rows = list(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    )
    test_rows = sample_rows(
        list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")),
        num_samples,
        seed,
    )
    demos = random.Random(seed).sample(
        train_rows,
        min(DEFAULT_SHOTS["ARC-Challenge"], len(train_rows)),
    )

    prefix_parts = []
    for demo in demos:
        labels = list(demo["choices"]["label"])
        prefix_parts.append(
            format_demo(
                demo["question"],
                list(demo["choices"]["text"]),
                labels,
                labels.index(str(demo["answerKey"])),
                "Answer:",
            )
        )
    prefix = "".join(prefix_parts)

    examples = []
    for row in test_rows:
        labels = list(row["choices"]["label"])
        examples.append(
            Example(
                prompt=prefix
                + format_question(
                    row["question"],
                    list(row["choices"]["text"]),
                    labels,
                    "Answer:",
                ),
                continuations=[f" {label}" for label in labels],
                answer=labels.index(str(row["answerKey"])),
            )
        )
    return examples


def load_hellaswag(num_samples: int, seed: int) -> list[Example]:
    train_rows = list(load_dataset("Rowan/hellaswag", split="train"))
    validation_rows = sample_rows(
        list(load_dataset("Rowan/hellaswag", split="validation")),
        num_samples,
        seed,
    )
    demos = random.Random(seed).sample(
        train_rows,
        min(DEFAULT_SHOTS["HellaSwag"], len(train_rows)),
    )
    prefix = "".join(
        f"Context: {demo['ctx'].strip()}\n"
        f"Completion: {demo['endings'][int(demo['label'])].strip()}\n\n"
        for demo in demos
    )

    return [
        Example(
            prompt=prefix + f"Context: {row['ctx'].strip()}\nCompletion:",
            continuations=list(row["endings"]),
            answer=int(row["label"]),
        )
        for row in validation_rows
    ]


def load_experiments(
    experiment_names: list[str],
    num_samples: int,
    seed: int,
    kmmlu_prompt_language: str,
) -> dict[str, list[Example]]:
    loaders = {
        "MMLU-Pro": load_mmlu_pro,
        "KMMLU-Pro": load_kmmlu_pro,
        "ARC-Challenge": load_arc_challenge,
        "HellaSwag": load_hellaswag,
    }
    experiments = {}
    for name in experiment_names:
        print(
            f"Loading {name}: samples={num_samples} "
            f"seed={seed} shots={DEFAULT_SHOTS[name]}"
        )
        try:
            if name == "KMMLU-Pro":
                experiments[name] = load_kmmlu_pro(
                    num_samples,
                    seed,
                    kmmlu_prompt_language,
                )
            else:
                experiments[name] = loaders[name](num_samples, seed)
        except DatasetNotFoundError as exc:
            if name != "KMMLU-Pro" or "gated dataset" not in str(exc):
                raise
            print(
                "Skipping KMMLU-Pro: gated dataset access is required. "
                "Request access at https://huggingface.co/datasets/"
                "LGAI-EXAONE/KMMLU-Pro and authenticate with "
                "`hf auth login` or `HF_TOKEN`."
            )
    return experiments


def tokenize_prompt_and_continuations(
    tokenizer,
    prompt: str,
    continuations: list[str],
    max_seq_len: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    continuation_ids = [
        tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        for continuation in continuations
    ]

    max_continuation = max(ids.numel() for ids in continuation_ids)
    if prompt_ids.numel() + max_continuation > max_seq_len:
        keep_prompt = max_seq_len - max_continuation
        if keep_prompt <= 0:
            raise ValueError("Continuation is longer than --max-seq-len")
        prompt_ids = prompt_ids[-keep_prompt:]
    return prompt_ids, continuation_ids


@torch.inference_mode()
def continuation_scores(
    model,
    tokenizer,
    prompt: str,
    continuations: list[str],
    max_seq_len: int,
) -> list[float]:
    if not continuations:
        raise ValueError("Likelihood evaluation requires continuations")
    prompt_ids, continuation_ids = tokenize_prompt_and_continuations(
        tokenizer,
        prompt,
        continuations,
        max_seq_len,
    )
    device = next(model.parameters()).device

    if all(ids.numel() == 1 for ids in continuation_ids):
        logits = model.forward(prompt_ids.unsqueeze(0).to(device), start_pos=0)
        logprobs = F.log_softmax(logits[0, -1], dim=-1)
        return [float(logprobs[int(ids[0])].item()) for ids in continuation_ids]

    scores = []
    for ids in continuation_ids:
        input_ids = torch.cat([prompt_ids, ids]).unsqueeze(0).to(device)
        logits = model.forward(input_ids[:, :-1], start_pos=0)
        continuation_start = prompt_ids.numel()
        score_logits = logits[:, continuation_start - 1 :, :]
        targets = input_ids[:, continuation_start:]
        token_logprobs = -F.cross_entropy(
            score_logits.reshape(-1, score_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        scores.append(float(token_logprobs.mean().item()))
    return scores


def extract_mmlu_pro_answer(text: str) -> tuple[str | None, str]:
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1), "answer-is"

    match = re.search(r".*[aA]nswer:\s*([A-J])", text, re.DOTALL)
    if match:
        return match.group(1), "answer-colon"

    matches = re.findall(r"\b[A-J]\b", text)
    if matches:
        return matches[-1], "last-letter"
    return None, "unparsed"


def extract_kmmlu_pro_answer(
    text: str,
    language: str,
) -> tuple[str | None, str]:
    if language == "ko":
        strict_pattern = r"정답[^A-E]*:[^A-E]*([A-E])"
        flexible_pattern = r"정답[^A-E]*([A-E])|([A-E])\)"
    else:
        strict_pattern = r"Answer[^A-E]*:[^A-E]*([A-E])"
        flexible_pattern = r"Answer[^A-E]*([A-E])|([A-E])\)"

    strict_matches = re.findall(strict_pattern, text, flags=re.IGNORECASE)
    if strict_matches:
        return strict_matches[-1].upper(), "strict"

    flexible_matches = re.findall(flexible_pattern, text, flags=re.IGNORECASE)
    if flexible_matches:
        left, right = flexible_matches[-1]
        return (left or right).upper(), "flexible"
    return None, "unparsed"


def raw_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    return tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()


def chat_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return inputs["input_ids"][0].tolist()


def select_generation_prompt_tokens(
    generator,
    example: Example,
    experiment_name: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> list[int]:
    tokenizer = generator.tokenizer
    prompt_variants = example.prompt_variants or [example.prompt]
    reserve = min(max_new_tokens, max_seq_len - 1)

    for prompt in prompt_variants:
        if experiment_name == "MMLU-Pro":
            prompt_tokens = raw_prompt_tokens(tokenizer, prompt)
        else:
            prompt_tokens = chat_prompt_tokens(tokenizer, prompt)
        if len(prompt_tokens) <= max_seq_len - reserve:
            return prompt_tokens

    raise ValueError(
        f"{experiment_name} prompt does not fit in max_seq_len={max_seq_len} "
        f"with max_new_tokens={max_new_tokens}"
    )


def generate_response(
    generator,
    example: Example,
    experiment_name: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    prompt_tokens = select_generation_prompt_tokens(
        generator,
        example,
        experiment_name,
        max_seq_len,
        max_new_tokens,
    )
    generation_limit = min(max_new_tokens, max_seq_len - len(prompt_tokens))
    generated_tokens, _ = generator.generate(
        prompt_tokens=[prompt_tokens],
        max_gen_len=generation_limit,
        temperature=0.0,
        top_p=1.0,
    )
    return generator.tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    ).strip()


def evaluate_generation(
    generator,
    examples: list[Example],
    experiment_name: str,
    model_id: str,
    max_seq_len: int,
    max_new_tokens: int,
    kmmlu_prompt_language: str,
) -> float:
    correct = 0
    parsed = 0
    parse_methods: dict[str, int] = defaultdict(int)
    fallback_rng = random.Random(MMLU_PRO_FALLBACK_SEED)

    for example in tqdm(examples, desc=f"{experiment_name} / {model_id}"):
        response = generate_response(
            generator,
            example,
            experiment_name,
            max_seq_len,
            max_new_tokens,
        )
        if experiment_name == "MMLU-Pro":
            prediction, method = extract_mmlu_pro_answer(response)
        else:
            prediction, method = extract_kmmlu_pro_answer(
                response,
                kmmlu_prompt_language,
            )

        if prediction is not None:
            parsed += 1
        elif experiment_name == "MMLU-Pro":
            prediction = fallback_rng.choice(example.answer_labels)
            method = "random-fallback"

        parse_methods[method] += 1
        target = example.answer_labels[example.answer]
        correct += int(prediction == target)

    method_report = " ".join(
        f"{method}={count}" for method, count in sorted(parse_methods.items())
    )
    print(
        f"{experiment_name} / {model_id}: parsed={parsed}/{len(examples)} "
        f"{method_report}"
    )
    return correct / len(examples)


def evaluate_likelihood(
    generator,
    examples: list[Example],
    experiment_name: str,
    model_id: str,
    max_seq_len: int,
) -> float:
    correct = 0
    for example in tqdm(examples, desc=f"{experiment_name} / {model_id}"):
        scores = continuation_scores(
            generator.model,
            generator.tokenizer,
            example.prompt,
            example.continuations or [],
            max_seq_len,
        )
        prediction = max(range(len(scores)), key=scores.__getitem__)
        correct += int(prediction == example.answer)
    return correct / len(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=EXPERIMENTS,
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--kmmlu-prompt-language",
        choices=("ko", "en"),
        default="ko",
    )
    parser.add_argument(
        "--torch-dtype",
        type=parse_torch_dtype,
        default=torch.bfloat16,
        choices=tuple(DTYPE_MAP.values()),
        metavar="{" + ",".join(DTYPE_MAP) + "}",
    )
    parser.add_argument(
        "--weight-format",
        choices=("auto", "bin", "pth"),
        default="auto",
        help=(
            "Local weight format. auto prefers model.bin over model.pth. "
            "HF model IDs always use safetensors."
        ),
    )
    args = parser.parse_args()

    experiments = load_experiments(
        args.experiments,
        args.num_samples,
        args.seed,
        args.kmmlu_prompt_language,
    )
    results: dict[str, dict[str, float]] = {
        experiment: {} for experiment in args.experiments
    }

    if experiments:
        for model_id in args.model_id:
            print(f"Loading model: {model_id}")
            generator = Llama.build(
                model_id=model_id,
                max_seq_len=args.max_seq_len,
                max_batch_size=1,
                dtype=args.torch_dtype,
                weight_format=args.weight_format,
            )
            for experiment_name, examples in experiments.items():
                if experiment_name in GENERATION_EXPERIMENTS:
                    accuracy = evaluate_generation(
                        generator,
                        examples,
                        experiment_name,
                        model_id,
                        args.max_seq_len,
                        args.max_new_tokens,
                        args.kmmlu_prompt_language,
                    )
                else:
                    accuracy = evaluate_likelihood(
                        generator,
                        examples,
                        experiment_name,
                        model_id,
                        args.max_seq_len,
                    )
                results[experiment_name][model_id] = accuracy

            del generator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("experiment," + ",".join(args.model_id))
    for experiment_name in args.experiments:
        if experiment_name in experiments:
            values = [
                f"{results[experiment_name][model_id] * 100:.6f}"
                for model_id in args.model_id
            ]
        else:
            values = ["NA"] * len(args.model_id)
        print(experiment_name + "," + ",".join(values))


if __name__ == "__main__":
    main()
