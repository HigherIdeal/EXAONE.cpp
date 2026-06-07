#!/usr/bin/env python3

import argparse
import gc
import math
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.generate import Llama


DEFAULT_MODELS = (
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "EXAONE-4.0-1.2B",
)
DEFAULT_DATASETS = ("wikitext2", "c4", "ptb")

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_torch_dtype(value: str) -> torch.dtype:
    try:
        return DTYPE_MAP[value]
    except KeyError as exc:
        choices = ", ".join(DTYPE_MAP)
        raise argparse.ArgumentTypeError(
            f"Unsupported torch dtype: {value}. Choose from: {choices}"
        ) from exc


def sample_start_positions(
    total_tokens: int,
    seq_len: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    max_start = total_tokens - seq_len - 1
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens for seq_len={seq_len}. total_tokens={total_tokens}"
        )

    generator = torch.Generator().manual_seed(seed)
    population = max_start + 1
    if num_samples <= population:
        return torch.randperm(population, generator=generator)[:num_samples].tolist()
    return torch.randint(0, population, (num_samples,), generator=generator).tolist()


def samples_from_text(
    tokenizer,
    text: str,
    seq_len: int,
    num_samples: int,
    seed: int,
) -> list[torch.Tensor]:
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    starts = sample_start_positions(tokens.numel(), seq_len, num_samples, seed)
    return [tokens[start : start + seq_len + 1].clone() for start in starts]


def load_wikitext2_samples(
    tokenizer,
    seq_len: int,
    num_samples: int,
    seed: int,
) -> list[torch.Tensor]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    return samples_from_text(tokenizer, text, seq_len, num_samples, seed)


def load_ptb_samples(
    tokenizer,
    seq_len: int,
    num_samples: int,
    seed: int,
) -> list[torch.Tensor]:
    dataset = load_dataset(
        "ptb-text-only/ptb_text_only",
        "penn_treebank",
        split="test",
    )
    text = "\n".join(row for row in dataset["sentence"] if row.strip())
    return samples_from_text(tokenizer, text, seq_len, num_samples, seed)


def load_c4_samples(
    tokenizer,
    seq_len: int,
    num_samples: int,
    seed: int,
    shuffle_buffer: int,
) -> list[torch.Tensor]:
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    sample_size = seq_len + 1
    token_buffer: list[int] = []
    samples: list[torch.Tensor] = []
    cursor = 0

    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue

        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if tokenizer.eos_token_id is not None:
            token_buffer.append(tokenizer.eos_token_id)

        while len(token_buffer) - cursor >= sample_size:
            samples.append(torch.tensor(token_buffer[cursor : cursor + sample_size]))
            cursor += sample_size
            if len(samples) == num_samples:
                return samples

        if cursor >= sample_size * 16:
            token_buffer = token_buffer[cursor:]
            cursor = 0

    raise RuntimeError(f"C4 stream ended after collecting only {len(samples)} samples")


def prepare_samples(
    tokenizer,
    dataset_names: list[str],
    seq_len: int,
    num_samples: int,
    seed: int,
    c4_shuffle_buffer: int,
) -> dict[str, list[torch.Tensor]]:
    samples = {}
    for dataset_name in dataset_names:
        print(f"Preparing {dataset_name}: seq_len={seq_len} samples={num_samples}")
        if dataset_name == "wikitext2":
            samples[dataset_name] = load_wikitext2_samples(
                tokenizer, seq_len, num_samples, seed
            )
        elif dataset_name == "c4":
            samples[dataset_name] = load_c4_samples(
                tokenizer,
                seq_len,
                num_samples,
                seed,
                c4_shuffle_buffer,
            )
        elif dataset_name == "ptb":
            samples[dataset_name] = load_ptb_samples(
                tokenizer, seq_len, num_samples, seed
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return samples


@torch.inference_mode()
def compute_sample_nll(model, input_ids: torch.Tensor) -> float:
    logits = model.forward(input_ids[:, :-1], start_pos=0)
    targets = input_ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(loss.item())


def evaluate_dataset(model, samples: list[torch.Tensor], description: str) -> tuple[float, float]:
    device = next(model.parameters()).device
    nlls = []
    for sample in tqdm(samples, desc=description, total=len(samples)):
        nlls.append(compute_sample_nll(model, sample.unsqueeze(0).to(device)))

    avg_nll = sum(nlls) / len(nlls)
    return avg_nll, math.exp(avg_nll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--c4-shuffle-buffer", type=int, default=10_000)
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

    if args.seq_len > args.max_seq_len:
        parser.error("--seq-len cannot exceed --max-seq-len")

    results: dict[str, dict[str, tuple[float, float]]] = {}
    prepared_samples = None

    for model_id in args.model_id:
        print()
        print(f"Loading model: {model_id}")
        generator = Llama.build(
            model_id=model_id,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
            dtype=args.torch_dtype,
            weight_format=args.weight_format,
        )

        if prepared_samples is None:
            prepared_samples = prepare_samples(
                generator.tokenizer,
                args.datasets,
                args.seq_len,
                args.num_samples,
                args.seed,
                args.c4_shuffle_buffer,
            )

        model_results = {}
        for dataset_name in args.datasets:
            avg_nll, ppl = evaluate_dataset(
                generator.model,
                prepared_samples[dataset_name],
                description=f"{model_id} / {dataset_name}",
            )
            model_results[dataset_name] = (avg_nll, ppl)
            print(
                f"model={model_id} dataset={dataset_name} "
                f"avg_nll={avg_nll:.6f} ppl={ppl:.6f}"
            )

        results[model_id] = model_results
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print(
        f"seq_len={args.seq_len} num_samples={args.num_samples} "
        f"seed={args.seed} torch_dtype={args.torch_dtype}"
    )
    print(f"{'model':34} {'dataset':12} {'avg_nll':>12} {'ppl':>12}")
    print("-" * 74)
    for model_id in args.model_id:
        for dataset_name in args.datasets:
            avg_nll, ppl = results[model_id][dataset_name]
            print(f"{model_id:34} {dataset_name:12} {avg_nll:12.6f} {ppl:12.6f}")


if __name__ == "__main__":
    main()
