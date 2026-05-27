#!/usr/bin/env python3

import argparse
import sys
import time

import torch

from src.generate import Llama, MODEL_ID, sample_top_p


def visible_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

STREAM_TOKENS_PER_SEC = 50.0


def parse_torch_dtype(value: str) -> torch.dtype:
    try:
        return DTYPE_MAP[value]
    except KeyError as exc:
        choices = ", ".join(DTYPE_MAP)
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype: {value}. Choose from: {choices}") from exc


@torch.inference_mode()
def stream_generate_text(
    generator: Llama,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokens_per_sec: float,
) -> list[int]:
    model = generator.model
    tokenizer = generator.tokenizer
    device = next(model.parameters()).device

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
    prev_pos = 0
    generated_tokens: list[int] = []
    rendered_text = ""
    token_interval = 1.0 / tokens_per_sec
    stream_start = time.perf_counter()

    for step in range(max_new_tokens):
        cur_pos = prompt.shape[1]
        logits = model.forward(prompt[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

        token_id = int(next_token.item())
        if token_id == eos_id:
            break

        generated_tokens.append(token_id)
        prompt = torch.cat([prompt, next_token], dim=1)
        prev_pos = cur_pos

        updated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        delta = updated_text[len(rendered_text) :]
        if delta:
            print(delta, end="", flush=True)
            rendered_text = updated_text

        target_elapsed = (step + 1) * token_interval
        remaining = target_elapsed - (time.perf_counter() - stream_start)
        if remaining > 0:
            time.sleep(remaining)

    if rendered_text and not rendered_text.endswith("\n"):
        print()
    elif not rendered_text:
        print("", flush=True)

    return generated_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="자기소개해봐.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        type=parse_torch_dtype,
        default=torch.bfloat16,
        choices=tuple(DTYPE_MAP.values()),
        metavar="{" + ",".join(DTYPE_MAP) + "}",
        help="Torch dtype to use when loading and running the model. Default: bfloat16",
    )
    args = parser.parse_args()

    generator = Llama.build(
        model_id=args.model_id,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
        dtype=args.torch_dtype,
    )

    tokenizer = generator.tokenizer
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=args.reasoning,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=args.reasoning,
    )
    input_ids = inputs["input_ids"][0].tolist()

    generated_tokens = stream_generate_text(
        generator=generator,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tokens_per_sec=STREAM_TOKENS_PER_SEC,
    )
    


if __name__ == "__main__":
    main()
