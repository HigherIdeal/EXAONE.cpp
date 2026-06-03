#!/usr/bin/env python3

import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.generate import MODEL_ID


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
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype: {value}. Choose from: {choices}") from exc


def sample_start_positions(total_tokens: int, seq_len: int, num_samples: int, seed: int) -> list[int]:
    max_start = total_tokens - seq_len - 1
    if max_start < 0:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}. total_tokens={total_tokens}")

    generator = torch.Generator().manual_seed(seed)
    population = max_start + 1
    if num_samples <= population:
        return torch.randperm(population, generator=generator)[:num_samples].tolist()
    return torch.randint(0, population, (num_samples,), generator=generator).tolist()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--split", default="test")
    parser.add_argument("--quant", action="store_true", help="Use the quantized model path under python/quant.")
    parser.add_argument("--partial_quant", "--partial-quant", action="store_true", help="Quantize only transformer layer 0.")
    parser.add_argument("--quant-block-size", type=int, default=128)
    parser.add_argument("--target-module", default=None, help="Quantize one exact module, e.g. layers.0.attention.wq.")
    parser.add_argument("--target-projections", default=None, help="Comma-separated projection names, e.g. wq,wk,wv,wo.")
    parser.add_argument("--exclude-projections", default=None, help="Comma-separated projection names to skip, e.g. wd.")
    parser.add_argument("--calib-seq-len", type=int, default=2048)
    parser.add_argument("--calib-samples", type=int, default=8)
    parser.add_argument("--calib-seed", type=int, default=99)
    parser.add_argument("--calib-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=tuple(DTYPE_MAP),
        metavar="{" + ",".join(DTYPE_MAP) + "}",
        help="Torch dtype to use when loading and running the model. Default: %(default)s",
    )
    args = parser.parse_args()
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    if args.partial_quant or args.target_module or args.target_projections or args.exclude_projections:
        args.quant = True

    if args.quant:
        from quant.generate import Llama
    else:
        from src.generate import Llama

    build_kwargs = {
        "model_id": args.model_id,
        "max_seq_len": args.max_seq_len,
        "max_batch_size": 1,
        "dtype": torch_dtype,
    }
    if args.quant:
        build_kwargs["quant_block_size"] = args.quant_block_size
        build_kwargs["partial_quant"] = args.partial_quant
        build_kwargs["target_module"] = args.target_module
        build_kwargs["target_projections"] = args.target_projections
        build_kwargs["exclude_projections"] = args.exclude_projections
        build_kwargs["calib_seq_len"] = args.calib_seq_len
        build_kwargs["calib_samples"] = args.calib_samples
        build_kwargs["calib_seed"] = args.calib_seed
        build_kwargs["calib_max_tokens"] = args.calib_max_tokens

    generator = Llama.build(**build_kwargs)

    tokenizer = generator.tokenizer
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_tokens = encoded["input_ids"][0]

    starts = sample_start_positions(
        total_tokens=all_tokens.size(0),
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    device = next(generator.model.parameters()).device
    nlls: list[float] = []

    for start in tqdm(starts, desc="PPL samples", total=len(starts)):
        sample = all_tokens[start : start + args.seq_len + 1].unsqueeze(0).to(device)
        nlls.append(compute_sample_nll(generator.model, sample))

    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)

    print(f"dataset=wikitext2 split={args.split}")
    print(
        f"seq_len={args.seq_len} seed={args.seed} num_samples={args.num_samples} "
        f"torch_dtype={args.torch_dtype} quant={args.quant} partial_quant={args.partial_quant}"
    )
    if args.quant:
        print(
            f"quant_block_size={args.quant_block_size} calib_seq_len={args.calib_seq_len} "
            f"calib_samples={args.calib_samples} calib_max_tokens={args.calib_max_tokens} "
            f"target_module={args.target_module} target_projections={args.target_projections} "
            f"exclude_projections={args.exclude_projections}"
        )
    print(f"avg_nll={avg_nll:.6f}")
    print(f"ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
