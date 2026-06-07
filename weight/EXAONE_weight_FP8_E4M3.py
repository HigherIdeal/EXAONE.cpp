#!/usr/bin/env python3

import argparse
import gc
import json
import os
import re
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm
from transformers.utils.hub import cached_file


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_CHUNK_ELEMENTS = 4 * 1024 * 1024

ALIGNMENT = 64
EXPECTED_LAYERS = 30
EXPECTED_WEIGHTS = 332

# Little-endian header:
#   uint32 weight_count
#   repeated weight_count times:
#       uint32 weight_index
#       uint64 absolute_file_offset_bytes
COUNT_STRUCT = struct.Struct("<I")
ENTRY_STRUCT = struct.Struct("<IQ")

TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

LAYER_SPECS = (
    ("layers.{layer}.attention.wq", "fp8", True),
    ("layers.{layer}.attention.q_norm.weight", "bf16", False),
    ("layers.{layer}.attention.wk", "fp8", True),
    ("layers.{layer}.attention.k_norm.weight", "bf16", False),
    ("layers.{layer}.attention.wv", "fp8", True),
    ("layers.{layer}.attention.wo", "fp8", True),
    ("layers.{layer}.attention_norm.weight", "bf16", False),
    ("layers.{layer}.feed_forward.wu", "fp8", True),
    ("layers.{layer}.feed_forward.wg", "fp8", True),
    ("layers.{layer}.feed_forward.wd", "fp8", True),
    ("layers.{layer}.ffn_norm.weight", "bf16", False),
)


@dataclass(frozen=True)
class WeightSpec:
    index: int
    name: str
    storage: str
    transpose: bool


@dataclass(frozen=True)
class PackedWeight:
    spec: WeightSpec
    offset: int
    size_bytes: int
    source_shape: tuple[int, ...]
    stored_shape: tuple[int, ...]


def default_output_dir(model_id: str) -> Path:
    model_name = model_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not model_name or model_name in {".", ".."}:
        raise ValueError(f"Cannot derive output directory from: {model_id!r}")
    return REPO_ROOT / "weight" / model_name


def resolve_hf_file(model_id: str, filename: str) -> Path:
    resolved = cached_file(model_id, filename)
    if resolved is None:
        raise FileNotFoundError(f"{filename} was not found in {model_id}")
    return Path(resolved)


def copy_model_files(model_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in tqdm(TOKENIZER_FILES, desc="Copy tokenizer/config files"):
        try:
            source = resolve_hf_file(model_id, filename)
        except Exception:
            continue
        shutil.copy2(source, output_dir / filename)


def load_hf_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    try:
        index_path = resolve_hf_file(
            model_id,
            "model.safetensors.index.json",
        )
    except Exception:
        index_path = None

    if index_path is None:
        paths = [resolve_hf_file(model_id, "model.safetensors")]
    else:
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        paths = [
            resolve_hf_file(model_id, shard_name)
            for shard_name in sorted(set(weight_map.values()))
        ]

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm(paths, desc="Load safetensors"):
        state_dict.update(load_safetensors_file(path))
    return state_dict


def convert_hf_state_dict(
    state_dict: dict[str, torch.Tensor],
    n_layers: int,
) -> dict[str, torch.Tensor]:
    checkpoint = {
        "tok_embeddings.weight": state_dict["model.embed_tokens.weight"],
        "norm.weight": state_dict["model.norm.weight"],
    }

    for layer in tqdm(range(n_layers), desc="Convert layer keys"):
        hf = f"model.layers.{layer}"
        dst = f"layers.{layer}"
        checkpoint[f"{dst}.attention.wq"] = state_dict[
            f"{hf}.self_attn.q_proj.weight"
        ]
        checkpoint[f"{dst}.attention.wk"] = state_dict[
            f"{hf}.self_attn.k_proj.weight"
        ]
        checkpoint[f"{dst}.attention.wv"] = state_dict[
            f"{hf}.self_attn.v_proj.weight"
        ]
        checkpoint[f"{dst}.attention.wo"] = state_dict[
            f"{hf}.self_attn.o_proj.weight"
        ]
        checkpoint[f"{dst}.attention.q_norm.weight"] = state_dict[
            f"{hf}.self_attn.q_norm.weight"
        ]
        checkpoint[f"{dst}.attention.k_norm.weight"] = state_dict[
            f"{hf}.self_attn.k_norm.weight"
        ]
        checkpoint[f"{dst}.feed_forward.wg"] = state_dict[
            f"{hf}.mlp.gate_proj.weight"
        ]
        checkpoint[f"{dst}.feed_forward.wd"] = state_dict[
            f"{hf}.mlp.down_proj.weight"
        ]
        checkpoint[f"{dst}.feed_forward.wu"] = state_dict[
            f"{hf}.mlp.up_proj.weight"
        ]
        checkpoint[f"{dst}.attention_norm.weight"] = state_dict[
            f"{hf}.post_attention_layernorm.weight"
        ]
        checkpoint[f"{dst}.ffn_norm.weight"] = state_dict[
            f"{hf}.post_feedforward_layernorm.weight"
        ]
    return checkpoint


def build_specs() -> list[WeightSpec]:
    specs = [
        WeightSpec(
            index=0,
            name="tok_embeddings.weight",
            storage="fp8",
            transpose=False,
        )
    ]

    index = 1
    for layer in range(EXPECTED_LAYERS):
        for name_template, storage, transpose in LAYER_SPECS:
            specs.append(
                WeightSpec(
                    index=index,
                    name=name_template.format(layer=layer),
                    storage=storage,
                    transpose=transpose,
                )
            )
            index += 1

    specs.append(
        WeightSpec(
            index=index,
            name="norm.weight",
            storage="bf16",
            transpose=False,
        )
    )

    if len(specs) != EXPECTED_WEIGHTS or specs[-1].index != 331:
        raise AssertionError("Internal layout must contain weights 0..331")
    return specs


def transform_tensor(
    tensor: torch.Tensor,
    chunk_elements: int,
) -> tuple[torch.Tensor, dict]:
    source_flat = tensor.detach().reshape(-1)
    output_flat = torch.empty(source_flat.shape, dtype=torch.bfloat16)
    exponent_histogram: Counter[int] = Counter()
    zero_or_subnormal = 0
    high_exponent_clamped = 0

    for start in range(0, source_flat.numel(), chunk_elements):
        source = source_flat[start : start + chunk_elements]

        source_bf16 = source.to(torch.bfloat16)
        source_bits = source_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
        sign_bits = source_bits & 0x8000
        fraction_bits = source_bits & 0x0070

        fp16 = source.to(torch.float16)
        fp16_bits = fp16.view(torch.int16).to(torch.int32) & 0xFFFF
        fp16_exponent = (fp16_bits >> 10) & 0x1F

        zero_mask = fp16_exponent == 0
        high_mask = fp16_exponent > 0x0F
        zero_or_subnormal += int(zero_mask.sum().item())
        high_exponent_clamped += int(high_mask.sum().item())

        truncated_exponent = fp16_exponent.clamp(max=0x0F)
        bf16_exponent = 0x70 | truncated_exponent
        output_bits = sign_bits | (bf16_exponent << 7) | fraction_bits
        output_flat[start : start + source.numel()] = (
            output_bits.to(torch.int16).view(torch.bfloat16)
        )

        unique, counts = torch.unique(truncated_exponent, return_counts=True)
        exponent_histogram.update(
            {
                int(field) - 15: int(count)
                for field, count in zip(unique.tolist(), counts.tolist())
            }
        )

    return output_flat.reshape(tensor.shape), {
        "histogram": exponent_histogram,
        "zero_or_subnormal": zero_or_subnormal,
        "high_exponent_clamped": high_exponent_clamped,
    }


def truncate_checkpoint(
    checkpoint: dict[str, torch.Tensor],
    specs: list[WeightSpec],
    chunk_elements: int,
) -> None:
    global_histogram: Counter[int] = Counter()
    zero_or_subnormal = 0
    high_exponent_clamped = 0

    for spec in tqdm(specs, desc="Truncate weights"):
        tensor = checkpoint[spec.name]
        if spec.storage == "bf16":
            checkpoint[spec.name] = tensor.to(torch.bfloat16).contiguous()
            continue

        transformed, report = transform_tensor(tensor, chunk_elements)
        checkpoint[spec.name] = transformed.contiguous()
        global_histogram.update(report["histogram"])
        zero_or_subnormal += report["zero_or_subnormal"]
        high_exponent_clamped += report["high_exponent_clamped"]

    print("BF16 intermediate=S|0111xxxx|xxx0000")
    print("FP8 packed=S|xxxx|xxx")
    print(f"zero_or_subnormal_mapped_to_2^-15={zero_or_subnormal:,}")
    print(f"exponents_above_0_clamped_to_2^0={high_exponent_clamped:,}")
    if global_histogram:
        print(
            "stored_exponent_range="
            f"2^{min(global_histogram)}..2^{max(global_histogram)}"
        )


def validate_checkpoint(
    checkpoint: dict[str, torch.Tensor],
    specs: list[WeightSpec],
) -> None:
    expected_names = {spec.name for spec in specs}
    missing = sorted(expected_names - checkpoint.keys())
    if missing:
        raise KeyError(f"Missing converted tensors: {missing}")

    layer_ids = {
        int(match.group(1))
        for name in checkpoint
        if (match := re.match(r"layers\.(\d+)\.", name))
    }
    if layer_ids != set(range(EXPECTED_LAYERS)):
        raise ValueError(
            f"Expected layers 0..29, found {sorted(layer_ids)}"
        )

    for spec in specs:
        tensor = checkpoint[spec.name]
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"{spec.name} must be bfloat16, found {tensor.dtype}"
            )
        if spec.transpose and tensor.ndim != 2:
            raise ValueError(
                f"{spec.name} must be 2D, shape={tuple(tensor.shape)}"
            )
        if spec.storage == "bf16" and tensor.ndim != 1:
            raise ValueError(
                f"{spec.name} must be a 1D norm, shape={tuple(tensor.shape)}"
            )


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def build_layout(
    checkpoint: dict[str, torch.Tensor],
    specs: list[WeightSpec],
) -> tuple[list[PackedWeight], int]:
    header_size = COUNT_STRUCT.size + len(specs) * ENTRY_STRUCT.size
    next_offset = align_up(header_size)
    layout = []

    for spec in specs:
        tensor = checkpoint[spec.name]
        source_shape = tuple(tensor.shape)
        stored_shape = (
            (source_shape[1], source_shape[0])
            if spec.transpose
            else source_shape
        )
        size_bytes = tensor.numel() * (1 if spec.storage == "fp8" else 2)
        layout.append(
            PackedWeight(
                spec=spec,
                offset=next_offset,
                size_bytes=size_bytes,
                source_shape=source_shape,
                stored_shape=stored_shape,
            )
        )
        next_offset = align_up(next_offset + size_bytes)
    return layout, next_offset


def write_zeros(handle, count: int) -> None:
    if count > 0:
        handle.write(b"\x00" * count)


def write_header(handle, layout: list[PackedWeight]) -> None:
    handle.write(COUNT_STRUCT.pack(len(layout)))
    for packed in layout:
        handle.write(ENTRY_STRUCT.pack(packed.spec.index, packed.offset))
    write_zeros(handle, align_up(handle.tell()) - handle.tell())


def iter_flat_chunks(
    tensor: torch.Tensor,
    transpose: bool,
    chunk_elements: int,
):
    stored = tensor.t().contiguous() if transpose else tensor.contiguous()
    flat = stored.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        yield flat[start : start + chunk_elements]


def pack_fp8_chunk(source: torch.Tensor, name: str) -> torch.Tensor:
    source_bits = source.view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (source_bits >> 7) & 0xFF
    fraction = source_bits & 0x7F

    valid = ((exponent & 0xF0) == 0x70) & ((fraction & 0x0F) == 0)
    if not bool(torch.all(valid)):
        invalid = int((~valid).sum().item())
        raise ValueError(
            f"{name} contains {invalid} values outside "
            "S|0111xxxx|xxx0000"
        )

    return (
        ((source_bits >> 8) & 0x80)
        | ((exponent & 0x0F) << 3)
        | ((fraction >> 4) & 0x07)
    ).to(torch.uint8)


def write_fp8_tensor(
    handle,
    tensor: torch.Tensor,
    spec: WeightSpec,
    chunk_elements: int,
) -> None:
    for source in iter_flat_chunks(
        tensor,
        spec.transpose,
        chunk_elements,
    ):
        packed = pack_fp8_chunk(source, spec.name)
        handle.write(packed.numpy().tobytes(order="C"))


def write_bf16_tensor(handle, tensor: torch.Tensor) -> None:
    if struct.pack("=H", 1) != struct.pack("<H", 1):
        raise RuntimeError("BF16 serialization requires a little-endian host")
    raw = tensor.contiguous().view(torch.uint8)
    handle.write(raw.numpy().tobytes(order="C"))


def write_payloads(
    handle,
    checkpoint: dict[str, torch.Tensor],
    layout: list[PackedWeight],
    chunk_elements: int,
) -> None:
    for packed in tqdm(layout, desc="Write model.bin"):
        if handle.tell() > packed.offset:
            raise RuntimeError(
                f"Payload overlap at weight {packed.spec.index}"
            )
        write_zeros(handle, packed.offset - handle.tell())

        start = handle.tell()
        tensor = checkpoint[packed.spec.name]
        if packed.spec.storage == "fp8":
            write_fp8_tensor(
                handle,
                tensor,
                packed.spec,
                chunk_elements,
            )
        else:
            write_bf16_tensor(handle, tensor)

        written = handle.tell() - start
        if written != packed.size_bytes:
            raise RuntimeError(
                f"Wrong size for {packed.spec.name}: "
                f"expected={packed.size_bytes}, written={written}"
            )


def write_model_bin(
    checkpoint: dict[str, torch.Tensor],
    specs: list[WeightSpec],
    output: Path,
    chunk_elements: int,
) -> tuple[list[PackedWeight], int]:
    validate_checkpoint(checkpoint, specs)
    layout, final_size = build_layout(checkpoint, specs)

    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            write_header(handle, layout)
            write_payloads(
                handle,
                checkpoint,
                layout,
                chunk_elements,
            )
            write_zeros(handle, final_size - handle.tell())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return layout, final_size


def save_checkpoint(
    checkpoint: dict[str, torch.Tensor],
    output: Path,
) -> None:
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(size) for size in shape)


def print_layout(
    layout: list[PackedWeight],
    final_size: int,
) -> None:
    print(
        f"{'idx':>3} {'tensor':52} {'type':5} {'T':1} "
        f"{'source':16} {'stored':16} {'offset':>12} {'bytes':>12}"
    )
    print("-" * 128)
    for packed in layout:
        print(
            f"{packed.spec.index:3d} "
            f"{packed.spec.name:52} "
            f"{packed.spec.storage:5} "
            f"{'Y' if packed.spec.transpose else 'N'} "
            f"{shape_text(packed.source_shape):16} "
            f"{shape_text(packed.stored_shape):16} "
            f"{packed.offset:12d} "
            f"{packed.size_bytes:12d}"
        )
    print("-" * 128)
    print(f"weight_count={len(layout)}")
    print(f"header_bytes={COUNT_STRUCT.size + len(layout) * ENTRY_STRUCT.size}")
    print(f"first_payload_offset={layout[0].offset}")
    print(f"final_file_bytes={final_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download EXAONE-4.0-1.2B and create the custom truncated "
            "BF16 checkpoint plus aligned E4M3 model.bin."
        )
    )
    parser.add_argument("--hf-model-id", default=HF_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: weight/<HF model name>",
    )
    parser.add_argument("--pth-name", default="model.pth")
    parser.add_argument("--bin-name", default="model.bin")
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=DEFAULT_CHUNK_ELEMENTS,
    )
    parser.add_argument(
        "--no-save-pth",
        action="store_true",
        help="Do not save the intermediate truncated BF16 model.pth.",
    )
    args = parser.parse_args()

    if args.chunk_elements <= 0:
        raise ValueError("--chunk-elements must be greater than zero")

    output_dir = (
        args.output_dir or default_output_dir(args.hf_model_id)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source={args.hf_model_id}")
    print(f"output_dir={output_dir}")

    print("[1/4] Download/copy tokenizer and config")
    copy_model_files(args.hf_model_id, output_dir)
    with (output_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    n_layers = int(config["num_hidden_layers"])
    if n_layers != EXPECTED_LAYERS:
        raise ValueError(
            f"This binary layout expects {EXPECTED_LAYERS} layers, "
            f"but config contains {n_layers}"
        )

    print("[2/4] Download and convert HF safetensors")
    hf_state_dict = load_hf_state_dict(args.hf_model_id)
    checkpoint = convert_hf_state_dict(hf_state_dict, n_layers)
    del hf_state_dict
    gc.collect()

    specs = build_specs()
    validate_checkpoint_keys = {spec.name for spec in specs}
    missing = sorted(validate_checkpoint_keys - checkpoint.keys())
    if missing:
        raise KeyError(f"Missing converted tensors: {missing}")

    print("[3/4] Truncate to custom E4M3-compatible BF16 layout")
    truncate_checkpoint(checkpoint, specs, args.chunk_elements)

    if not args.no_save_pth:
        pth_path = output_dir / args.pth_name
        save_checkpoint(checkpoint, pth_path)
        print(f"saved={pth_path}")

    print("[4/4] Write aligned model.bin")
    bin_path = output_dir / args.bin_name
    layout, final_size = write_model_bin(
        checkpoint,
        specs,
        bin_path,
        args.chunk_elements,
    )
    print(f"saved={bin_path}")
    print("header=<uint32 count> + 332 * <uint32 index, uint64 offset>")
    print("all payload offsets are absolute and 64-byte aligned")
    print_layout(layout, final_size)
    print("Done")


if __name__ == "__main__":
    main()
