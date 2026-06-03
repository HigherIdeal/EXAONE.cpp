#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm
from transformers.utils.hub import cached_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.generate import HF_MODEL_ID  # noqa: E402


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


def resolve_hf_file(model_id: str, filename: str) -> Path:
    return Path(cached_file(model_id, filename))


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
        index_path = resolve_hf_file(model_id, "model.safetensors.index.json")
    except Exception:
        index_path = None

    if index_path is None:
        weights_path = resolve_hf_file(model_id, "model.safetensors")
        state_dict: dict[str, torch.Tensor] = {}
        for path in tqdm([weights_path], desc="Load safetensors"):
            state_dict.update(load_safetensors_file(path))
        return state_dict

    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    state_dict: dict[str, torch.Tensor] = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in tqdm(shard_names, desc="Load safetensor shards"):
        state_dict.update(load_safetensors_file(resolve_hf_file(model_id, shard_name)))
    return state_dict


def convert_hf_state_dict(state_dict: dict[str, torch.Tensor], n_layers: int) -> dict[str, torch.Tensor]:
    checkpoint = {
        "tok_embeddings.weight": state_dict["model.embed_tokens.weight"],
        "norm.weight": state_dict["model.norm.weight"],
    }

    for layer_id in tqdm(range(n_layers), desc="Convert layer keys"):
        hf = f"model.layers.{layer_id}"
        dst = f"layers.{layer_id}"
        checkpoint[f"{dst}.attention.wq"] = state_dict[f"{hf}.self_attn.q_proj.weight"]
        checkpoint[f"{dst}.attention.wk"] = state_dict[f"{hf}.self_attn.k_proj.weight"]
        checkpoint[f"{dst}.attention.wv"] = state_dict[f"{hf}.self_attn.v_proj.weight"]
        checkpoint[f"{dst}.attention.wo"] = state_dict[f"{hf}.self_attn.o_proj.weight"]
        checkpoint[f"{dst}.attention.q_norm.weight"] = state_dict[f"{hf}.self_attn.q_norm.weight"]
        checkpoint[f"{dst}.attention.k_norm.weight"] = state_dict[f"{hf}.self_attn.k_norm.weight"]
        checkpoint[f"{dst}.feed_forward.wg"] = state_dict[f"{hf}.mlp.gate_proj.weight"]
        checkpoint[f"{dst}.feed_forward.wd"] = state_dict[f"{hf}.mlp.down_proj.weight"]
        checkpoint[f"{dst}.feed_forward.wu"] = state_dict[f"{hf}.mlp.up_proj.weight"]
        checkpoint[f"{dst}.attention_norm.weight"] = state_dict[f"{hf}.post_attention_layernorm.weight"]
        checkpoint[f"{dst}.ffn_norm.weight"] = state_dict[f"{hf}.post_feedforward_layernorm.weight"]

    return checkpoint


def infer_n_layers(config_path: Path) -> int:
    with open(config_path, encoding="utf-8") as handle:
        return int(json.load(handle)["num_hidden_layers"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-id", default=HF_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "weight" / "exaone4-1.2b")
    parser.add_argument("--output-name", default="model.pth")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    copy_model_files(args.hf_model_id, output_dir)

    n_layers = infer_n_layers(output_dir / "config.json")
    hf_state_dict = load_hf_state_dict(args.hf_model_id)
    checkpoint = convert_hf_state_dict(hf_state_dict, n_layers)

    output_path = output_dir / args.output_name
    print(f"Saving {output_path}")
    torch.save(checkpoint, output_path)
    print("Done")


if __name__ == "__main__":
    main()
