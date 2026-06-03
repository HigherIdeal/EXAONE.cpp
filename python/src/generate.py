import time
import builtins
import io
import json
import locale
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

try:
    from .model import ModelArgs, Transformer
except ImportError:
    from model import ModelArgs, Transformer


HF_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_ID = os.path.join(REPO_ROOT, "weight", "exaone4-1.2b")
MODEL_PTH = "model.pth"


@contextmanager
def force_utf8_locale():
    # Transformers may read tokenizer sidecar files with the process default
    # text encoding. Force UTF-8 briefly so the same code path works on
    # Windows cp949 and on UTF-8-first Linux environments.
    original_getpreferredencoding = locale.getpreferredencoding
    original_getencoding = getattr(locale, "getencoding", None)

    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"
    if original_getencoding is not None:
        locale.getencoding = lambda: "utf-8"
    try:
        yield
    finally:
        locale.getpreferredencoding = original_getpreferredencoding
        if original_getencoding is not None:
            locale.getencoding = original_getencoding


@contextmanager
def force_utf8_chat_template_open():
    # Some tokenizer repos ship `chat_template.jinja` as UTF-8 text.
    # On Windows, Transformers may open it with the process default encoding
    # (for example cp949), which raises a decode error. Intercept only plain
    # text reads for that filename and leave every other file operation alone.
    original_open = builtins.open
    original_io_open = io.open

    def utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else None
        if (
            path is not None
            and "b" not in mode
            and encoding is None
            and os.path.basename(path) == "chat_template.jinja"
        ):
            encoding = "utf-8"
        return original_open(
            file,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    builtins.open = utf8_open
    io.open = utf8_open
    try:
        yield
    finally:
        builtins.open = original_open
        io.open = original_io_open


def convert_hf_state_dict(state_dict: dict[str, torch.Tensor], n_layers: int) -> dict[str, torch.Tensor]:
    checkpoint = {
        "tok_embeddings.weight": state_dict["model.embed_tokens.weight"],
        "norm.weight": state_dict["model.norm.weight"],
    }

    for layer_id in range(n_layers):
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


def is_converted_model_dir(model_id: str) -> bool:
    return os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, MODEL_PTH))


def resolve_runtime_model_id(model_id: str) -> str:
    if is_converted_model_dir(model_id):
        return model_id
    if os.path.abspath(model_id) == os.path.abspath(MODEL_ID):
        return HF_MODEL_ID
    return model_id


def resolve_model_file(model_id: str, filename: str) -> str:
    candidate = os.path.join(model_id, filename)
    if os.path.exists(candidate):
        return candidate
    return cached_file(model_id, filename)


def load_model_config(model_id: str) -> dict:
    config_path = resolve_model_file(model_id, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_hf_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    index_filename = "model.safetensors.index.json"
    try:
        index_path = resolve_model_file(model_id, index_filename)
    except Exception:
        index_path = None

    if index_path is None:
        weights_path = resolve_model_file(model_id, "model.safetensors")
        return load_safetensors_file(weights_path)

    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = resolve_model_file(model_id, shard_name)
        state_dict.update(load_safetensors_file(shard_path))
    return state_dict


def load_converted_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    checkpoint_path = os.path.join(model_id, MODEL_PTH)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_model_state_dict(model_id: str, n_layers: int) -> dict[str, torch.Tensor]:
    if is_converted_model_dir(model_id):
        return load_converted_state_dict(model_id)
    return convert_hf_state_dict(load_hf_state_dict(model_id), n_layers)


class Llama:
    @staticmethod
    def build(
        model_id: str = MODEL_ID,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> "Llama":
        start_time = time.time()
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        model_id = resolve_runtime_model_id(model_id)

        with force_utf8_locale(), force_utf8_chat_template_open():
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        config = load_model_config(model_id)
        model_args = ModelArgs(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            hidden_dim=config["intermediate_size"],
            norm_eps=config["rms_norm_eps"],
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_theta=config["rope_theta"],
            rope_factor=config["rope_scaling"]["factor"],
            rope_low_freq_factor=config["rope_scaling"]["low_freq_factor"],
            rope_high_freq_factor=config["rope_scaling"]["high_freq_factor"],
            rope_original_max_position_embeddings=config["rope_scaling"]["original_max_position_embeddings"],
        )

        model = Transformer(model_args).to(device=device, dtype=dtype)
        checkpoint = load_model_state_dict(model_id, model_args.n_layers)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        model.eval()
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        device = next(self.model.parameters()).device
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return out_tokens, out_logprobs if logprobs else None

    def chat_completion(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_gen_len: int = 128,
        reasoning: bool = False,
    ) -> List[str]:
        prompt_tokens = []
        for prompt in prompts:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=reasoning,
            )
            prompt_tokens.append(inputs["input_ids"][0].tolist())

        generation_tokens, _ = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return [self.tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in generation_tokens]


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
