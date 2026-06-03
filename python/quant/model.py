import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import sys
torch.set_printoptions(precision=8)

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 30
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 102400
    hidden_dim: int = 4096
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_len: int = 65536
    rope_theta: float = 1000000.0
    rope_factor: float = 16.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:     
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if not is_power_of_two(n):
        return x

    y = x.float()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, h * 2)
        left = y[..., :h]
        right = y[..., h:]
        y = torch.cat((left + right, left - right), dim=-1)
        y = y.reshape(*y.shape[:-2], -1)
        h *= 2
    return y / math.sqrt(n)


def haar_transform(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if not is_power_of_two(n):
        return x

    current = x.float()
    details = []
    scale = math.sqrt(2.0)
    while current.shape[-1] > 1:
        even = current[..., 0::2]
        odd = current[..., 1::2]
        current = (even + odd) / scale
        details.append((even - odd) / scale)
    return torch.cat([current, *reversed(details)], dim=-1)


def inverse_haar_transform(coeffs: torch.Tensor) -> torch.Tensor:
    n = coeffs.shape[-1]
    if not is_power_of_two(n):
        return coeffs

    current = coeffs[..., :1].float()
    offset = 1
    scale = math.sqrt(2.0)
    levels = int(math.log2(n))
    for level in range(levels):
        detail_len = 2**level
        detail = coeffs[..., offset : offset + detail_len].float()
        offset += detail_len
        expanded = torch.empty(*current.shape[:-1], detail_len * 2, device=coeffs.device, dtype=torch.float32)
        expanded[..., 0::2] = (current + detail) / scale
        expanded[..., 1::2] = (current - detail) / scale
        current = expanded
    return current


def apply_weight_transform(x: torch.Tensor, mode_id: int) -> torch.Tensor:
    if mode_id == 1:
        return hadamard_transform(x)
    if mode_id == 2:
        return haar_transform(x)
    return x.float()


def invert_weight_transform(x: torch.Tensor, mode_id: int) -> torch.Tensor:
    if mode_id == 1:
        return hadamard_transform(x)
    if mode_id == 2:
        return inverse_haar_transform(x)
    return x.float()


def refine_signs(
    coeffs: torch.Tensor,
    signs: torch.Tensor,
    h_diag: torch.Tensor,
    alpha: torch.Tensor,
    iters: int,
    max_flip_ratio: float,
) -> torch.Tensor:
    if iters <= 0 or max_flip_ratio <= 0:
        return signs

    max_flips = max(1, int(coeffs.shape[-1] * max_flip_ratio))
    for _ in range(iters):
        current_error = h_diag * (coeffs - alpha * signs).pow(2)
        flipped_error = h_diag * (coeffs + alpha * signs).pow(2)
        improvement = current_error - flipped_error
        candidates = improvement > 0
        if not bool(candidates.any()):
            break

        scores = improvement.masked_fill(~candidates, float("-inf"))
        flip_count = min(max_flips, scores.shape[-1])
        _, indices = torch.topk(scores, k=flip_count, dim=-1)
        row_has_flip = torch.isfinite(torch.gather(scores, -1, indices))
        flip_mask = torch.zeros_like(signs, dtype=torch.bool)
        flip_mask.scatter_(-1, indices, row_has_flip)
        signs = torch.where(flip_mask, -signs, signs)
        numerator = (h_diag * coeffs * signs).sum(dim=-1, keepdim=True)
        denominator = h_diag.sum().clamp_min(1e-12)
        alpha = numerator / denominator
    return signs


def sigma_delta_signs(coeffs: torch.Tensor, h_diag: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    adjusted = coeffs.float().clone()
    signs = torch.empty_like(adjusted)
    high_to_low = torch.argsort(h_diag.flatten(), descending=True).tolist()
    low_to_high = torch.argsort(h_diag.flatten(), descending=False).tolist()
    remaining = set(high_to_low)

    for coeff_index in high_to_low:
        remaining.discard(coeff_index)
        value = adjusted[:, coeff_index]
        sign = sign_no_zero(value)
        signs[:, coeff_index] = sign
        residual = value - alpha.squeeze(-1) * sign
        for target_index in low_to_high:
            if target_index in remaining:
                adjusted[:, target_index] += residual
                break
    return signs


def output_cost(target: torch.Tensor, pred: torch.Tensor, gamma: float = 0.01) -> torch.Tensor:
    target = target.float()
    pred = pred.float()
    rel_mse = (target - pred).pow(2).mean(dim=0) / target.pow(2).mean(dim=0).clamp_min(1e-8)
    cosine = F.cosine_similarity(target, pred, dim=0, eps=1e-8)
    return rel_mse + gamma * (1.0 - cosine)


def fisher_scale(coeffs: torch.Tensor, signs: torch.Tensor, h_diag: torch.Tensor) -> torch.Tensor:
    numerator = (h_diag * coeffs * signs).sum(dim=-1, keepdim=True)
    denominator = h_diag.sum().clamp_min(1e-12)
    return numerator / denominator


def refine_signs_with_output_cost(
    coeffs: torch.Tensor,
    signs: torch.Tensor,
    alpha: torch.Tensor,
    z_block: torch.Tensor,
    target: torch.Tensor,
    h_diag: torch.Tensor,
    iters: int,
    max_flip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if iters <= 0 or max_flip_ratio <= 0:
        return signs, alpha

    max_flips = max(1, int(coeffs.shape[-1] * max_flip_ratio))
    for _ in range(iters):
        pred = z_block @ (alpha * signs).t()
        current_cost = output_cost(target, pred)
        weighted_error = h_diag * (coeffs - alpha * signs).pow(2)
        near_scale = -((coeffs.abs() - alpha.abs()).abs())
        scores = weighted_error + 0.01 * near_scale
        candidate_indices = torch.topk(scores, k=max_flips, dim=-1).indices

        accepted = False
        for row in range(signs.shape[0]):
            row_pred = pred[:, row]
            row_cost = current_cost[row]
            for coeff_index in candidate_indices[row].tolist():
                delta = -2.0 * alpha[row, 0] * signs[row, coeff_index] * z_block[:, coeff_index]
                new_cost = output_cost(target[:, row : row + 1], (row_pred + delta)[:, None])[0]
                if bool(new_cost < row_cost):
                    signs[row, coeff_index] = -signs[row, coeff_index]
                    row_pred = row_pred + delta
                    row_cost = new_cost
                    accepted = True

        if not accepted:
            break
        alpha = fisher_scale(coeffs, signs, h_diag)
    return signs, alpha


class QuantizedLinearWeight(nn.Module):
    MODE_NAMES = ("identity", "hadamard", "haar")

    def __init__(
        self,
        signs: torch.Tensor,
        scales: torch.Tensor,
        mode_ids: torch.Tensor,
        block_size: int,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("signs", signs.to(torch.int8), persistent=True)
        self.register_buffer("scales", scales, persistent=True)
        self.register_buffer("mode_ids", mode_ids.to(torch.uint8), persistent=True)
        self._dequantized_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        weight: torch.Tensor,
        activation: torch.Tensor,
        block_size: int = 128,
        modes: Sequence[str] = MODE_NAMES,
        scale_dtype: torch.dtype = torch.float16,
        sigma_delta: bool = True,
        sign_refine_iters: int = 2,
        max_flip_ratio: float = 0.02,
        progress_desc: Optional[str] = None,
    ) -> "QuantizedLinearWeight":
        out_features, in_features = weight.shape
        if in_features % block_size != 0:
            raise ValueError(f"in_features={in_features} must be divisible by block_size={block_size}")
        if activation is None:
            raise ValueError("FM-SDBT requires calibration activation for every quantized weight.")

        mode_to_id = {name: index for index, name in enumerate(cls.MODE_NAMES)}
        mode_ids_to_try = [mode_to_id[name] for name in modes]
        n_blocks = in_features // block_size
        all_signs = torch.empty(out_features, in_features, device=weight.device, dtype=torch.int8)
        all_scales = torch.empty(out_features, n_blocks, device=weight.device, dtype=scale_dtype)
        all_modes = torch.empty(out_features, n_blocks, device=weight.device, dtype=torch.uint8)

        block_iter = range(n_blocks)
        if progress_desc is not None:
            block_iter = tqdm(block_iter, desc=progress_desc, leave=False)
        for block_index in block_iter:
            start = block_index * block_size
            end = start + block_size
            block = weight[:, start:end].float()
            x_block = activation[:, start:end].to(device=weight.device, dtype=torch.float32)
            target = x_block @ block.t()
            best_error = None
            best_signs = None
            best_scales = None
            best_modes = None

            for mode_id in mode_ids_to_try:
                coeffs = apply_weight_transform(block, mode_id)
                z_block = apply_weight_transform(x_block, mode_id)
                h_diag = z_block.pow(2).mean(dim=0, keepdim=True).clamp_min(1e-8)
                signs = sign_no_zero(coeffs)
                alpha = fisher_scale(coeffs, signs, h_diag)
                if sigma_delta:
                    signs = sigma_delta_signs(coeffs, h_diag, alpha)
                    alpha = fisher_scale(coeffs, signs, h_diag)
                signs, alpha = refine_signs_with_output_cost(
                    coeffs=coeffs,
                    signs=signs,
                    alpha=alpha,
                    z_block=z_block,
                    target=target,
                    h_diag=h_diag,
                    iters=sign_refine_iters,
                    max_flip_ratio=max_flip_ratio,
                )
                pred = z_block @ (alpha * signs).t()
                error = output_cost(target, pred)

                if best_error is None:
                    best_error = error
                    best_signs = signs
                    best_scales = alpha.squeeze(-1)
                    best_modes = torch.full((out_features,), mode_id, device=weight.device, dtype=torch.uint8)
                    continue

                improved = error < best_error
                best_error = torch.where(improved, error, best_error)
                best_signs = torch.where(improved[:, None], signs, best_signs)
                best_scales = torch.where(improved, alpha.squeeze(-1), best_scales)
                best_modes = torch.where(
                    improved,
                    torch.full_like(best_modes, mode_id, dtype=torch.uint8),
                    best_modes,
                )

            all_signs[:, start:end] = best_signs.to(torch.int8)
            all_scales[:, block_index] = best_scales.to(scale_dtype)
            all_modes[:, block_index] = best_modes

        return cls(
            signs=all_signs,
            scales=all_scales,
            mode_ids=all_modes,
            block_size=block_size,
            in_features=in_features,
            out_features=out_features,
        )

    @torch.no_grad()
    def dequantize(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (device, dtype)
        cached = self._dequantized_cache.get(cache_key)
        if cached is not None:
            return cached

        weight = torch.empty(self.out_features, self.in_features, device=device, dtype=torch.float32)
        signs = self.signs.to(device=device)
        scales = self.scales.to(device=device, dtype=torch.float32)
        mode_ids = self.mode_ids.to(device=device)
        n_blocks = self.in_features // self.block_size

        for block_index in range(n_blocks):
            start = block_index * self.block_size
            end = start + self.block_size
            coeffs = signs[:, start:end].float() * scales[:, block_index : block_index + 1]

            for mode_id in range(len(self.MODE_NAMES)):
                rows = mode_ids[:, block_index] == mode_id
                if bool(rows.any()):
                    weight[rows, start:end] = invert_weight_transform(coeffs[rows], mode_id)

        quantized_weight = weight.to(dtype=dtype)
        self._dequantized_cache[cache_key] = quantized_weight
        return quantized_weight


def quantize_linear_weight(
    module: nn.Module,
    name: str,
    activation: torch.Tensor,
    block_size: int = 128,
    progress_desc: Optional[str] = None,
) -> None:
    weight = getattr(module, name)
    quantized = QuantizedLinearWeight.from_float(
        weight.detach(),
        activation=activation,
        block_size=block_size,
        progress_desc=progress_desc,
    )
    del module._parameters[name]
    setattr(module, name, quantized)


def get_quantizable_weights(model: nn.Module, partial_quant: bool = False) -> list[tuple[str, nn.Module, str]]:
    target_weights = {"wq", "wk", "wv", "wo", "wg", "wd", "wu"}
    weights = []
    for module_name, module in model.named_modules():
        if partial_quant and not module_name.startswith("layers.0."):
            continue
        for name in target_weights:
            if name in module._parameters:
                full_name = f"{module_name}.{name}" if module_name else name
                weights.append((full_name, module, name))
    return weights


def quantize_model_weights(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
    block_size: int = 128,
    partial_quant: bool = False,
    show_progress: bool = True,
) -> None:
    weights = get_quantizable_weights(model, partial_quant=partial_quant)
    iterator = tqdm(weights, desc="Quantizing weights", leave=True) if show_progress else weights
    for full_name, module, name in iterator:
        if show_progress:
            iterator.set_postfix_str(full_name)
        quantize_linear_weight(
            module,
            name,
            activation=activations[full_name],
            block_size=block_size,
            progress_desc=f"Blocks {full_name}" if show_progress else None,
        )


def collect_quantized_state_dict(model: nn.Module) -> dict[str, dict[str, torch.Tensor | int]]:
    state_dict = {}
    for module_name, module in model.named_modules():
        for attr_name, value in module.__dict__.get("_modules", {}).items():
            if isinstance(value, QuantizedLinearWeight):
                full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                state_dict[full_name] = {
                    "signs": value.signs.cpu(),
                    "scales": value.scales.cpu(),
                    "mode_ids": value.mode_ids.cpu(),
                    "block_size": value.block_size,
                    "in_features": value.in_features,
                    "out_features": value.out_features,
                }
    return state_dict


def load_quantized_state_dict(model: nn.Module, state_dict: dict[str, dict[str, torch.Tensor | int]]) -> None:
    module_lookup = dict(model.named_modules())
    iterator = tqdm(state_dict.items(), desc="Loading quant cache", leave=True)
    for full_name, payload in iterator:
        iterator.set_postfix_str(full_name)
        module_name, attr_name = full_name.rsplit(".", 1)
        module = module_lookup[module_name]
        if attr_name in module._parameters:
            del module._parameters[attr_name]
        setattr(
            module,
            attr_name,
            QuantizedLinearWeight(
                signs=payload["signs"],
                scales=payload["scales"],
                mode_ids=payload["mode_ids"],
                block_size=int(payload["block_size"]),
                in_features=int(payload["in_features"]),
                out_features=int(payload["out_features"]),
            ),
        )


def linear(x: torch.Tensor, weight: torch.Tensor | QuantizedLinearWeight) -> torch.Tensor:
    if isinstance(weight, QuantizedLinearWeight):
        return torch.matmul(x, weight.dequantize(x.device, x.dtype).t())
    return torch.matmul(x, weight.t())


def flatten_activation(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1, x.shape[-1]).to(device="cpu", dtype=torch.float16)


def append_activation(capture: Optional[dict[str, list[torch.Tensor]]], key: str, value: torch.Tensor) -> None:
    if capture is not None:
        capture.setdefault(key, []).append(value)


def precompute_freqs_cis(args: ModelArgs) -> tuple[torch.Tensor, torch.Tensor]:
    dim = args.dim // args.n_heads
    inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, dim, 2).float() / dim))

    low_freq_wavelen = args.rope_original_max_position_embeddings / args.rope_low_freq_factor
    high_freq_wavelen = args.rope_original_max_position_embeddings / args.rope_high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / args.rope_factor, inv_freq)
    smooth_factor = (
        args.rope_original_max_position_embeddings / wavelen - args.rope_low_freq_factor
    ) / (args.rope_high_freq_factor - args.rope_low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / args.rope_factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    positions = torch.arange(args.max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cos[None, :, None, :]
    sin = freqs_sin[None, :, None, :]
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.activation_capture: Optional[dict[str, list[torch.Tensor]]] = None
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Parameter(torch.empty(args.n_heads * self.head_dim, args.dim))
        self.wk = nn.Parameter(torch.empty(self.n_kv_heads * self.head_dim, args.dim))
        self.wv = nn.Parameter(torch.empty(self.n_kv_heads * self.head_dim, args.dim))
        self.wo = nn.Parameter(torch.empty(args.dim, args.n_heads * self.head_dim))

        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.register_buffer(
            "cache_k",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        if self.activation_capture is not None:
            qkv_input = flatten_activation(x)
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.qkv", qkv_input)
        xq = linear(x, self.wq)      
        xk = linear(x, self.wk)
        xv = linear(x, self.wv)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.wo", flatten_activation(output))
        return linear(output, self.wo)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.activation_capture: Optional[dict[str, list[torch.Tensor]]] = None
        self.wg = nn.Parameter(torch.empty(args.hidden_dim, args.dim))
        self.wd = nn.Parameter(torch.empty(args.dim, args.hidden_dim))
        self.wu = nn.Parameter(torch.empty(args.hidden_dim, args.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_capture is not None:
            wgu_input = flatten_activation(x)
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wgu", wgu_input)
        gate = linear(x, self.wg)
        up = linear(x, self.wu)
        down_input = F.silu(gate) * up
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wd", flatten_activation(down_input))
        return linear(down_input, self.wd)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args, layer_id=layer_id)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention_norm(self.attention(x, start_pos, freqs_cos, freqs_sin, mask))
        out = h + self.ffn_norm(self.feed_forward(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.activation_capture: Optional[dict[str, list[torch.Tensor]]] = None
        self.capture_layer_ids: set[int] = set()

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, params) for layer_id in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(params)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def start_activation_capture(self, layer_ids: set[int]) -> None:
        self.activation_capture = {}
        self.capture_layer_ids = layer_ids
        for layer in self.layers:
            capture = self.activation_capture if layer.layer_id in layer_ids else None
            layer.attention.activation_capture = capture
            layer.feed_forward.activation_capture = capture

    def stop_activation_capture(self) -> None:
        for layer in self.layers:
            layer.attention.activation_capture = None
            layer.feed_forward.activation_capture = None

    def materialize_activation_capture(self, max_tokens: int, seed: int) -> dict[str, torch.Tensor]:
        if self.activation_capture is None:
            raise RuntimeError("activation capture was not started")

        raw = {key: torch.cat(values, dim=0).float() for key, values in self.activation_capture.items()}
        generator = torch.Generator().manual_seed(seed)
        sampled = {}
        for key, value in raw.items():
            if value.shape[0] > max_tokens:
                indices = torch.randperm(value.shape[0], generator=generator)[:max_tokens]
                value = value[indices]
            sampled[key] = value

        activations = {}
        for layer_id in self.capture_layer_ids:
            qkv = sampled[f"layers.{layer_id}.attention.qkv"]
            activations[f"layers.{layer_id}.attention.wq"] = qkv
            activations[f"layers.{layer_id}.attention.wk"] = qkv
            activations[f"layers.{layer_id}.attention.wv"] = qkv
            activations[f"layers.{layer_id}.attention.wo"] = sampled[f"layers.{layer_id}.attention.wo"]
            wgu = sampled[f"layers.{layer_id}.feed_forward.wgu"]
            activations[f"layers.{layer_id}.feed_forward.wg"] = wgu
            activations[f"layers.{layer_id}.feed_forward.wu"] = wgu
            activations[f"layers.{layer_id}.feed_forward.wd"] = sampled[f"layers.{layer_id}.feed_forward.wd"]
        return activations

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        
        # tokens = torch.tensor([[360, 560]], dtype=torch.long, device=tokens.device) # 토큰 고정
        
        h = self.tok_embeddings(tokens)

                        
        # # jw debug import
        
        # temp = self.tok_embeddings.weight
        # print(f'temp: {temp}')
        # sys.exit()  
        
        # # jw debug end
        
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen].to(h.device, dtype=h.dtype)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen].to(h.device, dtype=h.dtype)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)
            
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cos, freqs_sin, mask)
        h = self.norm(h)
        return linear(h, self.tok_embeddings.weight).float()
    
