import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

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


QUANT_PROJECTION_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:     
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    hard = sign_no_zero(x)
    return x + (hard - x).detach()


def robust_diag(x: torch.Tensor, tau: float = 0.2, gamma: float = 0.2) -> torch.Tensor:
    diag = x.float().pow(2).mean(dim=0).clamp_min(1e-8)
    if diag.numel() > 1 and tau > 0:
        threshold = torch.quantile(diag, min(1.0, 1.0 - tau))
        diag = torch.minimum(diag, threshold)
    return ((1.0 - gamma) * diag + gamma * diag.mean()).sqrt().clamp_min(1e-6)


def nanoquant_rank(out_features: int, in_features: int, target_bits: float = 1.0) -> int:
    rank = round(target_bits * out_features * in_features / (out_features + in_features))
    return max(1, min(rank, out_features, in_features))


def low_rank_binary_admm(
    weight: torch.Tensor,
    rank: int,
    iters: int = 6,
    rho: float = 1.0,
    lamb: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    rank = min(rank, s.numel())
    root_s = s[:rank].sqrt()
    u_lat = u[:, :rank] * root_s
    v_lat = vh[:rank, :].t() * root_s
    u_bin = sign_no_zero(u_lat)
    v_bin = sign_no_zero(v_lat)
    u_dual = torch.zeros_like(u_lat)
    v_dual = torch.zeros_like(v_lat)
    eye = torch.eye(rank, device=weight.device, dtype=torch.float32)

    for _ in range(iters):
        lhs = v_lat.t() @ v_lat + (rho + lamb) * eye
        rhs = weight.float() @ v_lat + rho * (u_bin - u_dual)
        u_lat = torch.linalg.solve(lhs, rhs.t()).t()
        u_bin = sign_no_zero(u_lat + u_dual)
        u_dual = u_dual + u_lat - u_bin

        lhs = u_lat.t() @ u_lat + (rho + lamb) * eye
        rhs = weight.float().t() @ u_lat + rho * (v_bin - v_dual)
        v_lat = torch.linalg.solve(lhs, rhs.t()).t()
        v_bin = sign_no_zero(v_lat + v_dual)
        v_dual = v_dual + v_lat - v_bin

    return u_lat, v_lat


def balance_latent_factors(u_lat: torch.Tensor, v_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    factor = (v_lat.norm().clamp_min(1e-8) / u_lat.norm().clamp_min(1e-8)).sqrt()
    u_bal = u_lat * factor
    v_bal = v_lat / factor
    s_out = u_bal.abs().mean(dim=1).clamp_min(1e-6)
    s_in = v_bal.abs().mean(dim=1).clamp_min(1e-6)
    return sign_no_zero(u_bal / s_out[:, None]), sign_no_zero(v_bal / s_in[:, None]), s_out, s_in


def nanoquant_weight(
    u: torch.Tensor,
    v: torch.Tensor,
    out_scale: torch.Tensor,
    in_scale: torch.Tensor,
    ste: bool = False,
) -> torch.Tensor:
    u_bin = sign_ste(u) if ste else sign_no_zero(u)
    v_bin = sign_ste(v) if ste else sign_no_zero(v)
    return out_scale[:, None] * (u_bin @ v_bin.t()) * in_scale[None, :]


def tune_latent_ste(
    x: torch.Tensor,
    y: torch.Tensor,
    u_lat: torch.Tensor,
    v_lat: torch.Tensor,
    out_scale: torch.Tensor,
    in_scale: torch.Tensor,
    steps: int = 8,
    lr: float = 5e-4,
    progress_desc: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if steps <= 0:
        return u_lat, v_lat, out_scale, in_scale, float("nan")

    u_param = nn.Parameter(u_lat.float())
    v_param = nn.Parameter(v_lat.float())
    out_scale_param = nn.Parameter(out_scale.float())
    in_scale_param = nn.Parameter(in_scale.float())
    optimizer = torch.optim.AdamW(
        [u_param, v_param, out_scale_param, in_scale_param],
        lr=lr,
        weight_decay=0.0,
    )
    target_norm = y.pow(2).mean().clamp_min(1e-8)
    iterator = range(steps)
    if progress_desc is not None:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

    last_loss = float("nan")
    for _ in iterator:
        optimizer.zero_grad(set_to_none=True)
        weight_hat = nanoquant_weight(u_param, v_param, out_scale_param, in_scale_param, ste=True)
        pred = x @ weight_hat.t()
        loss = (pred - y).pow(2).mean() / target_norm
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if progress_desc is not None:
            iterator.set_postfix_str(f"loss={last_loss:.4f}")

    return (
        u_param.detach(),
        v_param.detach(),
        out_scale_param.detach(),
        in_scale_param.detach(),
        last_loss,
    )


class NanoQuantLinearWeight(nn.Module):
    def __init__(
        self,
        blocks: list[dict[str, torch.Tensor | int]],
        out_features: int,
        in_features: int,
        block_size: int,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.block_size = block_size
        self.block_ranges: list[tuple[int, int]] = []
        for block_index, block in enumerate(blocks):
            self.block_ranges.append((int(block["start"]), int(block["end"])))
            self.register_buffer(f"u_sign_{block_index}", block["u_sign"].to(torch.int8), persistent=True)
            self.register_buffer(f"v_sign_{block_index}", block["v_sign"].to(torch.int8), persistent=True)
            self.register_buffer(f"out_scale_{block_index}", block["out_scale"].to(torch.float16), persistent=True)
            self.register_buffer(f"in_scale_{block_index}", block["in_scale"].to(torch.float16), persistent=True)
        self._dequantized_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def from_float(
        cls,
        weight: torch.Tensor,
        activation: dict[str, torch.Tensor],
        target_bits: float = 1.0,
        rank: Optional[int] = None,
        admm_iters: int = 6,
        rho: float = 1.0,
        lamb: float = 1e-4,
        tau: float = 0.2,
        gamma: float = 0.2,
        ste_steps: int = 8,
        ste_lr: float = 5e-4,
        block_size: int = 128,
        progress_desc: Optional[str] = None,
    ) -> "NanoQuantLinearWeight":
        with torch.no_grad():
            x = activation["x"].to(device=weight.device, dtype=torch.float32)
            y_full = activation["y"].to(device=weight.device, dtype=torch.float32)
            out_features, in_features = weight.shape
            d_out = robust_diag(y_full, tau=tau, gamma=gamma).to(weight.device)

        blocks: list[dict[str, torch.Tensor | int]] = []
        ranges = [(start, min(start + block_size, in_features)) for start in range(0, in_features, block_size)]
        iterator = tqdm(ranges, desc=progress_desc, leave=False) if progress_desc is not None else ranges
        for start, end in iterator:
            with torch.no_grad():
                x_block = x[:, start:end]
                weight_block = weight[:, start:end].float()
                y_block = x_block @ weight_block.t()
                block_rank = rank or nanoquant_rank(out_features, end - start, target_bits=target_bits)
                d_in = robust_diag(x_block, tau=tau, gamma=gamma).to(weight.device)
                target = d_out[:, None] * weight_block * d_in[None, :]
                u_lat, v_lat = low_rank_binary_admm(target, rank=block_rank, iters=admm_iters, rho=rho, lamb=lamb)
                _, _, s_out, s_in = balance_latent_factors(u_lat, v_lat)
                out_scale = s_out / d_out
                in_scale = s_in / d_in

            u_lat, v_lat, out_scale, in_scale, ste_loss = tune_latent_ste(
                x=x_block,
                y=y_block,
                u_lat=u_lat,
                v_lat=v_lat,
                out_scale=out_scale,
                in_scale=in_scale,
                steps=ste_steps,
                lr=ste_lr,
            )

            with torch.no_grad():
                u_sign = sign_no_zero(u_lat)
                v_sign = sign_no_zero(v_lat)
                base_weight = (u_sign @ v_sign.t()) * in_scale[None, :]
                pred = x_block @ base_weight.t()
                out_scale = ((pred * y_block).sum(dim=0) / pred.pow(2).sum(dim=0).clamp_min(1e-8)).float()
                blocks.append(
                    {
                        "start": start,
                        "end": end,
                        "u_sign": u_sign,
                        "v_sign": v_sign,
                        "out_scale": out_scale,
                        "in_scale": in_scale,
                    }
                )
            if progress_desc is not None:
                iterator.set_postfix_str(f"{start}:{end} r={block_rank} loss={ste_loss:.4f}")

        return cls(
            blocks=blocks,
            out_features=out_features,
            in_features=in_features,
            block_size=block_size,
        )

    def iter_blocks(self):
        for block_index, (start, end) in enumerate(self.block_ranges):
            yield (
                start,
                end,
                getattr(self, f"u_sign_{block_index}"),
                getattr(self, f"v_sign_{block_index}"),
                getattr(self, f"out_scale_{block_index}"),
                getattr(self, f"in_scale_{block_index}"),
            )

    @torch.no_grad()
    def dequantize(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (device, dtype)
        cached = self._dequantized_cache.get(cache_key)
        if cached is not None:
            return cached
        weight = torch.empty(self.out_features, self.in_features, device=device, dtype=torch.float32)
        for start, end, u_sign, v_sign, out_scale, in_scale in self.iter_blocks():
            u = u_sign.to(device=device, dtype=torch.float32)
            v = v_sign.to(device=device, dtype=torch.float32)
            out_s = out_scale.to(device=device, dtype=torch.float32)
            in_s = in_scale.to(device=device, dtype=torch.float32)
            weight[:, start:end] = out_s[:, None] * (u @ v.t()) * in_s[None, :]
        weight = weight.to(dtype=dtype)
        self._dequantized_cache[cache_key] = weight
        return weight


@torch.no_grad()
def print_reconstruction_metrics(full_name: str, weight: torch.Tensor, quantized: NanoQuantLinearWeight, activation: dict[str, torch.Tensor]) -> None:
    fp_weight = weight.float()
    q_weight = quantized.dequantize(weight.device, torch.float32)
    diff = q_weight - fp_weight
    mse = diff.pow(2).mean()
    rel = diff.norm() / fp_weight.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(fp_weight.flatten(), q_weight.flatten(), dim=0, eps=1e-8)
    print(
        f"Weight metric {full_name}: mse={float(mse):.6e} rel={float(rel):.6f} cos={float(cosine):.6f}",
        flush=True,
    )

    x = activation["x"].to(device=weight.device, dtype=torch.float32)
    y = activation["y"].to(device=weight.device, dtype=torch.float32)
    pred = x @ q_weight.t()
    out_diff = pred - y
    out_mse = out_diff.pow(2).mean()
    out_rel = out_diff.norm() / y.norm().clamp_min(1e-8)
    out_cosine = F.cosine_similarity(y.flatten(), pred.flatten(), dim=0, eps=1e-8)
    print(
        f"Output metric {full_name}: mse={float(out_mse):.6e} rel={float(out_rel):.6f} cos={float(out_cosine):.6f}",
        flush=True,
    )


def quantize_linear_weight(
    full_name: str,
    module: nn.Module,
    name: str,
    activation: dict[str, torch.Tensor],
    block_size: int,
    progress_desc: Optional[str] = None,
) -> None:
    weight = getattr(module, name)
    quantized = NanoQuantLinearWeight.from_float(
        weight.detach(),
        activation=activation,
        block_size=block_size,
        progress_desc=progress_desc,
    )
    print_reconstruction_metrics(full_name, weight.detach(), quantized, activation)
    del module._parameters[name]
    setattr(module, name, quantized)


def get_quantizable_weights(
    model: nn.Module,
    partial_quant: bool = False,
    target_modules: Optional[set[str]] = None,
    target_projections: Optional[set[str]] = None,
    exclude_projections: Optional[set[str]] = None,
) -> list[tuple[str, nn.Module, str]]:
    target_weights = set(target_projections or QUANT_PROJECTION_NAMES)
    target_weights -= set(exclude_projections or set())
    invalid = target_weights - set(QUANT_PROJECTION_NAMES)
    if invalid:
        raise ValueError(f"Unknown target projections: {sorted(invalid)}")

    weights = []
    for module_name, module in model.named_modules():
        if partial_quant and not module_name.startswith("layers.0."):
            continue
        for name in QUANT_PROJECTION_NAMES:
            if name not in target_weights:
                continue
            if name in module._parameters:
                full_name = f"{module_name}.{name}" if module_name else name
                if target_modules is not None and full_name not in target_modules:
                    continue
                weights.append((full_name, module, name))
    if target_modules is not None:
        found = {full_name for full_name, _, _ in weights}
        missing = target_modules - found
        if missing:
            raise ValueError(f"Target modules not found or not quantizable: {sorted(missing)}")
    return weights


def quantize_model_weights(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
    block_size: int = 128,
    partial_quant: bool = False,
    target_modules: Optional[set[str]] = None,
    target_projections: Optional[set[str]] = None,
    exclude_projections: Optional[set[str]] = None,
    show_progress: bool = True,
) -> None:
    weights = get_quantizable_weights(
        model,
        partial_quant=partial_quant,
        target_modules=target_modules,
        target_projections=target_projections,
        exclude_projections=exclude_projections,
    )
    iterator = tqdm(weights, desc="Quantizing weights", leave=True) if show_progress else weights
    for full_name, module, name in iterator:
        if show_progress:
            iterator.set_postfix_str(full_name)
        quantize_linear_weight(
            full_name,
            module,
            name,
            activation=activations[full_name],
            block_size=block_size,
            progress_desc=f"Tune {full_name}" if show_progress else None,
        )


def collect_quantized_state_dict(model: nn.Module) -> dict[str, dict[str, torch.Tensor | int]]:
    state_dict = {}
    for module_name, module in model.named_modules():
        for attr_name, value in module.__dict__.get("_modules", {}).items():
            if isinstance(value, NanoQuantLinearWeight):
                full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                state_dict[full_name] = {
                    "type": "nanoquant",
                    "block_size": value.block_size,
                    "blocks": [
                        {
                            "start": start,
                            "end": end,
                            "u_sign": u_sign.cpu(),
                            "v_sign": v_sign.cpu(),
                            "out_scale": out_scale.cpu(),
                            "in_scale": in_scale.cpu(),
                        }
                        for start, end, u_sign, v_sign, out_scale, in_scale in value.iter_blocks()
                    ],
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
            NanoQuantLinearWeight(
                blocks=payload["blocks"],
                in_features=int(payload["in_features"]),
                out_features=int(payload["out_features"]),
                block_size=int(payload["block_size"]),
            ),
        )


def linear(x: torch.Tensor, weight: torch.Tensor | NanoQuantLinearWeight) -> torch.Tensor:
    if isinstance(weight, NanoQuantLinearWeight):
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
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.wq.y", flatten_activation(xq))
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.wk.y", flatten_activation(xk))
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.wv.y", flatten_activation(xv))

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
        out = linear(output, self.wo)
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.attention.wo.y", flatten_activation(out))
        return out


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
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wg.y", flatten_activation(gate))
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wu.y", flatten_activation(up))
        down_input = F.silu(gate) * up
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wd", flatten_activation(down_input))
        out = linear(down_input, self.wd)
        if self.activation_capture is not None:
            append_activation(self.activation_capture, f"layers.{self.layer_id}.feed_forward.wd.y", flatten_activation(out))
        return out


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
        first_value = next(iter(raw.values()))
        indices = None
        if first_value.shape[0] > max_tokens:
            indices = torch.randperm(first_value.shape[0], generator=generator)[:max_tokens]
        sampled = {}
        for key, value in raw.items():
            if indices is not None:
                value = value[indices]
            sampled[key] = value

        activations = {}
        for layer_id in self.capture_layer_ids:
            qkv = sampled[f"layers.{layer_id}.attention.qkv"]
            activations[f"layers.{layer_id}.attention.wq"] = {
                "x": qkv,
                "y": sampled[f"layers.{layer_id}.attention.wq.y"],
            }
            activations[f"layers.{layer_id}.attention.wk"] = {
                "x": qkv,
                "y": sampled[f"layers.{layer_id}.attention.wk.y"],
            }
            activations[f"layers.{layer_id}.attention.wv"] = {
                "x": qkv,
                "y": sampled[f"layers.{layer_id}.attention.wv.y"],
            }
            activations[f"layers.{layer_id}.attention.wo"] = {
                "x": sampled[f"layers.{layer_id}.attention.wo"],
                "y": sampled[f"layers.{layer_id}.attention.wo.y"],
            }
            wgu = sampled[f"layers.{layer_id}.feed_forward.wgu"]
            activations[f"layers.{layer_id}.feed_forward.wg"] = {
                "x": wgu,
                "y": sampled[f"layers.{layer_id}.feed_forward.wg.y"],
            }
            activations[f"layers.{layer_id}.feed_forward.wu"] = {
                "x": wgu,
                "y": sampled[f"layers.{layer_id}.feed_forward.wu.y"],
            }
            activations[f"layers.{layer_id}.feed_forward.wd"] = {
                "x": sampled[f"layers.{layer_id}.feed_forward.wd"],
                "y": sampled[f"layers.{layer_id}.feed_forward.wd.y"],
            }
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
    
