import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

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

def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight.t())


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
    def __init__(self, args: ModelArgs):
        super().__init__()
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
        return linear(output, self.wo)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(args.hidden_dim, args.dim))
        self.w2 = nn.Parameter(torch.empty(args.dim, args.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(args.hidden_dim, args.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = linear(x, self.w1)
        silu = F.silu(f1)
        f3 = linear(x, self.w3)
        mul = silu * f3
        f2 = linear(mul, self.w2)
        return f2


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
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

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, params) for layer_id in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(params)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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
    
