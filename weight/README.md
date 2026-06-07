# Weight Pipeline

This directory contains the standalone EXAONE weight conversion script and the
generated local model directory.

## Files

```text
weight/
├── EXAONE_weight_FP8_E4M3.py
└── EXAONE-4.0-1.2B/
    ├── config.json
    ├── tokenizer.json
    ├── ...
    ├── model.pth
    └── model.bin
```

## One-Way Conversion

Run from the repository root:

```bash
python weight/EXAONE_weight_FP8_E4M3.py
```

The script performs the whole pipeline in one direction:

1. Download `LGAI-EXAONE/EXAONE-4.0-1.2B` config, tokenizer files, and
   safetensors weights.
2. Convert Hugging Face tensor names into the local EXAONE layout.
3. Build the custom truncated BF16 representation.
4. Save `model.pth` unless `--no-save-pth` is used.
5. Pack the aligned C++ binary as `model.bin`.

If only the binary is needed:

```bash
python weight/EXAONE_weight_FP8_E4M3.py --no-save-pth
```

## Tensor Name Conversion

The Hugging Face weights are remapped to the local names below:

- `model.embed_tokens.weight` -> `tok_embeddings.weight`
- `model.norm.weight` -> `norm.weight`
- `model.layers.{i}.self_attn.q_proj.weight` -> `layers.{i}.attention.wq`
- `model.layers.{i}.self_attn.k_proj.weight` -> `layers.{i}.attention.wk`
- `model.layers.{i}.self_attn.v_proj.weight` -> `layers.{i}.attention.wv`
- `model.layers.{i}.self_attn.o_proj.weight` -> `layers.{i}.attention.wo`
- `model.layers.{i}.self_attn.q_norm.weight` -> `layers.{i}.attention.q_norm.weight`
- `model.layers.{i}.self_attn.k_norm.weight` -> `layers.{i}.attention.k_norm.weight`
- `model.layers.{i}.mlp.up_proj.weight` -> `layers.{i}.feed_forward.wu`
- `model.layers.{i}.mlp.gate_proj.weight` -> `layers.{i}.feed_forward.wg`
- `model.layers.{i}.mlp.down_proj.weight` -> `layers.{i}.feed_forward.wd`
- `model.layers.{i}.post_attention_layernorm.weight` -> `layers.{i}.attention_norm.weight`
- `model.layers.{i}.post_feedforward_layernorm.weight` -> `layers.{i}.ffn_norm.weight`

## Truncation Rule

The script does not store ordinary IEEE FP8 values directly from the original
weights. It first creates a BF16 intermediate whose bits follow a constrained
pattern, and that constrained BF16 is then packed into one byte.

### Step 1: Start from the original tensor

For every non-normalization weight:

- Convert values to BF16 to read the sign bit and BF16 significand bits.
- Convert values to FP16 to determine the exponent bucket.

### Step 2: Exponent handling

Let the FP16 exponent field be `eeeee`.

- If FP16 is zero or subnormal, force the stored exponent bucket to `2^-15`.
- If FP16 exponent is larger than the supported range, clamp it to `2^0`.
- Otherwise keep the lower 4 bits of the FP16 exponent bucket.

The stored BF16 exponent is forced to:

```text
0111xxxx
```

where `xxxx` comes from the clamped FP16 exponent field.

### Step 3: Significand handling

From the original BF16 significand:

- Keep the upper 3 bits.
- Zero the lower 4 bits.

So the constrained BF16 pattern becomes:

```text
S | 0111xxxx | xxx0000
```

This constrained BF16 tensor is what gets written to `model.pth`.

## Custom FP8 E4M3 Byte

Every non-normalization element is packed into one byte:

```text
{sign[1], exponent[4], significand[3]}
```

Bit layout:

```text
bit 7     : sign
bits 6..3 : exponent low 4 bits
bits 2..0 : upper 3 BF16 significand bits
```

Equivalently:

```text
BF16 intermediate: S | 0111xxxx | xxx0000
Packed byte:       S | xxxx     | xxx
```

This is the custom format used in `model.bin`.

## What Stays BF16

The following weights are not packed to 8-bit and remain raw BF16:

- `layers.{i}.attention.q_norm.weight`
- `layers.{i}.attention.k_norm.weight`
- `layers.{i}.attention_norm.weight`
- `layers.{i}.ffn_norm.weight`
- `norm.weight`

These are serialized as their 16-bit BF16 memory bytes in little-endian order.

## Serialization Order

The binary contains exactly 332 payloads.

### Global order

1. `tok_embeddings.weight`
2. layer 0
3. layer 1
4. ...
5. layer 29
6. `norm.weight`

### Per-layer order

Each layer is serialized in this exact order:

1. `q_proj` -> `layers.{i}.attention.wq`
2. `q_norm` -> `layers.{i}.attention.q_norm.weight`
3. `k_proj` -> `layers.{i}.attention.wk`
4. `k_norm` -> `layers.{i}.attention.k_norm.weight`
5. `v_proj` -> `layers.{i}.attention.wv`
6. `o_proj` -> `layers.{i}.attention.wo`
7. `a_norm` -> `layers.{i}.attention_norm.weight`
8. `u_proj` -> `layers.{i}.feed_forward.wu`
9. `g_proj` -> `layers.{i}.feed_forward.wg`
10. `d_proj` -> `layers.{i}.feed_forward.wd`
11. `f_norm` -> `layers.{i}.ffn_norm.weight`

So the index mapping is:

- `0` -> embedding
- `1` -> layer 0 `q_proj`
- `2` -> layer 0 `q_norm`
- ...
- `11` -> layer 0 `f_norm`
- ...
- `330` -> layer 29 `f_norm`
- `331` -> final `norm.weight`

## Transpose Rule

The local PyTorch checkpoint stores linear weights in the usual shape:

```text
[out_features, in_features]
```

The reference Python model multiplies with `weight.t()`, so the checkpoint
itself is not stored pre-transposed.

For `model.bin`, the following tensors are transposed before serialization:

- `tok_embeddings.weight`: no
- `layers.{i}.attention.wq`: yes
- `layers.{i}.attention.q_norm.weight`: no
- `layers.{i}.attention.wk`: yes
- `layers.{i}.attention.k_norm.weight`: no
- `layers.{i}.attention.wv`: yes
- `layers.{i}.attention.wo`: yes
- `layers.{i}.attention_norm.weight`: no
- `layers.{i}.feed_forward.wu`: yes
- `layers.{i}.feed_forward.wg`: yes
- `layers.{i}.feed_forward.wd`: yes
- `layers.{i}.ffn_norm.weight`: no
- `norm.weight`: no

Notes:

- `q_proj` and `o_proj` are square, so their shape does not visibly change.
- Their byte order still changes because the matrix is actually transposed.

## `model.bin` Header

The binary begins with a compact little-endian header:

```c
uint32_t weight_count;

// repeated weight_count times
uint32_t weight_index;
uint64_t absolute_byte_offset;
```

Meaning:

- `weight_count` is `332`.
- Each `weight_index` is `0, 1, 2, ..., 331`.
- Each `absolute_byte_offset` is the absolute file position where that payload
  begins.

The header itself is followed by zero padding up to the next 64-byte boundary.

## Alignment Rule

Every payload begins at an offset that is a multiple of 64 bytes.

If one payload ends before the next 64-byte boundary:

- the gap is filled with zero bytes,
- and the next payload starts exactly at the next aligned offset.

This is done so the C++ side can map the file and access each weight from a
clean 64-byte boundary.

## Payload Interpretation

To read a payload in C++:

1. Read `weight_count`.
2. Read the `(index, offset)` table.
3. Use the known tensor shape for that index.
4. Seek to the stored offset.
5. Read:
   - 1 byte per element for FP8 weights
   - 2 bytes per element for BF16 norm weights

For FP8 payloads, restore each byte:

```text
S | xxxx | xxx
```

back to the constrained BF16 pattern:

```text
S | 0111xxxx | xxx0000
```

If the tensor was written transposed, interpret the stored bytes in the
transposed shape and transpose back only if the consumer wants the original
PyTorch orientation.

## Current Output Names

By default the script writes:

```text
weight/EXAONE-4.0-1.2B/model.pth
weight/EXAONE-4.0-1.2B/model.bin
```

The tokenizer and config sidecar files are copied into the same directory so
both Python and C++ tooling can use one local model root.
