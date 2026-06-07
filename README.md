# EXAONE.cpp

EXAONE.cpp is a research-oriented native C++ inference project for LG AI
Research's EXAONE model.

The goal is to run EXAONE inference on embedded and native environments without
depending on a large deep learning framework. The current weight pipeline
targets:

```text
LGAI-EXAONE/EXAONE-4.0-1.2B
```

## Project Scope

- Native C++ implementation
- Inference-only execution
- Custom FP8 E4M3 weight storage
- 64-byte-aligned binary weights for SIMD-friendly access
- [FlashAttention1/2 implementation](https://github.com/Dao-AILab/flash-attention)
- [CFHead implementation](https://github.com/HigherIdeal/CFHead)
- Research and non-commercial use

This project does not provide model training, fine-tuning, or official EXAONE
model distribution.

## Requirements

Install the Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

Hugging Face authentication may be required depending on the model or dataset:

```bash
huggingface-cli login
```

## Download And Convert Weights

Run the standalone conversion script from the repository root:

```bash
python weight/EXAONE_weight_FP8_E4M3.py
```

This single command:

1. Downloads `LGAI-EXAONE/EXAONE-4.0-1.2B`.
2. Converts the Hugging Face tensor names to the local EXAONE layout.
3. Classifies the exponent using the FP16 representation.
4. Maps FP16 zero and subnormal values to `2^-15`.
5. Keeps the upper 3 BF16 significand bits and truncates the lower 4 bits.
6. Writes the BF16 intermediate checkpoint and the C++ binary.

The output directory is:

```text
weight/EXAONE-4.0-1.2B/
├── config.json
├── tokenizer.json
├── ...
├── model.pth
└── model.bin
```

Use `--no-save-pth` when only the C++ binary is needed:

```bash
python weight/EXAONE_weight_FP8_E4M3.py --no-save-pth
```

### Custom FP8 Format

Non-normalization weights are stored as one byte:

```text
{sign[1], exponent[4], significand[3]}
```

The intermediate BF16 pattern and packed byte are:

```text
BF16 intermediate: S | 0111xxxx | xxx0000
Packed FP8:        S |     xxxx | xxx
```

Normalization weights remain raw BF16. The binary contains 332 ordered weight
payloads. Each payload begins on a 64-byte boundary, and unused alignment bytes
are zero-filled.

The little-endian header is:

```c
uint32_t weight_count;

// Repeated weight_count times:
uint32_t weight_index;
uint64_t absolute_byte_offset;
```

Projection matrices are transposed before serialization. Embedding and
normalization weights retain their original orientation.

## Python Validation

Python inference automatically prefers `model.bin` when both `model.bin` and
`model.pth` exist:

```bash
python python/scripts/local_infer.py --prompt "대한민국의 수도는 어디인가?"
```

Compare the Hugging Face original model against the local custom FP8 binary:

```bash
python python/scripts/ppl_benchmark.py
python python/scripts/mc_benchmark.py
```

Use `--weight-format pth` to explicitly evaluate `model.pth`, or
`--weight-format bin` to require `model.bin`.

## Benchmark Results

Higher is better. Values are accuracy percentages.

| Benchmark | Original (FP16) | Truncation (FP8 E4M3) |
|---|---:|---:|
| MMLU-Pro | 21.484 | 24.219 |
| KMMLU-Pro | 39.453 | 39.063 |
| ARC-Challenge | 50.391 | 50.781 |
| HellaSwag | 39.844 | 41.016 |

## Repository Structure

```text
EXAONE.cpp/
├── python/
│   ├── scripts/      # Inference and benchmark entry points
│   └── src/          # Reference PyTorch model and binary loader
├── weight/
│   └── EXAONE_weight_FP8_E4M3.py
└── README.md
```
