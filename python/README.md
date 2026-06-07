# EXAONE Python Inference

The Python implementation provides local inference, perplexity measurement,
multiple-choice benchmarks, and validation of the custom C++ `model.bin`.

Run the commands below from the repository root.

## Install

```bash
python -m pip install -r python/requirements.txt
```

## Download And Build Weights

The complete weight pipeline is contained in one standalone script:

```bash
python weight/EXAONE_weight_FP8_E4M3.py
```

It downloads `LGAI-EXAONE/EXAONE-4.0-1.2B` and creates:

```text
weight/EXAONE-4.0-1.2B/
├── tokenizer and config files
├── model.pth
└── model.bin
```

`model.pth` contains the truncated BF16 representation used for Python
inspection. `model.bin` contains non-norm weights packed as custom E4M3 bytes
and norm weights stored as raw BF16.

The custom E4M3 layout is:

```text
BF16 intermediate: S | 0111xxxx | xxx0000
Packed byte:        S | xxxx     | xxx
```

The binary contains 332 payloads. Projection matrices are transposed for C++
access, and every payload begins at a 64-byte-aligned absolute file offset.

To omit the intermediate `model.pth`:

```bash
python weight/EXAONE_weight_FP8_E4M3.py --no-save-pth
```

## Run Inference

```bash
python python/scripts/local_infer.py \
  --prompt "Strawberry라는 단어에는 알파벳 'r'이 몇 개 있는가?"
```

Reasoning mode can be enabled with `--reasoning`.

For a local model directory, the loader uses:

1. `model.bin` when available.
2. `model.pth` when `model.bin` is absent.

The custom FP8 bytes are restored to their exact truncated BF16 bit patterns
before running the reference PyTorch model.

## Perplexity

Evaluate WikiText-2:

```bash
python python/scripts/ppl.py --torch-dtype bfloat16
```

Compare the Hugging Face original against local `model.bin` on WikiText-2, C4,
and Penn Treebank:

```bash
python python/scripts/ppl_benchmark.py
```

Default settings:

```text
sequence length: 2048
samples per dataset: 128
seed: 99
```

## Accuracy Benchmarks

Run MMLU-Pro, KMMLU-Pro, ARC-Challenge, and HellaSwag:

```bash
python python/scripts/mc_benchmark.py
```

The default model pair is:

```text
Original: LGAI-EXAONE/EXAONE-4.0-1.2B
FP8:      weight/EXAONE-4.0-1.2B/model.bin
```

MMLU-Pro and KMMLU-Pro use generation and answer extraction. ARC-Challenge and
HellaSwag use continuation likelihood. KMMLU-Pro is a gated dataset and
requires Hugging Face access approval and authentication.

## Weight Selection

Both benchmark scripts accept:

```text
--weight-format auto  # Prefer model.bin, then model.pth
--weight-format bin   # Require model.bin
--weight-format pth   # Require model.pth
```

An explicit local directory or Hugging Face model ID can be supplied through
`--model-id`.
