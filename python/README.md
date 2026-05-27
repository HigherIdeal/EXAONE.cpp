# EXAONE Python Inference

Minimal example for running `LGAI-EXAONE/EXAONE-4.0-1.2B` with Hugging Face Transformers.
The script prints the raw prompt, the chat-template-rendered text including special tokens, the token IDs, and then the model output.

```bash
pip install torch transformers accelerate
python local_infer.py --prompt "Strawberry라는 단어에는 알파벳 'r'이 몇 개 있는가?"
```

Reasoning mode can be enabled with `--reasoning`:

```bash
python local_infer.py \
  --reasoning \
  --prompt "Strawberry라는 단어에는 알파벳 'r'이 몇 개 있는가?"
```

The first run downloads the model from Hugging Face. If the model requires access approval, log in first:

```bash
huggingface-cli login
```
