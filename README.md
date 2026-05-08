# EXAONE_CPP

EXAONE_CPP is a research-oriented native C++ inference project for LG AI Research's EXAONE model.

The goal of this project is to provide a simple and minimal C++ implementation that can run EXAONE inference on various embedded devices without relying on a large deep learning framework.

This repository does not aim to train, fine-tune, or modify the EXAONE model.  
The initial scope is limited to tokenizer implementation and inference execution.

## Project Scope

This project focuses on:

- Native C++ implementation
- Minimal source code structure
- Tokenizer implementation
- Inference-only execution
- Embedded-device-oriented experimentation
- Research and non-commercial use

This project does not include:

- Model training
- Fine-tuning
- Dataset construction
- Commercial deployment
- Official EXAONE model distribution

Quantization may be added later if time and resources allow.

## Repository Structure

```text
EXAONE_CPP/
├── src/              # Entry point
├── tokenizer/        # Tokenizer implementation
├── inference/        # Inference engine implementation
├── examples/         # Simple usage examples
└── quantization/     # Optional future quantization support
