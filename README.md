---
license: cc-by-nc-sa-4.0
language:
- en
- zh
tags:
- dermatology
- medical
- multimodal
- vision-language-model
- skin-lesion
pipeline_tag: image-text-to-text
---

# Trustworthy and Fair SkinGPT-R1 for Democratizing Dermatological Reasoning across Diverse Ethnicities

![CUHKSZ Logo](cuhksz-logo.png)

*Update: We will soon release the SkinGPT-R1-7B weights.*

SkinGPT-R1 is a dermatological reasoning vision language model for research and education. 🩺✨

**The Chinese University of Hong Kong, Shenzhen**

## Disclaimer

This project is for **research and educational use only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. ⚠️

## License

This repository is released under **CC BY-NC-SA 4.0**.
See [LICENSE](LICENSE) for details.

## Overview

```text
SkinGPT-R1/
├── checkpoints/
├── environment.yml
├── inference/
│   ├── full_precision/
│   └── int4_quantized/
├── requirements.txt
└── README.md
```

Model weights directory:

- `./checkpoints`

## Highlights

- 🔬 Dermatology-oriented multimodal reasoning
- 🧠 Full-precision and INT4 inference paths
- 💬 Multi-turn chat and API serving
- ⚡ RTX 50 series friendly SDPA-backed INT4 runtime

## Install

`environment.yml` is a Conda environment definition file. It captures the Python version and the package versions we use, so other users can recreate a working environment from scratch with one command.

Recommended from scratch:

```bash
cd SkinGPT-R1
conda env create -f environment.yml
conda activate skingpt-r1
```

Manual setup:

```bash
cd SkinGPT-R1
conda create -n skingpt-r1 python=3.10.20 -y
conda activate skingpt-r1
pip install -r requirements.txt
```

This repo is currently aligned to the maintainers' working environment:

- `torch==2.10.0`
- `torchvision==0.25.0`
- `transformers==5.3.0`
- `qwen-vl-utils==0.0.14`

For RTX 50 series, start with the default `sdpa` path and do not install `flash-attn`
unless you have already verified that your CUDA stack supports it.

## Quick Start

1. Clone the repository and enter it.

```bash
git clone <your-repo-url>
cd SkinGPT-R1
```

2. Create the environment.

```bash
conda env create -f environment.yml
conda activate skingpt-r1
```

3. Put model weights under:

```text
./checkpoints
```

4. Prepare a test image, for example:

```text
./test_images/lesion.jpg
```

5. Run one of the inference commands below.

## Attention Backend Notes

This repo uses two attention acceleration paths:

- `flash_attention_2`: external package, optional
- `sdpa`: PyTorch native scaled dot product attention

Recommended choice:

- 🚀 RTX 50 series: use `sdpa`
- 🚀 A100 / RTX 3090 / RTX 4090 / H100 and other GPUs explicitly listed by the FlashAttention project: you can try `flash_attention_2`

Practical notes:

- The current repo pins `torch==2.10.0`, and SDPA is already built into PyTorch in this version.
- FlashAttention's official README currently lists Ampere, Ada, and Hopper support for FlashAttention-2. It does not list RTX 50 / Blackwell consumer GPUs in that section, so this repo defaults to `sdpa` for that path.
- Newer PyTorch releases continue improving SDPA, and this repo is already on a modern PyTorch stack. Even so, RTX 50 series should still default to `sdpa` unless `flash-attn` has been explicitly validated in your environment.

If you are on an RTX 5090 and `flash-attn` is unavailable or unstable in your environment, use the INT4 path in this repo, which is already configured with `attn_implementation="sdpa"`.

## Usage

### Full Precision

Single image:

```bash
bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg
```

If you are on a multi-GPU server and want to select one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg
```

Multi-turn chat:

```bash
bash inference/full_precision/run_chat.sh --image ./test_images/lesion.jpg
```

API service:

```bash
bash inference/full_precision/run_api.sh
```

Default API port: `5900`

### INT4 Quantized

Single image:

```bash
bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

If you are on a multi-GPU server and want to select one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

Multi-turn chat:

```bash
bash inference/int4_quantized/run_chat.sh --image ./test_images/lesion.jpg
```

API service:

```bash
bash inference/int4_quantized/run_api.sh
```

Default API port: `5901`

The INT4 path uses:

- `bitsandbytes` 4-bit quantization
- `attn_implementation="sdpa"`
- the adapter-aware quantized model implementation in `inference/int4_quantized/`

## GPU Selection

You do not need to add `CUDA_VISIBLE_DEVICES=0` if the machine has only one visible GPU or if you are fine with the default CUDA device. 🧩

Use it only when you want to pin the process to a specific GPU, for example on a multi-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

The same pattern also works for:

- `inference/full_precision/run_infer.sh`
- `inference/full_precision/run_chat.sh`
- `inference/full_precision/run_api.sh`
- `inference/int4_quantized/run_chat.sh`
- `inference/int4_quantized/run_api.sh`

## API Endpoints

Both API services expose the same endpoints:

- `POST /v1/upload/{state_id}`
- `POST /v1/predict/{state_id}`
- `POST /v1/reset/{state_id}`
- `POST /diagnose/stream`
- `GET /health`

![SkinGPT-R1 Figure](figure.png)

## Which One To Use

- 🎯 Use `full_precision` when you want the original model path and best fidelity.
- ⚡ Use `int4_quantized` when GPU memory is tight or when you are on an environment where `flash-attn` is not the practical option.
