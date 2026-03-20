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

This repo provides full-precision inference, INT4 quantized inference, multi-turn chat, and FastAPI serving.

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

## Quick Start

1. Clone the repository.

```bash
git clone <your-repo-url>
cd SkinGPT-R1
```

2. Create the environment.

```bash
conda env create -f environment.yml
conda activate skingpt-r1
```

3. Put your model weights into:

```text
./checkpoints/full_precision
./checkpoints/int4
```

4. Prepare a test image, for example `./test_images/lesion.jpg`.

5. Run a first test.

Full precision:

```bash
bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg
```

INT4:

```bash
bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

## Usage

### Full Precision

Single image

```bash
bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg
```

Multi-turn chat

```bash
bash inference/full_precision/run_chat.sh --image ./test_images/lesion.jpg
```

API service

```bash
bash inference/full_precision/run_api.sh
```

Default API port: `5900`

### INT4 Quantized

Single image

```bash
bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

Multi-turn chat

```bash
bash inference/int4_quantized/run_chat.sh --image ./test_images/lesion.jpg
```

API service

```bash
bash inference/int4_quantized/run_api.sh
```

Default API port: `5901`

Notes

- On multi-GPU servers, prepend commands with `CUDA_VISIBLE_DEVICES=0` if you want to pin one GPU.
- RTX 50 series should use the default `sdpa` path.
- A100 / RTX 3090 / RTX 4090 / H100 can also try `flash_attention_2` if their CUDA stack supports it.

## API Endpoints

Both API services expose the same endpoints:

- `POST /v1/upload/{state_id}`
- `POST /v1/predict/{state_id}`
- `POST /v1/reset/{state_id}`
- `POST /diagnose/stream`
- `GET /health`

![SkinGPT-R1 Figure](figure.png)

## License

This repository is released under **CC BY-NC-SA 4.0**.
See [LICENSE](LICENSE) for details.
