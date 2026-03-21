# Trustworthy and Fair SkinGPT-R1 for Democratizing Dermatological Reasoning across Diverse Ethnicities

![CUHKSZ Logo](cuhksz-logo.png)

[![PDF](https://img.shields.io/badge/PDF-arXiv%3A2511.15242-B31B1B)](https://arxiv.org/abs/2511.15242)

SkinGPT-R1 is a dermatological reasoning vision language model. 🩺✨

**The Chinese University of Hong Kong, Shenzhen**

## Updates

- We will soon release the SkinGPT-R1-7B weights.

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

`environment.yml` is a Conda environment definition file for reproducing the recommended runtime environment.

From scratch:

```bash
git clone https://huggingface.co/yuhos16/SkinGPT-R1
cd SkinGPT-R1
conda env create -f environment.yml
conda activate skingpt-r1
```

Manual setup:

```bash
git clone https://huggingface.co/yuhos16/SkinGPT-R1
cd SkinGPT-R1
conda create -n skingpt-r1 python=3.10.20 -y
conda activate skingpt-r1
pip install -r requirements.txt
```

## Quick Start

1. Use the repository `./checkpoints` directory as the model weights directory.

2. Prepare a test image, for example `./test_images/lesion.jpg`.

3. Run a first test.

Full precision:

```bash
bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg
```

INT4:

```bash
bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg
```

## Usage

| Mode | Full Precision | INT4 Quantized |
| --- | --- | --- |
| Single image | `bash inference/full_precision/run_infer.sh --image ./test_images/lesion.jpg` | `bash inference/int4_quantized/run_infer.sh --image_path ./test_images/lesion.jpg` |
| Multi-turn chat | `bash inference/full_precision/run_chat.sh --image ./test_images/lesion.jpg` | `bash inference/int4_quantized/run_chat.sh --image ./test_images/lesion.jpg` |
| API service | `bash inference/full_precision/run_api.sh` | `bash inference/int4_quantized/run_api.sh` |

Default API ports:

- Full precision: `5900`
- INT4 quantized: `5901`

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

## Disclaimer

This project is for **research and educational use only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. ⚠️

## License

This repository is released under the **MIT License**.
See [LICENSE](LICENSE) for details.
