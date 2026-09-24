# SkinGPT-R1 training

This directory provides the supervised training implementation, the skin-colour-aware MoE adapter, the joint training objective, and the data-loading interface for SkinGPT-R1.

The released model is available at [Hugging Face](https://huggingface.co/yuhos16/SkinGPT-R1). The [model asset manifest](manifests/released_model.json) records an immutable repository revision and SHA-256 identifiers for the released weight shards. The same training source is distributed on [GitHub](https://github.com/yuhos16/SkinGPT-R1/tree/main/train).

## Implementation map

| Component | Source |
| --- | --- |
| Four adapter layers, eight experts, top-2 routing and three-class skin classifier | `src/llamafactory/model/skin_vlm_adapter.py` |
| Frozen backbone and trainable side branches | `src/llamafactory/model/loader.py` |
| SFT, cosine distillation, skin classification and routing losses | `src/llamafactory/train/sft/trainer.py` |
| Dataset registration and image/teacher-feature loading | `data/dataset_info.json`, `src/llamafactory/data/loader.py` |
| Preservation of structured labels and teacher features in batches | `src/llamafactory/data/converter.py`, `src/llamafactory/data/collator.py` |
| SFT workflow and command-line entry | `src/llamafactory/train/sft/workflow.py`, `src/train.py` |
| Manuscript-aligned SFT configuration | `configs/sft.yaml` |

The configuration follows the manuscript Methods: eight GPU workers, a per-device batch size of 4, gradient accumulation over 4 steps, an effective batch size of 128, a peak learning rate of 1e-4, 10 epochs, cosine decay and a warmup ratio of 0.03. The backbone is frozen and the adaptation branches are trained.

See [architecture and objective](docs/architecture.md) and the [data interface](data/README.md) for tensor flow and input conventions.

## Environment

Use Python 3.10 or a compatible Python environment with the versions listed in `environment/requirements.txt`. Install a PyTorch build appropriate for the CUDA runtime on the training host. FlashAttention-2 can be installed with `pip install flash-attn --no-build-isolation`; the workflow uses SDPA when FlashAttention-2 is unavailable.

```bash
python -m pip install -r environment/requirements.txt
```

## Training interface

Set paths to the base model, registered training JSON files, images and cached teacher features:

```bash
export SKINGPT_MODEL_DIR=/path/to/base_model
export SKINGPT_DATASET_DIR=/path/to/training_json
export SKINGPT_IMAGE_ROOT=/path/to/images
export SKINGPT_FEATURE_ROOT=/path/to/teacher_features
export SKINGPT_OUTPUT_DIR=/path/to/output
bash scripts/train.sh --dry-run
bash scripts/train.sh
```

`SKINGPT_PYTHON` selects the Python executable. `NPROC_PER_NODE` and `CUDA_VISIBLE_DEVICES` select the distributed workers and devices. Additional arguments are forwarded to the training parser. The configuration uses bfloat16, gradient checkpointing and a cosine learning-rate schedule. `skin_labels` and `teacher_feat` are retained through collation.

The synthetic record in `data/example_record.json` illustrates the input schema. Supply training assets under their applicable access conditions. The launcher checks the external asset locations before training.

## Checks

```bash
python scripts/check_package.py
python scripts/smoke_core.py
```

The first command checks source syntax, local imports and package hashes. The second exercises the adapter on synthetic tensors and checks finite gradients without loading pretrained weights or case data.

Third-party copyright and license notices are retained under [licenses](licenses/README.md).
