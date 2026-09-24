# Inference

The evaluation example uses the diagnostic prompts and settings in [`docs/evaluation_prompts_and_settings.md`](../docs/evaluation_prompts_and_settings.md). It keeps system and user messages separate and loads the existing `SkinVLModelWithAdapter` architecture, including its auxiliary model components.

## Install

Use the repository environment with a CUDA-enabled PyTorch build:

```bash
conda env create -f environment.yml
conda activate skingpt-r1
python -m pip install flash-attn --no-build-isolation
```

FlashAttention-2 requires a compatible GPU, CUDA toolkit, and PyTorch build. Use `--attn-implementation sdpa` when selecting the PyTorch attention implementation. This changes the attention setting recorded with the run. See the [Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl) for attention and image processing requirements.

## Single-image example

Run from the repository root. Point `--model-path` to the local `checkpoint` directory containing the actual weight shards and processor files.

```bash
CUDA_VISIBLE_DEVICES=0 python -m inference.evaluation.run_inference --model-path ./checkpoint --image /path/to/lesion.jpg --mode ddi --output outputs/lesion.json
```

The system message is:

```text
You are a dermatology assistant. Analyze the supplied skin-lesion image, produce a concise hierarchical clinical rationale based on observable image evidence, consider plausible differential diagnoses, and state the final diagnosis. Do not invent findings that are not supported by the image.
```

The user message contains the image and `prompts/ddi_user.txt`. No reference diagnosis or candidate list is supplied in this mode.

For runs with the learned vocabulary bias, add `--skin-vocab-mask /path/to/original_mask.json`. This must be the original binary training-vocabulary mask with one entry per model vocabulary position. The output records whether it was applied and its file digest. The example does not generate a mask from the tested image or reference label.

## Candidate-label classification

This mode has no system message. The supplied JSON file defines the candidate labels and their order.

```bash
CUDA_VISIBLE_DEVICES=0 python -m inference.evaluation.run_inference --model-path ./checkpoint --image /path/to/lesion.jpg --mode classification --labels prompts/labels_160case.json --output outputs/classification.json
```

The vocabulary file provides the 23 candidate diagnoses used in the 160-case comparison. The original candidate order varied by case. Supply the intended ordering for a specific evaluation request.

## Runtime controls

| Control | Default |
| --- | --- |
| Precision | bfloat16 |
| Attention | FlashAttention-2 |
| TF32 | Enabled for CUDA matrix multiplication |
| Padding | Left |
| Image pixels | 3136 to 1003520 |
| Batch size | 16 images per worker |
| Maximum new tokens | 4096 |
| Sampling | Temperature 0.7, top-p 0.9 |
| Repetition penalty | 1.0 |
| Seed | 42 |
| Cache | Enabled |

Repeat `--image` to supply multiple images. The example processes them in batches on one visible GPU. `--batch-size`, `--max-new-tokens`, `--seed`, and `--attn-implementation` override their configuration values. `--greedy` disables sampling. The output includes messages, effective settings, generation configuration, model configuration digest, and runtime versions. It records full clinical model outputs locally under the path you choose.

Inspect a request before loading the model:

```bash
python -m inference.evaluation.run_inference --image /path/to/lesion.jpg --mode ddi --dry-run
```

Dry runs use the standard library and do not load weights or require a GPU. Actual inference requires local weight tensors and fails explicitly if a shard is absent or is only a Git LFS pointer. This entry point does not download checkpoints.

## Other interfaces

The existing `full_precision/` and `int4_quantized/` directories provide interactive chat and FastAPI examples. Their interactive prompts and generation defaults are separate from the evaluation configuration. Both use `./checkpoint` as the default model path.

| Interface | Command from the repository root |
| --- | --- |
| Full-precision chat | `bash inference/full_precision/run_chat.sh --image /path/to/lesion.jpg` |
| INT4 chat | `bash inference/int4_quantized/run_chat.sh --image /path/to/lesion.jpg` |
| Full-precision API | `bash inference/full_precision/run_api.sh` |
| INT4 API | `bash inference/int4_quantized/run_api.sh` |

The API endpoints include `/v1/upload/{state_id}`, `/v1/predict/{state_id}`, `/v1/reset/{state_id}`, `/diagnose/stream`, and `/health`. Default ports are 5900 for the full-precision service and 5901 for INT4. See each entry point for its request schema.
