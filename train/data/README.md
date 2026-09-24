# Training input interface

Each registered JSON file contains records with `messages`, `images`, and `skin_labels`:

- `messages`: a list of `role` and `content` pairs, with image placeholders where applicable.
- `images`: relative paths resolved beneath `SKINGPT_IMAGE_ROOT`.
- `skin_labels`: an integer, with `0 = Light`, `1 = Medium`, `2 = Dark`.

`example_record.json` is a synthetic schema illustration. `dataset_info.json` registers the training-file interface. It contains file registrations rather than case data.

For each image, the loader resolves the teacher feature under `SKINGPT_FEATURE_ROOT` using the same relative path with the image suffix changed to `.npy`. The stored feature is flattened to `teacher_feat`. The SFT interface uses 1,024-dimensional teacher features. Check that each image has the intended teacher feature before training; the supplied loader uses a zero vector when a feature cannot be read.

The converter and collator preserve `skin_labels` and `teacher_feat`. The integer `skin_labels` value supervises the skin classifier. The classes describe apparent skin colour in the image. Predicted skin-colour probabilities condition expert routing.

```text
SKINGPT_DATASET_DIR/
  dataset_info.json
  train.json
SKINGPT_IMAGE_ROOT/
  <relative image paths>
SKINGPT_FEATURE_ROOT/
  <corresponding relative .npy paths>
```
