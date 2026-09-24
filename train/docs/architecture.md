# Training architecture and objective

The implementation uses a Qwen2.5-VL-compatible backbone with a separate visual adaptation branch. Processor patches are projected to 1,024 dimensions. A mean-pooled image representation feeds a three-class skin-colour classifier. Four residual MoE layers each contain eight bottleneck experts of width 64 and select two experts per patch.

The router adds the visual projection to a projection of the predicted skin-colour probabilities, applies softmax, selects the top two experts, and renormalizes their weights. Structured `skin_labels` supervise the classifier. The routing input is the predicted probability vector.

The pooled visual embedding passes through a `1024 -> 64 -> hidden_size` bias branch. The shared language-model head maps it into vocabulary space. A learned scale controls its addition to the language logits at supervised token positions.

## Optimization

The freeze-loading path trains the projection/distillation branch, skin classifier, MoE experts, routing projections, text-bias branch and learned bias scale. The backbone and shared language-model head are frozen in this SFT configuration.

The custom trainer returns:

```text
L = alpha * L_sft + beta * L_distill + gamma * L_aux + delta * L_skin

L_sft     = shifted-token cross entropy, ignoring label -100
L_distill = 10 * mean(1 - cosine_similarity(student_feature, teacher_feature))
L_skin    = cross entropy(skin_logits, skin_labels)
L_aux     = sum of routing-balancing terms across images and adapter layers

alpha = 1.0; beta = 0.1; gamma = 0.001; delta = 0.1
```

Each balancing term is the number of experts multiplied by the sum of the products of mean routing probabilities and mean top-2 selection indicators. The four coefficients are configurable through `SFT_WEIGHT`, `DISTILL_WEIGHT`, `MOE_AUX_WEIGHT` and `SKIN_LOSS_WEIGHT`.

`CustomSeq2SeqTrainer.compute_loss` recomputes the supervised loss from the biased logits and returns the combined objective. It does not add the wrapper's `outputs.loss` a second time. The data loader and collator carry `teacher_feat` and `skin_labels` as separate fields.

## Labels and model identity

The structured label mapping is `0 = Light`, `1 = Medium`, `2 = Dark`. These classes describe apparent skin colour in the image. The integer `skin_labels` field supplies the classifier target. The classifier predicts skin-colour probabilities from visual features, and these probabilities condition expert routing.

The asset manifest identifies the released evaluation checkpoint. The SFT source, configuration, and weight assets have separate identities so that a source revision and a model revision can each be cited explicitly.
