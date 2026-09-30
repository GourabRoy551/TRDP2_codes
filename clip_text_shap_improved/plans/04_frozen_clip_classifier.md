# Part 4 — Frozen CLIP Sentiment Classification Head

## Purpose

Create a CLIP-based sentiment model whose outputs are genuine learned NEG/POS logits and are therefore more comparable with the fine-tuned BERT classifier.

## Planned work

1. Freeze all CLIP Text Encoder parameters.
2. Encode the sentiment training, validation and test splits.
3. Train a deterministic linear NEG/POS head using training embeddings only.
4. Select regularization and stopping criteria using validation data only.
5. Save the head weights, label map, preprocessing configuration and split manifest.
6. Evaluate the zero-shot prototype model and trained-head model separately.
7. Apply the approved Partition SHAP masking method to the trained logits.

The classifier will have the following form:

```text
sentiment logits for sentence x
    = classifier weights × frozen CLIP embedding + classifier bias

z(x) = W * e(x) + b
```

Here, `e(x)` is the frozen CLIP embedding. Only the classifier weights `W` and
classifier bias `b` are learned.

## Required tests

- CLIP parameters remain unchanged during training;
- saved and reloaded heads produce identical logits;
- split boundaries prevent leakage;
- class order remains `[NEG, POS]`;
- SHAP reconstructs both trained logits;
- training is reproducible with the saved seed.

## Deliverables

- saved classifier and training configuration;
- learning and validation curves;
- classification tables and confusion matrix;
- trained-logit SHAP plots;
- zero-shot versus trained-head comparison;
- short Part 4 conclusion.

## Stop condition

Do not launch the SHAP-budget study until the classifier has been approved.

