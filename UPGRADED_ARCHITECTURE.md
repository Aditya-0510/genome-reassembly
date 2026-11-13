# Upgraded Architecture: Token-Level Repeat Detection

## Overview

This architecture upgrades repeat detection from coarse window-level classification to token-level (per-base) prediction using a dedicated neural model. The serving pipeline is model-only (no non-neural heuristic baseline in inference).

## High-Level Flow

1. User DNA input enters FastAPI web layer.
2. Input is validated and normalized.
3. Sequence is split into overlapping windows.
4. TokenRepeatCNN predicts per-base logits for each window.
5. Sigmoid converts logits to probabilities and overlapping windows are aggregated.
6. Per-base scores are thresholded and short runs are filtered.
7. Positive bases are merged into repeat intervals.
8. Metrics are computed (`repeat_bp`, `repeat_fraction`, repeated/not repeated).
9. Interval-level counts are computed (count in full input, count in predicted intervals).
10. Top repeated segments are summarized via k-mer frequency.
11. UI/API response is returned.

## Model Design (New)

Model: TokenRepeatCNN

- Input: DNA token IDs per sequence window
- Backbone: embedding + dilated 1D convolutional stack
- Output: one logit per base position
- Objective: weighted binary cross-entropy over per-base repeat labels

Why this is better:

- Directly predicts repeat/non-repeat at base resolution
- Produces cleaner interval boundaries after thresholding and merging
- Aligns better with required outputs (which sequence repeats, where, and how much)

## Training Pipeline

1. Read FASTA and generate windows (`window_size`, `stride`)
2. Build per-base labels from soft-masked reference sequence (lowercase => repeat token)
3. Train with class-imbalance-aware weighted BCE loss (`pos_weight` auto or user-specified)
4. Save best checkpoint by validation F1

## Inference Pipeline

1. Normalize and validate input sequence
2. Create overlapping windows
3. Predict per-base probabilities per window
4. Aggregate overlapping scores to one score per base
5. Apply probability threshold and minimum interval length
6. Merge contiguous positives into final repeat intervals
7. Compute output metrics and interval-level counts

## Output Contract

The response includes:

1. sequence_summary
- model type, thresholds, interval counts, total repeat bp, repeat fraction, repeated/not repeated

2. repeat_intervals
- start, end, length, sequence
- occurrence_count_in_input
- occurrence_count_in_predicted_intervals

3. repeat_windows
- per-window average repeat probability and repeat flag

4. repeated_segments
- k-mer summary with count and first position

5. query_result (optional)
- user subsequence counts in whole sequence and predicted intervals

## Notes and Limitations

- Current labels are weak labels from soft-masked case information in FASTA.
- Synthetic uppercase repeats may still require threshold tuning or augmentation-aware retraining.
- For stronger biological ground truth, replace case-derived labels with curated repeat annotation intervals.
