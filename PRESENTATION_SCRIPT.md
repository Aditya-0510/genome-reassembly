# Progress Presentation Script: Genomic Language Model Project

## 1. Opening (30-45 seconds)
Good [morning/afternoon], Professor. I will present the current progress on my project, which is implementation of a DNA-based genomic language model from raw NCBI genome data, and its downstream use for repeat analysis, motif discovery, and variant scoring.

The core objective is to learn contextual DNA representations using self-supervised masked language modeling, then rigorously evaluate whether those learned embeddings are biologically useful — including fair comparisons against non-neural baselines and stricter base-pair-resolution evaluation.

## 2. What We Implemented So Far (3-4 minutes)
I have completed an end-to-end pipeline from raw genome sequence to downstream evaluation, including rigor and comparison tooling added in this latest pass.

### A. Data and preprocessing pipeline
1. Used genome FASTA from NCBI data package.
2. Built fixed-length window generation with genomic coordinates preserved.
3. Applied sequence quality filtering based on A/C/G/T fraction.
4. Kept coordinate metadata for mapping predictions back to genome positions.

### B. DNA language model setup
1. Implemented DNA tokenizer with vocabulary for A, C, G, T plus MASK/PAD/N.
2. Implemented masked language modeling objective.
3. Built a dilated 1D CNN genomic language model inspired by GPN-style architecture.
4. Added full training pipeline with validation metrics and checkpointing.

### C. Training instrumentation and outputs
1. Added detailed progress logging during training.
2. Saved epoch-wise history including:
- train loss
- validation loss
- masked accuracy
- perplexity
- epoch runtime
3. Saved best model checkpoint automatically.

### D. Embedding extraction and repeat baseline
1. Implemented embedding extraction from trained model.
2. Built binary repeat/non-repeat baseline classifier from embeddings.
3. Added detailed run-time logs for extraction and repeat baseline scripts.

### E. Robustness and reproducibility tools
1. Implemented robustness sweep script over repeat threshold, random seeds, and sample sizes.
2. Added CSV outputs for run-level and summary-level (mean +- std) metrics.
3. Executed the full sweep: 4 thresholds x 5 seeds, all results saved.

### F. Non-neural baseline for fair comparison
1. Implemented a GC-content + k-mer frequency classifier with no learned embeddings.
2. Trained and evaluated it on the same windows/labels as the gLM-embedding baseline.
3. Confirms whether the learned representations add value beyond simple sequence composition.

### G. Interval-level (base-pair resolution) evaluation
1. Implemented merging of predicted-positive windows into genomic intervals.
2. Extracted true repeat intervals directly from soft-masked (lowercase) FASTA regions.
3. Computed base-pair-level precision, recall, IoU, and F1 between predicted and true intervals.
4. This is a stricter test than window-level classification accuracy.

### H. Variant LLR scoring module
1. Implemented masked-position log-likelihood-ratio (LLR) scoring for SNVs: mask the variant site, compare P(ref) vs P(alt).
2. Supports curated variant CSV input or auto-sampled pilot variant sets.
3. Ran a 200-variant pilot scoring pass using the trained checkpoint.

### I. Motif discovery module
1. Implemented position-wise masking motif discovery.
2. Exported per-position probabilities P(A), P(C), P(G), P(T).
3. Computed uncertainty/confidence signals: true base probability, surprise score, entropy, top1-top2 margin.
4. Exported motif candidate tables and metadata.

### J. Motif report module
1. Implemented report generator for motif outputs.
2. Added top surprising/confident site tables, probability heatmap, entropy/confidence trend plots, and top motif frequency plot.

### K. Scaling vs. architecture diagnostic experiment
1. Trained a scaled-up configuration (10x more data, bigger model, longer training, better optimization) to test whether scale alone improves accuracy.
2. Found scaling produced no improvement, which led to auditing the model architecture directly.
3. Identified a receptive-field bug in the dilation schedule — it was capped at a fixed maximum regardless of network depth, so adding layers did not extend context.
4. Fixed the dilation schedule to grow exponentially with depth and retrained with the same data/epoch budget as the scaled run.
5. The architecture fix improved every single downstream metric, isolating the real bottleneck from the false lead of "needs more scale."

## 3. Current Results Summary (4-5 minutes)
### Three-way comparison: baseline (v1) vs scaled (v2) vs architecture-fixed (v3)

| Metric | v1 (baseline) | v2 (10x data, bigger model) | v3 (dilation fixed) |
|---|---|---|---|
| Best val loss | 1.0734 | 1.0754 | **1.0267** |
| Best val masked accuracy | 0.5201 | 0.5143 | **0.5418** |
| Best val perplexity | 2.9253 | 2.9311 | **2.7918** |
| Repeat ROC-AUC (threshold 0.2) | 0.9687 | 0.9662 | **0.9719** |
| Repeat accuracy (threshold 0.2) | 0.9114 | 0.9024 | **0.9232** |
| Interval-level F1 | 0.6698 | 0.6651 | **0.6753** |
| Motif mean entropy (bits) | 1.7534 | 1.7648 | **1.6846** |

The story: scaling up data/parameters/epochs (v2) did **not** improve anything over the baseline — in fact it was marginally worse on most metrics. That ruled out "needs more data/compute" as the explanation. Auditing the architecture found the real issue: the dilation schedule `2^(i mod 4)` cycles `1,2,4,8,1,2,...` and resets every 4 layers, capping the receptive field at ~37bp regardless of depth. Fixing it to grow exponentially (`2^i`) extended the receptive field to ~1023bp — enough to see the full 512bp window — and v3 now wins on every metric across every evaluation pipeline.

### Non-neural baseline result (GC% + k-mer, for comparison)
- Accuracy: 0.7600, F1: 0.7458, ROC-AUC: 0.8280, PR-AUC: 0.7736

The gap between this composition-only baseline (ROC-AUC 0.83) and the best gLM embeddings (v3, ROC-AUC 0.97) shows the model learned representations beyond simple sequence composition.

### Robustness sweep result (5 seeds x 4 thresholds, v3)
- Accuracy mean/std range across thresholds: 0.890-0.936, consistently low std (under 1%).
- ROC-AUC mean range: 0.955-0.972, std under 0.9%.
- v3 beats v1 and v2 at every threshold tested, not just one setting.

### Interval-level (base-pair) evaluation result (v3)
- Precision: 0.5647, Recall: 0.8398, IoU: 0.5098, F1: 0.6753

This is notably stricter than the ~92% window-level accuracy. v3 improved recall the most — it finds more of the true repeat region — but precision (~0.56) stayed similar across all three configs, meaning boundary over-calling is a separate, still-open limitation independent of the receptive-field fix.

### Variant LLR scoring pilot result (200 auto-sampled SNVs, v3)
- LLR values ranged from near-zero (interchangeable positions) to strongly constrained sites (e.g. 5.48 nats / 7.91 bits for a C-to-G substitution) — v3 shows the sharpest separation of the three configs, consistent with its lower overall entropy.

### Motif output status
Motif discovery outputs were generated for all three configurations; v3 shows the highest confidence signal (true-base probability 0.360 vs 0.335/0.330 for v1/v2, entropy 1.68 vs 1.75/1.76 bits).

## 4. What Is Completed vs Pending vs Blocked (60-90 seconds)
### Completed
1. Data pipeline and windowing
2. DNA MLM training baseline, plus a scaled variant and an architecture-fixed variant (v1/v2/v3)
3. Embedding extraction
4. Repeat detection baseline (gLM embeddings)
5. Non-neural (GC%/k-mer) baseline for fair comparison
6. Robustness sweep tooling, executed with full results across all three model versions
7. Interval-level (base-pair) overlap evaluation across all three model versions
8. Variant LLR scoring module with pilot runs across all three model versions
9. Motif discovery and motif reporting pipeline
10. Diagnosed and fixed a receptive-field bug in the model architecture (dilation schedule), validated by re-running the full evaluation suite

### Pending (achievable with more time, no new data needed)
1. Biological validation of motif candidates against known TF motif databases
2. Investigating the persistent interval-level precision gap (~0.56) independent of receptive field — e.g. threshold tuning or boundary post-processing
3. Further architecture experiments (e.g., adding self-attention) now that the dilation bug is fixed and no longer confounds comparisons
4. Scoring variant LLRs against a curated, biologically meaningful variant set (e.g. known pathogenic/benign SNVs) instead of only an auto-sampled pilot

### Blocked (requires external data not present in this workspace)
1. Multi-class repeat classification (LINE/SINE/LTR/etc.) — requires RepeatMasker-style class-labeled annotations (a `.out` file or `repeat_region` GFF features). The current dataset only contains soft-masked (lowercase) binary repeat signal, not repeat-family labels. This is a data-availability blocker, not an implementation gap.

## 5. Next Steps and Approach From Now Onwards (2-3 minutes)
With the architecture bottleneck diagnosed and fixed, the next phase focuses on biological depth and the remaining open limitation (interval precision).

### Phase A: Biological relevance (immediate)
1. Validate top motif candidates against known TF motif databases (e.g. JASPAR) using v3's higher-confidence outputs.
2. Stratify motif discovery by repeat-enriched vs non-repeat windows.
3. If a RepeatMasker annotation file becomes available, add multi-class repeat classification.

### Phase B: Interval precision follow-up
1. Investigate why predicted repeat intervals over-extend beyond true soft-masked boundaries even after the receptive-field fix.
2. Try probability-threshold tuning or a boundary-smoothing post-processing step.

### Phase C: Variant analysis depth
1. Apply the variant scoring module to a curated, biologically meaningful variant set.
2. Compare LLR distributions between known functional and neutral variants if labels become available.

### Phase D: Further controlled experimentation
1. Architecture comparison: current fixed-dilation CNN vs a small Transformer/self-attention block.
2. Context-length experiments: 256 vs 512 vs 1024, now with a receptive field that properly scales with window size.
3. Masking-rate experiments: 10%, 15%, 20%.

## 6. Expected Deliverables for Next Review (30-45 seconds)
By the next review, I plan to provide:
1. Motif validation results against a reference TF motif database, using the v3 checkpoint.
2. Investigation results for the interval-level precision gap.
3. Variant LLR scoring results on a curated variant set (pending data availability).
4. Initial results from a self-attention architecture experiment, now that the dilation bug no longer confounds the comparison.

## 7. Closing (20-30 seconds)
In summary, the project pipeline is fully implemented end-to-end, and this round of work produced a genuine engineering win: a scaling experiment that initially looked like a dead end led to discovering and fixing a real receptive-field bug in the model architecture, which then improved every downstream metric — training accuracy, repeat detection, interval overlap, and motif confidence. The next phase focuses on biological validation and closing the remaining interval-precision gap.

Thank you. I would appreciate feedback on whether to prioritize obtaining RepeatMasker class annotations for multi-class repeat classification, or investigating the interval-precision limitation further.
