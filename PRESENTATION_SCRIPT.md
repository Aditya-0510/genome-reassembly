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

## 3. Current Results Summary (3-4 minutes)
### MLM training trend
Across 12 epochs, validation loss improved from 1.1567 to 1.0734, masked accuracy improved from 0.4878 to 0.5201, and perplexity improved from 3.18 to 2.93 — indicating stable learning of genomic context with a plateau forming in later epochs.

### Repeat detection result (gLM embeddings)
- Accuracy: 0.9100
- F1: 0.9041
- ROC-AUC: 0.9670
- PR-AUC: 0.9691

### Non-neural baseline result (GC% + k-mer, for comparison)
- Accuracy: 0.7600
- F1: 0.7458
- ROC-AUC: 0.8280
- PR-AUC: 0.7736

The ~14-point ROC-AUC gap (0.967 vs 0.828) in favor of the gLM embeddings shows the model learned representations beyond simple sequence composition — this is direct evidence the self-supervised training added value.

### Robustness sweep result (5 seeds x 4 thresholds)
- Accuracy mean/std range across thresholds: 0.874-0.912, std consistently under 1%.
- ROC-AUC mean range: 0.949-0.969, std under 0.6%.
- Low variance across seeds/thresholds confirms the repeat-detection result is stable, not a lucky split.

### Interval-level (base-pair) evaluation result
- Precision: 0.5635
- Recall: 0.8255
- IoU: 0.5035
- F1: 0.6698

This is notably stricter than the 91% window-level accuracy. It shows the model finds most true repeat regions (high recall) but over-calls some boundaries (moderate precision) — a genuine, honestly-reported limitation at base-pair resolution.

### Variant LLR scoring pilot result (200 auto-sampled SNVs)
- Mean LLR: 0.7129 nats
- Mean |LLR|: 1.0653 nats
- LLR values ranged from near-zero (interchangeable positions, e.g. ~-0.002 nats) to strongly constrained (e.g. 5.47 nats / 7.89 bits for one T-to-C substitution), showing the model differentiates tolerant vs. constrained genomic positions.

### Motif output status
Motif discovery outputs were generated successfully as per-position probability tables, and report artifacts are available for structured interpretation. Mean entropy (1.75 bits, below the 2-bit max) indicates non-random structure in predictions.

## 4. What Is Completed vs Pending vs Blocked (60-90 seconds)
### Completed
1. Data pipeline and windowing
2. DNA MLM training baseline
3. Embedding extraction
4. Repeat detection baseline (gLM embeddings)
5. Non-neural (GC%/k-mer) baseline for fair comparison
6. Robustness sweep tooling, executed with full results
7. Interval-level (base-pair) overlap evaluation
8. Variant LLR scoring module with pilot run
9. Motif discovery and motif reporting pipeline

### Pending (achievable with more time, no new data needed)
1. Biological validation of motif candidates against known TF motif databases
2. Stratifying motif discovery outputs by repeat-enriched vs non-repeat windows
3. Architecture comparison (CNN vs small Transformer), context-length and masking-rate experiments
4. Scoring variant LLRs against a curated, biologically meaningful variant set (e.g. known pathogenic/benign SNVs) instead of only an auto-sampled pilot

### Blocked (requires external data not present in this workspace)
1. Multi-class repeat classification (LINE/SINE/LTR/etc.) — requires RepeatMasker-style class-labeled annotations (a `.out` file or `repeat_region` GFF features). The current dataset only contains soft-masked (lowercase) binary repeat signal, not repeat-family labels. This is a data-availability blocker, not an implementation gap.

## 5. Next Steps and Approach From Now Onwards (2-3 minutes)
With rigor and comparison tooling now in place, the next phase focuses on biological depth and controlled experimentation.

### Phase A: Biological relevance (immediate)
1. Validate top motif candidates against known TF motif databases (e.g. JASPAR) where feasible.
2. Stratify motif discovery by repeat-enriched vs non-repeat windows.
3. If a RepeatMasker annotation file becomes available, add multi-class repeat classification.

### Phase B: Variant analysis depth
1. Apply the variant scoring module to a curated, biologically meaningful variant set.
2. Compare LLR distributions between known functional and neutral variants if labels become available.

### Phase C: Controlled experimentation
1. Architecture comparison: CNN baseline vs small Transformer.
2. Context-length experiments: 256 vs 512 vs 1024.
3. Masking-rate experiments: 10%, 15%, 20%.
4. Dataset-size scaling experiments.

## 6. Expected Deliverables for Next Review (30-45 seconds)
By the next review, I plan to provide:
1. Motif validation results against a reference TF motif database.
2. Repeat-stratified motif discovery comparison.
3. Variant LLR scoring results on a curated variant set (pending data availability).
4. Initial architecture/context-length/masking-rate experiment results.

## 7. Closing (20-30 seconds)
In summary, the project pipeline is now fully implemented end-to-end, validated with a non-neural baseline comparison, stress-tested with a multi-seed robustness sweep, evaluated at both window and base-pair resolution, and extended to variant-level scoring. The next phase focuses on biological validation and controlled architecture experiments.

Thank you. I would appreciate feedback on whether to prioritize obtaining RepeatMasker class annotations for multi-class repeat classification, or deepening the variant scoring analysis with a curated variant set.
