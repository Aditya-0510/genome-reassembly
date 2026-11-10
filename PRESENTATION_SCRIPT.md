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

### L. Real-world data integration (RepeatMasker, JASPAR, ClinVar)
1. Obtained real RepeatMasker annotations, the JASPAR 2024 TF motif database, and a ClinVar variant set, and wired all three into the pipeline.
2. Replaced the binary soft-mask repeat proxy with real multi-class repeat-family labels (SINE/LINE/LTR/DNA/etc.) for a genuine multi-class classifier.
3. Replaced the hand-curated IUPAC motif approximation with real position-weight-matrix scoring against JASPAR.
4. Built a curated, balanced ClinVar SNV set (Pathogenic/Likely_pathogenic/Benign/Likely_benign) and compared model LLR scores against real clinical-significance labels.

## 3. Current Results Summary (5-6 minutes)
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

### Multi-class repeat classification (real RepeatMasker labels, v3)
- Overall accuracy: 70.4% across 7 real repeat families (SINE, LINE, LTR, DNA, Simple_repeat, Low_complexity, Non_repeat).
- Strong on abundant classes: SINE F1 0.85, Non_repeat F1 0.78.
- Weak on rare classes: DNA F1 0.08, Low_complexity F1 0.22 — a class-imbalance problem, not a fundamental signal failure.

### Real JASPAR motif validation
- 606 candidate-to-JASPAR matches found at ≥80% of max PWM score, across 200 candidates screened against 2346 real TF motifs.
- Recognizable transcription factor names matched (GATA4, GATA15, Deaf1, DOF-family) — genuine external validation, not just internal ranking.

### ClinVar variant LLR comparison (real pathogenicity labels, v3)
- 800 real ClinVar SNVs scored (200 each: Pathogenic, Likely_pathogenic, Benign, Likely_benign).
- Mean LLR: Likely_benign 0.199, Likely_pathogenic 0.010, Benign 0.009, Pathogenic -0.047.
- **Honest negative result:** the difference between Pathogenic and Benign means is small relative to the ~0.9 standard deviation, and runs in the opposite direction from what would be expected biologically. The model currently shows no significant, correctly-directioned pathogenicity signal on this pilot — an important, defensible limitation to report rather than hide.

## 4. What Is Completed vs Pending (60-90 seconds)
### Completed
1. Data pipeline and windowing
2. DNA MLM training baseline, plus a scaled variant and an architecture-fixed variant (v1/v2/v3)
3. Embedding extraction
4. Repeat detection baseline (gLM embeddings)
5. Non-neural (GC%/k-mer) baseline for fair comparison
6. Robustness sweep tooling, executed with full results across all three model versions
7. Interval-level (base-pair) overlap evaluation across all three model versions, plus a probability-threshold sweep
8. Variant LLR scoring module with pilot runs across all three model versions
9. Motif discovery and motif reporting pipeline
10. Diagnosed and fixed a receptive-field bug in the model architecture (dilation schedule), validated by re-running the full evaluation suite
11. Multi-class repeat classification using real RepeatMasker annotations
12. Motif validation using real JASPAR position-weight matrices
13. Variant LLR comparison using real curated ClinVar pathogenicity labels

### Pending (achievable with more time, no new data needed)
1. Investigate class imbalance to improve minority repeat-family classification (DNA, LTR, Simple_repeat, Low_complexity)
2. Widen motif discovery beyond 9bp to screen the ~39% of JASPAR motifs currently skipped for being too wide
3. Scale the ClinVar comparison beyond 800 chromosome-1 variants and apply a formal statistical test before drawing strong conclusions
4. Further architecture experiments (e.g., adding self-attention) now that the dilation bug is fixed and no longer confounds comparisons

## 5. Closing (20-30 seconds)
In summary, the project pipeline is fully implemented end-to-end, and this round of work produced two genuine wins: first, a scaling experiment that initially looked like a dead end led to discovering and fixing a real receptive-field bug in the model architecture, improving every downstream metric; second, integrating real RepeatMasker, JASPAR, and ClinVar data turned every previously-blocked or approximated analysis into a genuine, defensible result — including an honest negative finding on variant pathogenicity scoring, which I'm reporting transparently rather than omitting. The next phase focuses on closing the minority-class and variant-signal gaps.

Thank you.