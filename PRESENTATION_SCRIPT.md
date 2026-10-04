# Progress Presentation Script: Genomic Language Model Project

## 1. Opening (30-45 seconds)
Good [morning/afternoon], Professor. I will present the current progress on my project, which is implementation of a DNA-based genomic language model from raw NCBI genome data, and its downstream use for repeat analysis and motif discovery.

The core objective is to learn contextual DNA representations using self-supervised masked language modeling, then evaluate whether those learned embeddings are biologically useful.

## 2. What We Implemented So Far (2-3 minutes)
I have completed an end-to-end baseline pipeline from raw genome sequence to downstream evaluation.

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
1. Implemented robustness sweep script over:
- repeat threshold
- random seeds
- sample sizes
2. Added CSV outputs for run-level and summary-level metrics.

### F. Motif discovery module
1. Implemented position-wise masking motif discovery.
2. Exported per-position probabilities P(A), P(C), P(G), P(T).
3. Computed uncertainty/confidence signals:
- true base probability
- surprise score
- entropy
- top1-top2 margin
4. Exported motif candidate tables and metadata.

### G. Motif report module
1. Implemented report generator for motif outputs.
2. Added:
- top surprising sites table
- top confident sites table
- probability heatmap across positions
- entropy and confidence trend plots
- top motif frequency plot

## 3. Current Results Summary (1.5-2 minutes)
### MLM training trend
Across extended training, validation loss and perplexity improved steadily, and masked accuracy improved from the initial epochs to later epochs, indicating stable learning of genomic context.

### Repeat detection result
The embedding-based repeat baseline produced strong performance:
- accuracy: 0.9100
- F1: 0.9041
- ROC-AUC: 0.9670
- PR-AUC: 0.9691

This suggests the learned embeddings already capture repeat-related structure effectively under the current labeling setup.

### Motif output status
Motif discovery outputs were generated successfully as per-position probability tables, and report artifacts are now available for structured interpretation.

## 4. What Is Completed vs Pending (45-60 seconds)
### Completed
1. Data pipeline and windowing
2. DNA MLM training baseline
3. Embedding extraction
4. Repeat detection baseline
5. Robustness sweep tooling
6. Motif discovery and motif reporting pipeline

### Pending
1. Biological validation of motif candidates against known motif databases/annotations
2. Multi-class repeat classification (LINE/SINE/LTR/etc.)
3. Coordinate-level overlap evaluation against external repeat annotation intervals
4. Variant scoring module using log-likelihood ratio

## 5. Next Steps and Approach From Now Onwards (2-3 minutes)
My approach from this point is to move from baseline performance to biological rigor and reproducibility.

### Phase A: Strengthen evidence quality (immediate)
1. Run robustness sweeps fully and report mean +- std across seeds/thresholds.
2. Add non-neural baselines (for example GC% and k-mer based classifier) for fair comparison.
3. Produce final evaluation tables and plots for repeat analysis section.

### Phase B: Improve biological relevance
1. Implement interval-level evaluation between predicted repeat regions and annotation intervals.
2. Extend from binary repeat detection to multi-class repeat classification.
3. Stratify motif discovery outputs by repeat-enriched vs non-repeat windows.

### Phase C: Extend to variant analysis
1. Implement LLR-based scoring for SNVs from the trained model.
2. Compare variant scoring behavior on curated variant sets.

### Phase D: Controlled experimentation
1. Architecture comparison: CNN baseline vs small Transformer.
2. Context-length experiments: 256 vs 512 vs 1024.
3. Masking-rate experiments: 10%, 15%, 20%.
4. Dataset-size scaling experiments.

## 6. Expected Deliverables for Next Review (30-45 seconds)
By the next review, I plan to provide:
1. Robustness summary tables (mean +- std) for repeat detection.
2. Baseline comparison against non-neural features.
3. First interval-overlap evaluation results.
4. Multi-class repeat classification prototype.
5. Initial variant scoring module design and pilot output.

## 7. Closing (20-30 seconds)
In summary, the project baseline is now fully implemented end-to-end and producing strong early downstream results. The next phase focuses on rigorous validation, biologically meaningful evaluation, and expansion to multi-class repeat and variant-effect tasks.

Thank you. I would appreciate feedback on prioritization between multi-class repeat classification and variant scoring for the next milestone.
