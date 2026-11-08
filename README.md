# Genome Reassembly: DNA Genomic Language Model Baseline

This repository contains a runnable baseline for training a DNA masked language model (MLM) directly from genome FASTA files.

## What is implemented

- Fixed-length window generation from genomic FASTA
- DNA tokenizer with vocabulary `A, C, G, T, [MASK], [PAD]`
- Random masking for MLM training
- Dilated 1D CNN model inspired by GPN-style local-context modeling
- Training loop with validation metrics (loss, masked accuracy, perplexity)
- Embedding extraction utility for downstream repeat/non-repeat classification
- Non-neural (GC%/k-mer) baseline for fair comparison against gLM embeddings
- Interval-level (bp overlap) evaluation of predicted repeat regions against soft-masked ground truth
- Variant log-likelihood-ratio (LLR) scoring for SNVs using the trained MLM

## Data source in this workspace

The current genome FASTA is available under:

- `dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna`

## Setup

```bash
pip install -r requirements.txt
```

## Train a baseline model

```bash
python -m src.genome_lm.train \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --window-size 512 \
  --stride 512 \
  --max-windows 20000 \
  --batch-size 32 \
  --epochs 3 \
  --output-dir outputs/baseline_cnn
```

## Extract embeddings

```bash
python -m src.genome_lm.extract_embeddings \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --checkpoint outputs/baseline_cnn/best_model.pt \
  --window-size 512 \
  --stride 512 \
  --max-windows 5000 \
  --output-path outputs/baseline_cnn/embeddings.npz
```

## Optional repeat baseline

This baseline uses lowercase letters in soft-masked FASTA as weak repeat labels.

```bash
python -m src.genome_lm.repeat_baseline \
  --embeddings outputs/baseline_cnn/embeddings.npz
```

## Robustness sweep for repeat baseline

This sweep evaluates stability across repeat thresholds, random seeds, and sample sizes,
then writes run-level and summary CSV files.

```bash
python -m src.genome_lm.repeat_robustness \
  --embeddings outputs/baseline_cnn/embeddings.npz \
  --repeat-thresholds 0.1,0.2,0.3,0.4 \
  --seeds 13,21,34,55,89 \
  --sample-sizes 5000 \
  --output-csv outputs/baseline_cnn/robustness_runs.csv \
  --summary-csv outputs/baseline_cnn/robustness_summary.csv
```

## Non-neural baseline (GC% + k-mer) for comparison

This baseline uses only sequence-derived features (no learned embeddings) so you can
quantify how much the gLM embeddings add over simple composition-based features.

```bash
python -m src.genome_lm.nonneural_baseline \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --window-size 512 \
  --stride 512 \
  --max-windows 5000 \
  --kmer-size 3 \
  --repeat-threshold 0.2
```

## Interval-level repeat evaluation

This evaluates the embedding-based repeat classifier at base-pair resolution: predicted
repeat windows are merged into intervals and compared against the true soft-masked
(lowercase) repeat intervals from the reference FASTA, reporting precision, recall, IoU, and F1.

```bash
python -m src.genome_lm.repeat_interval_eval \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --embeddings outputs/baseline_cnn/embeddings.npz \
  --repeat-threshold 0.2 \
  --output-dir outputs/baseline_cnn/interval_eval
```

## Variant LLR scoring

Scores SNVs by masking the variant position and comparing the model's predicted
probability of the reference base versus the alternate base. Accepts a CSV of variants
(`seq_id,position,ref,alt`, 1-based position) or auto-samples a pilot set if omitted.

```bash
python -m src.genome_lm.variant_scoring \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --checkpoint outputs/baseline_cnn/best_model.pt \
  --auto-sample-n 200 \
  --output-csv outputs/baseline_cnn/variant_scoring/variant_llr.csv
```

## Motif discovery (position-wise masking)

This analysis masks each position in selected windows, records model probabilities
for A/C/G/T, and exports candidate motif contexts.

```bash
python -m src.genome_lm.motif_discovery \
  --fasta dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna \
  --checkpoint outputs/baseline_cnn/best_model.pt \
  --window-size 512 \
  --stride 512 \
  --max-windows 50 \
  --motif-width 9 \
  --top-k-motifs 200 \
  --output-dir outputs/baseline_cnn/motif_discovery
```

## Motif report (tables + plots)

Generate ranked site tables and motif plots from motif discovery outputs.

```bash
python -m src.genome_lm.motif_report \
  --input-dir outputs/baseline_cnn/motif_discovery \
  --output-dir outputs/baseline_cnn/motif_discovery/report \
  --top-n-sites 200 \
  --top-n-motifs 25
```

## Notes

- The code keeps chromosome/contig coordinates for each window.
- Ambiguous bases are preserved as `N` during preprocessing and excluded from MLM loss.
- This is a baseline scaffold for your phased experimental plan and can be extended with motif and variant scoring modules.
