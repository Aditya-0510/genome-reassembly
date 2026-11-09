# Project Flow Walkthrough

This is a narrated script explaining the project end-to-end, stage by stage — useful
for explaining the project to someone unfamiliar with the codebase.

## Narrated script — explaining the flow stage by stage

### Opening
"This project is a self-supervised DNA language model, trained directly on raw human
genome sequence, with no labels. The goal is to learn a general-purpose representation
of DNA, then prove that representation is actually useful for real genomic tasks."

### Stage 1: Data in, windows out
"We start with one raw genome FASTA file. The first step cuts it into fixed-length
windows — 512 base pairs each — and filters out low-quality stretches with too many
ambiguous bases. Every window keeps its genomic coordinates, so any prediction we make
can be traced back to an exact position in the genome."

### Stage 2: Tokenize and mask
"Each window is tokenized into A/C/G/T plus special MASK/PAD/N symbols. During
training, about 15% of positions in every window are randomly masked — the model
never sees the true base at those positions and has to predict it from context alone,
the same self-supervised idea behind BERT, just applied to DNA instead of words."

### Stage 3: Train the model
"The model itself is a dilated 1D convolutional network. Each layer doubles its
dilation — 1, 2, 4, 8, up to 256 — so by the final layer, a single output position has
effectively 'seen' over a thousand base pairs of context, even though each individual
convolution only looks at 3 positions at a time. This detail mattered a lot: an earlier
version of this schedule capped out at dilation 8 and reset every 4 layers, which
silently limited the model to a ~37bp effective context no matter how deep we made it.
Fixing that one line improved every downstream result."

### Stage 4: One checkpoint, many consumers
"Once trained, the checkpoint becomes the single source of truth for everything
downstream. Nothing past this point retrains the model — they all just load the
checkpoint and either read its predictions or read its internal representations."

### Stage 5: Embeddings and repeat detection
"We extract a pooled embedding for every window, then train a simple classifier on
top of those embeddings to tell repetitive DNA from non-repetitive DNA. To make sure
that result isn't a fluke of one lucky train/test split, we repeat the experiment
across 5 random seeds and 4 label thresholds and report the mean and standard
deviation. We also built a control baseline that uses only GC-content and k-mer
frequency — no learned embeddings at all — specifically so we can show the learned
representations are adding real signal, not just reflecting simple sequence
composition."

### Stage 6: Stricter, base-pair-level evaluation
"Window-level accuracy can be misleading, so we built a stricter check: merge the
model's predicted repeat windows into genomic intervals, and compare them
base-pair-by-base-pair against the genome's own soft-masked repeat regions. This
exposes a real, honestly-reported gap — the model finds most true repeat regions
(high recall) but doesn't always draw tight boundaries around them (moderate
precision)."

### Stage 7: Motif discovery
"Separately, we scan windows position by position, masking each base one at a time
and recording the model's predicted probability for A, C, G, and T at that spot. This
produces per-position confidence signals — entropy, surprise, margin — which get
turned into ranked tables and plots showing where the model is most and least certain,
and which short sequence contexts recur often enough to be candidate motifs."

### Stage 8: Variant scoring
"Finally, for any single-nucleotide variant, we mask that exact position and compare
the model's predicted probability for the reference base against the alternate base.
The log-ratio between those two probabilities tells you how strongly the model
'expects' the reference versus the mutation — a first step toward model-based variant
impact scoring."

### Closing
"Every one of these downstream stages is a different way of asking the same question:
did the self-supervised pretraining actually learn something about real genomic
structure? The answer so far is yes — and fixing the model's effective context length
made that answer measurably stronger across every single one of these evaluations."
