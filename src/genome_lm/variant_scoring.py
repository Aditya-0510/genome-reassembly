from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from .model import GenomicCNNMLM
from .tokenizer import DNATokenizer

BASES = ["A", "C", "G", "T"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score SNVs with a log-likelihood ratio (LLR) using a trained DNA MLM. "
            "For each variant, mask the variant position and compare P(ref) vs P(alt)."
        )
    )
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "CSV with columns seq_id,position,ref,alt (position is 1-based, VCF-style). "
            "If omitted, a pilot set is auto-sampled from the FASTA."
        ),
    )
    parser.add_argument(
        "--auto-sample-n",
        type=int,
        default=200,
        help="Number of pilot SNVs to auto-sample when --variants is not provided.",
    )
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/baseline_cnn/variant_scoring/variant_llr.csv",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, tokenizer: DNATokenizer) -> GenomicCNNMLM:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    train_args = ckpt["args"]
    model = GenomicCNNMLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=train_args["embedding_dim"],
        channels=train_args["channels"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_sequences(fasta_path: str, seq_ids: set[str] | None) -> dict[str, str]:
    """Load FASTA records (uppercased) matching seq_ids, or all records if seq_ids is None."""
    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if seq_ids is not None and record.id not in seq_ids:
            continue
        sequences[record.id] = str(record.seq).upper()
        if seq_ids is not None and len(sequences) == len(seq_ids):
            break
    return sequences


def auto_sample_variants(
    sequences: dict[str, str], n: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seq_id = next(iter(sequences))
    sequence = sequences[seq_id]

    valid_positions = [i for i, ch in enumerate(sequence) if ch in BASES]
    if len(valid_positions) < n:
        n = len(valid_positions)

    chosen = rng.choice(valid_positions, size=n, replace=False)
    rows = []
    for pos0 in chosen:
        ref = sequence[pos0]
        alt_choices = [b for b in BASES if b != ref]
        alt = rng.choice(alt_choices)
        rows.append({"seq_id": seq_id, "position": int(pos0) + 1, "ref": ref, "alt": alt})

    return pd.DataFrame(rows)


def score_variants(
    variants_df: pd.DataFrame,
    sequences: dict[str, str],
    model: GenomicCNNMLM,
    tokenizer: DNATokenizer,
    context_size: int,
) -> pd.DataFrame:
    half = context_size // 2
    rows = []
    skipped = 0

    for row in variants_df.itertuples(index=False):
        seq_id = row.seq_id
        position_1based = int(row.position)
        ref = str(row.ref).upper()
        alt = str(row.alt).upper()
        pos0 = position_1based - 1

        sequence = sequences.get(seq_id)
        if sequence is None:
            skipped += 1
            continue

        window_start = pos0 - half
        window_end = pos0 + (context_size - half)
        if window_start < 0 or window_end > len(sequence):
            skipped += 1
            continue

        window_seq = sequence[window_start:window_end]
        local_pos = pos0 - window_start

        if window_seq[local_pos] != ref:
            skipped += 1
            continue
        if ref not in BASES or alt not in BASES:
            skipped += 1
            continue

        token_ids = np.array(tokenizer.encode(window_seq), dtype=np.int64)
        token_ids[local_pos] = tokenizer.mask_id

        with torch.no_grad():
            input_tensor = torch.tensor(token_ids[None, :], dtype=torch.long)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)[0, local_pos, :4].numpy()

        ref_id = BASES.index(ref)
        alt_id = BASES.index(alt)
        p_ref = float(probs[ref_id])
        p_alt = float(probs[alt_id])

        eps = 1e-12
        llr_nats = math.log(max(p_ref, eps)) - math.log(max(p_alt, eps))
        llr_bits = llr_nats / math.log(2)

        rows.append(
            {
                "seq_id": seq_id,
                "position": position_1based,
                "ref": ref,
                "alt": alt,
                "window_start": window_start,
                "window_end": window_end,
                "p_ref": p_ref,
                "p_alt": p_alt,
                "llr_nats": llr_nats,
                "llr_bits": llr_bits,
            }
        )

    if skipped:
        print(f"Skipped {skipped} variants (out of bounds, ref mismatch, or missing seq_id).")

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("Starting variant LLR scoring")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FASTA: {args.fasta}")
    print(f"Context size: {args.context_size}")
    print("=" * 72)

    tokenizer = DNATokenizer()
    model = load_model(args.checkpoint, tokenizer)

    if args.variants:
        variants_df = pd.read_csv(args.variants)
        required_cols = {"seq_id", "position", "ref", "alt"}
        if not required_cols.issubset(variants_df.columns):
            raise ValueError(f"--variants CSV must contain columns: {sorted(required_cols)}")
        seq_ids = set(variants_df["seq_id"].astype(str).unique())
        print(f"Loaded {len(variants_df)} variants from {args.variants}")
        sequences = load_sequences(args.fasta, seq_ids)
    else:
        print(
            f"No --variants provided. Auto-sampling {args.auto_sample_n} pilot SNVs "
            "from the first FASTA record."
        )
        sequences = load_sequences(args.fasta, seq_ids=None)
        # Only need the first record for pilot sampling; drop the rest to save memory.
        first_id = next(iter(sequences))
        sequences = {first_id: sequences[first_id]}
        variants_df = auto_sample_variants(sequences, args.auto_sample_n, args.seed)
        print(f"Auto-sampled {len(variants_df)} pilot variants from seq_id={first_id}")

    print("Scoring variants...")
    results_df = score_variants(
        variants_df=variants_df,
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        context_size=args.context_size,
    )

    if results_df.empty:
        raise RuntimeError("No variants could be scored. Check positions/ref bases/context size.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nScored {len(results_df)}/{len(variants_df)} variants.")
    print(f"Mean LLR (nats): {results_df['llr_nats'].mean():.4f}")
    print(f"Mean |LLR| (nats): {results_df['llr_nats'].abs().mean():.4f}")
    print(f"Saved: {output_path}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
