from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DNAWindow, build_windows
from .model import GenomicCNNMLM
from .tokenizer import DNATokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Motif discovery via position-wise masking. For each base position in each window, "
            "mask that position and record model probabilities P(A), P(C), P(G), P(T)."
        )
    )
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)
    parser.add_argument(
        "--motif-width",
        type=int,
        default=9,
        help="Odd width of sequence context used for candidate motif extraction.",
    )
    parser.add_argument(
        "--top-k-motifs",
        type=int,
        default=200,
        help="Number of candidate motifs to keep in motif_candidates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline_cnn/motif_discovery",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, tokenizer: DNATokenizer) -> tuple[GenomicCNNMLM, dict[str, int | float | str]]:
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
    return model, train_args


def entropy_base4(prob_row: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(prob_row, eps, 1.0)
    return float(-(p * np.log2(p)).sum())


def center_kmer(sequence: str, center_pos: int, width: int) -> str | None:
    half = width // 2
    left = center_pos - half
    right = center_pos + half + 1
    if left < 0 or right > len(sequence):
        return None
    kmer = sequence[left:right]
    if any(ch not in {"A", "C", "G", "T"} for ch in kmer):
        return None
    return kmer


def run_window_scan(
    window: DNAWindow,
    model: GenomicCNNMLM,
    tokenizer: DNATokenizer,
    device: torch.device,
    motif_width: int,
) -> tuple[list[dict[str, int | float | str]], list[tuple[str, float, float]]]:
    token_ids = np.array(tokenizer.encode(window.sequence), dtype=np.int64)
    valid_positions = np.where(np.isin(token_ids, [0, 1, 2, 3]))[0]

    if len(valid_positions) == 0:
        return [], []

    input_batch = np.repeat(token_ids[None, :], repeats=len(valid_positions), axis=0)
    input_batch[np.arange(len(valid_positions)), valid_positions] = tokenizer.mask_id

    with torch.no_grad():
        input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    rows: list[dict[str, int | float | str]] = []
    motif_observations: list[tuple[str, float, float]] = []

    for row_idx, pos in enumerate(valid_positions):
        base_probs = probs[row_idx, pos, :4]
        true_base_id = int(token_ids[pos])
        true_base = "ACGT"[true_base_id]
        true_prob = float(base_probs[true_base_id])

        sorted_probs = np.sort(base_probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        surprise = float(-math.log(max(true_prob, 1e-12)))

        rows.append(
            {
                "seq_id": window.seq_id,
                "start": window.start,
                "end": window.end,
                "position_in_window": int(pos),
                "genomic_pos": int(window.start + pos),
                "true_base": true_base,
                "p_A": float(base_probs[0]),
                "p_C": float(base_probs[1]),
                "p_G": float(base_probs[2]),
                "p_T": float(base_probs[3]),
                "true_base_prob": true_prob,
                "surprise_nats": surprise,
                "entropy_bits": entropy_base4(base_probs),
                "top1_minus_top2": margin,
            }
        )

        kmer = center_kmer(window.sequence, int(pos), motif_width)
        if kmer is not None:
            motif_observations.append((kmer, true_prob, surprise))

    return rows, motif_observations


def main() -> None:
    args = parse_args()
    run_start = time.time()

    if args.motif_width % 2 == 0 or args.motif_width < 3:
        raise ValueError("motif-width must be an odd integer >= 3")

    print("=" * 72)
    print("Starting motif discovery")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FASTA: {args.fasta}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"motif_width={args.motif_width}, top_k_motifs={args.top_k_motifs}"
    )
    print("=" * 72)

    tokenizer = DNATokenizer()
    model, train_args = load_model(args.checkpoint, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    print(
        "Loaded model config: "
        f"embedding_dim={train_args['embedding_dim']}, channels={train_args['channels']}, "
        f"num_layers={train_args['num_layers']}"
    )

    windows = build_windows(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )

    if not windows:
        raise RuntimeError("No windows generated. Relax filters or verify FASTA path.")

    print(f"Windows to scan: {len(windows)}")

    all_rows: list[dict[str, int | float | str]] = []
    motif_stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "true_prob_sum": 0.0, "surprise_sum": 0.0}
    )

    for idx, window in enumerate(windows, start=1):
        rows, motif_observations = run_window_scan(
            window=window,
            model=model,
            tokenizer=tokenizer,
            device=device,
            motif_width=args.motif_width,
        )
        all_rows.extend(rows)

        for kmer, true_prob, surprise in motif_observations:
            motif_stats[kmer]["count"] = int(motif_stats[kmer]["count"]) + 1
            motif_stats[kmer]["true_prob_sum"] = float(motif_stats[kmer]["true_prob_sum"]) + true_prob
            motif_stats[kmer]["surprise_sum"] = float(motif_stats[kmer]["surprise_sum"]) + surprise

        if idx % 5 == 0 or idx == len(windows):
            print(f"  processed windows {idx:>4}/{len(windows)} | rows={len(all_rows)}")

    if not all_rows:
        raise RuntimeError("No valid A/C/G/T positions found for motif discovery.")

    predictions_df = pd.DataFrame(all_rows)

    position_summary_df = (
        predictions_df.groupby("position_in_window", as_index=False)
        .agg(
            n_sites=("true_base", "count"),
            p_A_mean=("p_A", "mean"),
            p_C_mean=("p_C", "mean"),
            p_G_mean=("p_G", "mean"),
            p_T_mean=("p_T", "mean"),
            true_base_prob_mean=("true_base_prob", "mean"),
            surprise_nats_mean=("surprise_nats", "mean"),
            entropy_bits_mean=("entropy_bits", "mean"),
            top1_minus_top2_mean=("top1_minus_top2", "mean"),
        )
        .sort_values("position_in_window")
    )

    motif_rows = []
    for kmer, stats in motif_stats.items():
        count = int(stats["count"])
        if count <= 0:
            continue
        motif_rows.append(
            {
                "motif": kmer,
                "count": count,
                "true_base_prob_mean": float(stats["true_prob_sum"]) / count,
                "surprise_nats_mean": float(stats["surprise_sum"]) / count,
            }
        )

    motif_df = pd.DataFrame(motif_rows)
    if not motif_df.empty:
        motif_df = motif_df.sort_values(
            by=["surprise_nats_mean", "count"],
            ascending=[False, False],
        ).head(args.top_k_motifs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "position_predictions.csv"
    position_summary_path = output_dir / "position_summary.csv"
    motif_candidates_path = output_dir / "motif_candidates.csv"
    metadata_path = output_dir / "run_metadata.json"

    predictions_df.to_csv(predictions_path, index=False)
    position_summary_df.to_csv(position_summary_path, index=False)
    motif_df.to_csv(motif_candidates_path, index=False)

    metadata = {
        "fasta": args.fasta,
        "checkpoint": args.checkpoint,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "motif_width": args.motif_width,
        "top_k_motifs": args.top_k_motifs,
        "n_windows_scanned": len(windows),
        "n_position_rows": int(len(predictions_df)),
        "n_unique_candidate_motifs": int(len(motif_rows)),
        "runtime_seconds": time.time() - run_start,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nMotif discovery complete.")
    print(f"Saved per-position predictions: {predictions_path}")
    print(f"Saved per-position summary:     {position_summary_path}")
    print(f"Saved motif candidates:         {motif_candidates_path}")
    print(f"Saved run metadata:             {metadata_path}")
    print(f"Total runtime: {metadata['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
