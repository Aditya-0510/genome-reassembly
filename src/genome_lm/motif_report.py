from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate motif discovery summaries and plots from motif_discovery outputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/baseline_cnn/motif_discovery",
        help="Directory containing position_predictions.csv and motif_candidates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline_cnn/motif_discovery/report",
        help="Directory where summary tables and plots are written.",
    )
    parser.add_argument("--top-n-sites", type=int, default=200)
    parser.add_argument("--top-n-motifs", type=int, default=25)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def save_top_tables(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    top_n_sites: int,
) -> tuple[Path, Path]:
    top_surprising = predictions_df.sort_values(
        by=["surprise_nats", "top1_minus_top2"],
        ascending=[False, False],
    ).head(top_n_sites)

    top_confident = predictions_df.sort_values(
        by=["entropy_bits", "top1_minus_top2", "true_base_prob"],
        ascending=[True, False, False],
    ).head(top_n_sites)

    surprising_path = output_dir / "top_surprising_sites.csv"
    confident_path = output_dir / "top_confident_sites.csv"

    top_surprising.to_csv(surprising_path, index=False)
    top_confident.to_csv(confident_path, index=False)
    return surprising_path, confident_path


def plot_base_probability_heatmap(position_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    base_cols = ["p_A_mean", "p_C_mean", "p_G_mean", "p_T_mean"]
    heatmap_data = position_summary_df[base_cols].to_numpy().T

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(heatmap_data, aspect="auto", interpolation="nearest", cmap="viridis")

    ax.set_yticks(range(4))
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("Position in window")
    ax.set_title("Mean base probabilities by masked position")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    out_path = output_dir / "position_probability_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_entropy_margin(position_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    x = position_summary_df["position_in_window"].to_numpy()
    entropy = position_summary_df["entropy_bits_mean"].to_numpy()
    margin = position_summary_df["top1_minus_top2_mean"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(14, 4.2))
    ax1.plot(x, entropy, color="#1b9e77", linewidth=1.8, label="Entropy (bits)")
    ax1.set_xlabel("Position in window")
    ax1.set_ylabel("Entropy (bits)", color="#1b9e77")
    ax1.tick_params(axis="y", labelcolor="#1b9e77")

    ax2 = ax1.twinx()
    ax2.plot(x, margin, color="#d95f02", linewidth=1.4, label="Top1-Top2 margin")
    ax2.set_ylabel("Margin", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    fig.suptitle("Uncertainty and confidence by masked position")
    out_path = output_dir / "position_entropy_margin.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_top_motif_counts(motif_df: pd.DataFrame, output_dir: Path, top_n_motifs: int) -> Path:
    plot_df = motif_df.sort_values(by=["count", "surprise_nats_mean"], ascending=[False, False]).head(
        top_n_motifs
    )
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(plot_df))))
    ax.barh(plot_df["motif"], plot_df["count"], color="#4C78A8")
    ax.set_xlabel("Count")
    ax.set_ylabel("Motif")
    ax.set_title(f"Top {len(plot_df)} motif contexts by frequency")

    out_path = output_dir / "top_motif_counts.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_summary_json(
    predictions_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    output_dir: Path,
    runtime_seconds: float,
) -> Path:
    summary = {
        "n_prediction_rows": int(len(predictions_df)),
        "n_unique_seq_ids": int(predictions_df["seq_id"].nunique()),
        "n_unique_motifs": int(len(motif_df)),
        "mean_true_base_prob": float(predictions_df["true_base_prob"].mean()),
        "mean_surprise_nats": float(predictions_df["surprise_nats"].mean()),
        "mean_entropy_bits": float(predictions_df["entropy_bits"].mean()),
        "mean_top1_minus_top2": float(predictions_df["top1_minus_top2"].mean()),
        "runtime_seconds": runtime_seconds,
    }

    out_path = output_dir / "report_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def main() -> None:
    args = parse_args()
    start_time = time.time()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = input_dir / "position_predictions.csv"
    position_summary_path = input_dir / "position_summary.csv"
    motif_candidates_path = input_dir / "motif_candidates.csv"

    require_file(predictions_path)
    require_file(position_summary_path)
    require_file(motif_candidates_path)

    print("=" * 72)
    print("Building motif report")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 72)

    predictions_df = pd.read_csv(predictions_path)
    position_summary_df = pd.read_csv(position_summary_path)
    motif_df = pd.read_csv(motif_candidates_path)

    print(
        f"Loaded rows: predictions={len(predictions_df)}, "
        f"position_summary={len(position_summary_df)}, motifs={len(motif_df)}"
    )

    surprising_csv, confident_csv = save_top_tables(
        predictions_df=predictions_df,
        output_dir=output_dir,
        top_n_sites=args.top_n_sites,
    )

    heatmap_png = plot_base_probability_heatmap(position_summary_df, output_dir)
    entropy_margin_png = plot_entropy_margin(position_summary_df, output_dir)
    motif_counts_png = plot_top_motif_counts(motif_df, output_dir, args.top_n_motifs)

    runtime_seconds = time.time() - start_time
    summary_json = write_summary_json(
        predictions_df=predictions_df,
        motif_df=motif_df,
        output_dir=output_dir,
        runtime_seconds=runtime_seconds,
    )

    print("\nMotif report complete.")
    print(f"Saved: {surprising_csv}")
    print(f"Saved: {confident_csv}")
    print(f"Saved: {heatmap_png}")
    print(f"Saved: {entropy_margin_png}")
    print(f"Saved: {motif_counts_png}")
    print(f"Saved: {summary_json}")
    print(f"Total runtime: {runtime_seconds:.1f}s")


if __name__ == "__main__":
    main()
