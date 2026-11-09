from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model LLR scores between ClinVar clinical-significance classes "
            "(e.g. Pathogenic vs Benign) to test whether the model's constraint signal "
            "aligns with known variant effect labels."
        )
    )
    parser.add_argument(
        "--curated-variants",
        type=str,
        required=True,
        help="CSV from prepare_clinvar_variants.py (has seq_id,position,ref,alt,clnsig).",
    )
    parser.add_argument(
        "--variant-llr",
        type=str,
        required=True,
        help="CSV from variant_scoring.py (has seq_id,position,ref,alt,llr_bits,...).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline_cnn_v3/variant_scoring/clinvar_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("Starting ClinVar LLR vs clinical-significance analysis")
    print("=" * 72)

    curated_df = pd.read_csv(args.curated_variants)
    llr_df = pd.read_csv(args.variant_llr)

    merged = llr_df.merge(
        curated_df[["seq_id", "position", "ref", "alt", "clnsig"]],
        on=["seq_id", "position", "ref", "alt"],
        how="inner",
    )
    print(f"Merged {len(merged)}/{len(llr_df)} scored variants with ClinVar labels")

    if merged.empty:
        raise RuntimeError("No matching rows after merge. Check inputs come from the same variant set.")

    summary = (
        merged.groupby("clnsig")["llr_bits"]
        .agg(["count", "mean", "std", "median"])
        .sort_values("mean", ascending=False)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "llr_by_clinical_significance.csv")
    merged.to_csv(output_dir / "merged_variant_llr_clinvar.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [g["llr_bits"].values for _, g in merged.groupby("clnsig")]
    labels = list(merged.groupby("clnsig").groups.keys())
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("LLR (bits)")
    ax.set_title("Model LLR by ClinVar clinical significance")
    fig.tight_layout()
    plot_path = output_dir / "llr_by_clinical_significance.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("\nLLR summary by clinical significance:")
    print(summary.to_string())
    print(f"\nSaved: {output_dir / 'llr_by_clinical_significance.csv'}")
    print(f"Saved: {output_dir / 'merged_variant_llr_clinvar.csv'}")
    print(f"Saved: {plot_path}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
