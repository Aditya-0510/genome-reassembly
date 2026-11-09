from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .chrom_map import load_chrom_maps

BASES = {"A", "C", "G", "T"}
DEFAULT_CLASSES = ["Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a balanced, curated SNV set from a ClinVar VCF for a target "
            "chromosome, labeled with clinical significance, for use with variant_scoring.py."
        )
    )
    parser.add_argument("--clinvar-vcf", type=str, required=True)
    parser.add_argument("--sequence-report", type=str, required=True)
    parser.add_argument("--target-seq-id", type=str, default="NC_000001.11")
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated exact CLNSIG values to keep.",
    )
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="dataset/clinvar.vcf/clinvar_chr1_curated.csv",
    )
    return parser.parse_args()


CLNSIG_RE = re.compile(r"CLNSIG=([^;]+)")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    wanted_classes = set(args.classes.split(","))

    print("=" * 72)
    print("Starting curated ClinVar SNV extraction")
    print(f"ClinVar VCF: {args.clinvar_vcf}")
    print(f"Target seq_id: {args.target_seq_id}")
    print(f"Classes kept: {sorted(wanted_classes)}")
    print("=" * 72)

    _, chrname_to_refseq = load_chrom_maps(args.sequence_report)
    target_chrnames = {
        name for name, refseq in chrname_to_refseq.items() if refseq == args.target_seq_id
    }
    if not target_chrnames:
        raise RuntimeError(f"No chrName found mapping to {args.target_seq_id} in sequence report.")
    print(f"Matching CHROM field(s) in VCF: {sorted(target_chrnames)}")

    by_class: dict[str, list[dict]] = {cls: [] for cls in wanted_classes}
    lines_read = 0

    with Path(args.clinvar_vcf).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            lines_read += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, _vid, ref, alt, _qual, _filt, info = fields[:8]

            if chrom not in target_chrnames:
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
                continue

            match = CLNSIG_RE.search(info)
            if not match:
                continue
            clnsig = match.group(1)
            if clnsig not in wanted_classes:
                continue

            by_class[clnsig].append(
                {
                    "seq_id": args.target_seq_id,
                    "position": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "clnsig": clnsig,
                }
            )

            if lines_read % 500_000 == 0:
                print(f"  scanned {lines_read:,} VCF lines...")

    print("\nMatched SNV counts per class (before sampling):")
    for cls, rows in by_class.items():
        print(f"  {cls}: {len(rows)}")

    sampled_rows: list[dict] = []
    for cls, rows in by_class.items():
        if not rows:
            continue
        n = min(args.max_per_class, len(rows))
        chosen_idx = rng.choice(len(rows), size=n, replace=False)
        sampled_rows.extend(rows[i] for i in chosen_idx)

    if not sampled_rows:
        raise RuntimeError("No matching variants found. Check --target-seq-id and --classes.")

    out_df = pd.DataFrame(sampled_rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSampled {len(out_df)} total variants (max {args.max_per_class} per class).")
    print(out_df["clnsig"].value_counts().to_string())
    print(f"Saved: {output_path}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
