from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[ch] for ch in reversed(seq.upper()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate motif candidates against real JASPAR position frequency matrices "
            "(PFMs), converted to log-odds PWMs scored against a uniform background."
        )
    )
    parser.add_argument("--motif-candidates", type=str, required=True)
    parser.add_argument("--jaspar-pfm", type=str, required=True)
    parser.add_argument(
        "--relative-score-threshold",
        type=float,
        default=0.80,
        help="Fraction of a motif's max possible log-odds score required to call a match.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/baseline_cnn/motif_discovery/report/known_motif_matches.csv",
    )
    return parser.parse_args()


def parse_jaspar_pfms(path: str) -> list[dict]:
    """Parse a JASPAR bulk PFM text file into a list of {id, name, pwm, max_score}."""
    motifs: list[dict] = []
    header_re = re.compile(r"^>(\S+)\s+(.*)$")
    row_re = re.compile(r"^([ACGT])\s*\[\s*(.+?)\s*\]\s*$")

    current_id: str | None = None
    current_name: str | None = None
    counts: dict[str, list[int]] = {}

    def flush() -> None:
        if current_id is None or len(counts) != 4:
            return
        width = len(counts["A"])
        count_mat = np.array([counts[b] for b in BASES], dtype=np.float64)
        col_totals = count_mat.sum(axis=0)
        # log-odds PWM against uniform 0.25 background, with a small pseudocount.
        freqs = (count_mat + 0.25) / (col_totals + 1.0)
        pwm = np.log2(freqs / 0.25)
        max_score = pwm.max(axis=0).sum()
        motifs.append(
            {
                "id": current_id,
                "name": current_name,
                "width": width,
                "pwm": pwm,
                "max_score": float(max_score),
            }
        )

    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            header_match = header_re.match(line)
            if header_match:
                flush()
                current_id = header_match.group(1)
                current_name = header_match.group(2).strip()
                counts = {}
                continue

            row_match = row_re.match(line)
            if row_match:
                base = row_match.group(1)
                values = [int(v) for v in row_match.group(2).split()]
                counts[base] = values
        flush()

    return motifs


def score_sequence(pwm: np.ndarray, seq: str) -> float:
    """Score a sequence of length == pwm.shape[1] against the PWM."""
    score = 0.0
    for i, ch in enumerate(seq):
        idx = BASES.index(ch) if ch in BASES else None
        if idx is None:
            return float("-inf")
        score += pwm[idx, i]
    return score


def best_window_score(pwm: np.ndarray, candidate: str) -> tuple[float, int]:
    """Slide a PWM across a (possibly longer) candidate sequence, return best score and offset."""
    width = pwm.shape[1]
    if width > len(candidate):
        return float("-inf"), -1

    best_score = float("-inf")
    best_offset = -1
    for offset in range(len(candidate) - width + 1):
        window = candidate[offset : offset + width]
        score = score_sequence(pwm, window)
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_score, best_offset


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("Starting real JASPAR motif validation (PWM log-odds scoring)")
    print(f"Motif candidates: {args.motif_candidates}")
    print(f"JASPAR PFM file: {args.jaspar_pfm}")
    print(f"Relative score threshold: {args.relative_score_threshold}")
    print("=" * 72)

    motif_df = pd.read_csv(args.motif_candidates)
    jaspar_motifs = parse_jaspar_pfms(args.jaspar_pfm)
    print(f"Loaded {len(jaspar_motifs)} JASPAR motifs")
    print(f"Screening {len(motif_df)} candidate motifs")

    fit_motifs = [m for m in jaspar_motifs if m["width"] <= 9]
    skipped = len(jaspar_motifs) - len(fit_motifs)
    print(f"{len(fit_motifs)} JASPAR motifs fit within 9bp candidates ({skipped} wider, skipped)")

    rows = []
    for idx, row in enumerate(motif_df.itertuples(index=False), start=1):
        candidate = str(row.motif).upper()
        candidate_rc = reverse_complement(candidate)

        for jm in fit_motifs:
            for strand, seq in (("+", candidate), ("-", candidate_rc)):
                score, offset = best_window_score(jm["pwm"], seq)
                if score == float("-inf") or jm["max_score"] <= 0:
                    continue
                relative_score = score / jm["max_score"]
                if relative_score >= args.relative_score_threshold:
                    rows.append(
                        {
                            "candidate_motif": candidate,
                            "jaspar_id": jm["id"],
                            "jaspar_name": jm["name"],
                            "strand": strand,
                            "offset": offset,
                            "raw_score": score,
                            "relative_score": relative_score,
                            "count": row.count,
                            "true_base_prob_mean": row.true_base_prob_mean,
                            "surprise_nats_mean": row.surprise_nats_mean,
                        }
                    )

        if idx % 50 == 0 or idx == len(motif_df):
            print(f"  screened {idx}/{len(motif_df)} candidates, matches so far: {len(rows)}")

    matches_df = pd.DataFrame(rows)
    if not matches_df.empty:
        matches_df = matches_df.sort_values("relative_score", ascending=False)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_path, index=False)

    print(f"\nFound {len(matches_df)} candidate-to-JASPAR-motif matches "
          f"(>= {args.relative_score_threshold:.0%} of max score).")
    if not matches_df.empty:
        print("\nTop matched JASPAR TFs by frequency:")
        print(matches_df["jaspar_name"].value_counts().head(15).to_string())
    print(f"Saved: {output_path}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

