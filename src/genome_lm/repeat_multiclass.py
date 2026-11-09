from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .chrom_map import load_chrom_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-class repeat family classification (LINE/SINE/LTR/etc.) using "
            "real RepeatMasker annotations instead of binary soft-mask labels."
        )
    )
    parser.add_argument("--repeatmasker-out", type=str, required=True)
    parser.add_argument("--sequence-report", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=30,
        help="Classes with fewer windows than this are grouped into 'Other'.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline_cnn_v3/repeat_multiclass",
    )
    return parser.parse_args()


def load_repeat_intervals(
    repeatmasker_path: str, ucsc_to_refseq: dict[str, str], target_seq_ids: set[str]
) -> dict[str, list[tuple[int, int, str]]]:
    """Stream the (large) RepeatMasker .out file, keeping only target sequence IDs."""
    target_ucsc_names = {
        ucsc for ucsc, refseq in ucsc_to_refseq.items() if refseq in target_seq_ids
    }

    intervals: dict[str, list[tuple[int, int, str]]] = {sid: [] for sid in target_seq_ids}
    lines_read = 0
    lines_kept = 0

    with Path(repeatmasker_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_read += 1
            fields = line.split()
            if len(fields) < 11:
                continue
            chrom = fields[4]
            if chrom not in target_ucsc_names:
                continue
            if not fields[0][0].isdigit():
                continue  # header/separator lines

            refseq = ucsc_to_refseq[chrom]
            start = int(fields[5]) - 1
            end = int(fields[6])
            repeat_class_family = fields[10]
            top_class = repeat_class_family.split("/")[0]
            intervals[refseq].append((start, end, top_class))
            lines_kept += 1

            if lines_read % 2_000_000 == 0:
                print(f"  scanned {lines_read:,} lines, kept {lines_kept:,} matching intervals")

    print(f"RepeatMasker scan complete: {lines_read:,} lines read, {lines_kept:,} kept")
    return intervals


def assign_window_labels(
    starts: np.ndarray, ends: np.ndarray, seq_ids: np.ndarray, intervals: dict[str, list[tuple[int, int, str]]]
) -> list[str]:
    labels: list[str] = []
    by_seq: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid, ivs in intervals.items():
        if not ivs:
            by_seq[sid] = (np.array([]), np.array([]), np.array([]))
            continue
        arr_start = np.array([iv[0] for iv in ivs])
        arr_end = np.array([iv[1] for iv in ivs])
        arr_class = np.array([iv[2] for iv in ivs])
        by_seq[sid] = (arr_start, arr_end, arr_class)

    for sid, start, end in zip(seq_ids.tolist(), starts.tolist(), ends.tolist()):
        arr_start, arr_end, arr_class = by_seq.get(sid, (np.array([]), np.array([]), np.array([])))
        if len(arr_start) == 0:
            labels.append("Non_repeat")
            continue

        mask = (arr_start < end) & (arr_end > start)
        if not mask.any():
            labels.append("Non_repeat")
            continue

        overlap_bp = np.minimum(arr_end[mask], end) - np.maximum(arr_start[mask], start)
        best_idx = np.argmax(overlap_bp)
        labels.append(str(arr_class[mask][best_idx]))

    return labels


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("Starting multi-class repeat classification (real RepeatMasker labels)")
    print(f"RepeatMasker file: {args.repeatmasker_out}")
    print(f"Embeddings: {args.embeddings}")
    print("=" * 72)

    data = np.load(args.embeddings, allow_pickle=True)
    x = data["embeddings"]
    seq_ids = data["seq_id"]
    starts = data["start"]
    ends = data["end"]
    target_seq_ids = set(seq_ids.tolist())
    print(f"Loaded embeddings shape: {x.shape}, target seq_ids: {target_seq_ids}")

    ucsc_to_refseq, _ = load_chrom_maps(args.sequence_report)
    intervals = load_repeat_intervals(args.repeatmasker_out, ucsc_to_refseq, target_seq_ids)

    print("Assigning per-window repeat-family labels...")
    labels = assign_window_labels(starts, ends, seq_ids, intervals)

    label_counts = Counter(labels)
    print("Raw label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    keep_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= args.min_class_count}
    y = np.array([lbl if lbl in keep_labels else "Other" for lbl in labels])
    final_counts = Counter(y.tolist())
    print(f"\nFinal label distribution (min_class_count={args.min_class_count}):")
    for label, count in final_counts.most_common():
        print(f"  {label}: {count}")

    if len(final_counts) < 2:
        raise RuntimeError("Only one class present after grouping; cannot train a classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"\nSplit sizes: train={len(y_train)} test={len(y_test)}")

    clf = LogisticRegression(max_iter=2000, random_state=args.seed)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    labels_sorted = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_dir / "classification_report.csv")
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    pd.Series(final_counts).to_csv(output_dir / "label_distribution.csv", header=["count"])

    print("\nMulti-class repeat classification complete.")
    print(report_df.to_string())
    print(f"\nSaved: {output_dir / 'classification_report.csv'}")
    print(f"Saved: {output_dir / 'confusion_matrix.csv'}")
    print(f"Saved: {output_dir / 'label_distribution.csv'}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
