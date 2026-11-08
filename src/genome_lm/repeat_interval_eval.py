from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interval-level evaluation of the embedding-based repeat classifier against "
            "soft-masked (lowercase) repeat intervals from the reference FASTA."
        )
    )
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument(
        "--repeat-threshold",
        type=float,
        default=0.2,
        help="Window is labeled repeat (for classifier training) if repeat_fraction >= threshold.",
    )
    parser.add_argument(
        "--predicted-probability-threshold",
        type=float,
        default=0.5,
        help="Classifier probability cutoff for calling a window predicted-repeat.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline_cnn/interval_eval",
    )
    return parser.parse_args()


def find_lowercase_intervals(sequence: str) -> list[tuple[int, int]]:
    """Return merged [start, end) 0-based intervals of contiguous lowercase bases."""
    intervals: list[tuple[int, int]] = []
    start = None
    for i, ch in enumerate(sequence):
        is_lower = ch.islower()
        if is_lower and start is None:
            start = i
        elif not is_lower and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(sequence)))
    return intervals


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i, j = 0, 0
    a = sorted(a)
    b = sorted(b)
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def total_bp(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def load_true_repeat_intervals(
    fasta_path: str, seq_ids: set[str], bounds: dict[str, tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    """Scan FASTA records matching seq_ids, restricted to [min_start, max_end) per seq_id."""
    true_intervals: dict[str, list[tuple[int, int]]] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if record.id not in seq_ids:
            continue
        lo, hi = bounds[record.id]
        raw = str(record.seq)[lo:hi]
        local_intervals = find_lowercase_intervals(raw)
        true_intervals[record.id] = [(s + lo, e + lo) for s, e in local_intervals]
    return true_intervals


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("Starting interval-level repeat evaluation")
    print(f"FASTA: {args.fasta}")
    print(f"Embeddings: {args.embeddings}")
    print(
        "Config: "
        f"repeat_threshold={args.repeat_threshold}, "
        f"predicted_probability_threshold={args.predicted_probability_threshold}"
    )
    print("=" * 72)

    data = np.load(args.embeddings, allow_pickle=True)
    x = data["embeddings"]
    repeat_fraction = data["repeat_fraction"]
    seq_id = data["seq_id"]
    starts = data["start"]
    ends = data["end"]

    print(f"Loaded embeddings shape: {x.shape}")

    y = (repeat_fraction >= args.repeat_threshold).astype(int)
    indices = np.arange(len(y))

    idx_train, idx_test = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Split sizes: train={len(idx_train)} test={len(idx_test)}")

    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(x[idx_train], y[idx_train])

    test_prob = clf.predict_proba(x[idx_test])[:, 1]
    test_pred = (test_prob >= args.predicted_probability_threshold).astype(int)

    test_seq_id = seq_id[idx_test]
    test_starts = starts[idx_test]
    test_ends = ends[idx_test]

    unique_seq_ids = sorted(set(test_seq_id.tolist()))
    print(f"Evaluated sequence IDs: {unique_seq_ids}")

    # Evaluated region = union of all test window spans per seq_id.
    evaluated_region: dict[str, list[tuple[int, int]]] = {}
    predicted_region: dict[str, list[tuple[int, int]]] = {}
    bounds: dict[str, tuple[int, int]] = {}

    for sid in unique_seq_ids:
        mask = test_seq_id == sid
        spans = list(zip(test_starts[mask].tolist(), test_ends[mask].tolist()))
        evaluated_region[sid] = merge_intervals(spans)

        pred_mask = mask & (test_pred == 1)
        pred_spans = list(zip(test_starts[pred_mask].tolist(), test_ends[pred_mask].tolist()))
        predicted_region[sid] = merge_intervals(pred_spans)

        bounds[sid] = (
            int(min(s for s, _ in evaluated_region[sid])),
            int(max(e for _, e in evaluated_region[sid])),
        )

    true_region = load_true_repeat_intervals(args.fasta, set(unique_seq_ids), bounds)

    per_seq_rows = []
    predicted_interval_rows = []
    true_interval_rows = []

    for sid in unique_seq_ids:
        eval_region = evaluated_region[sid]
        pred_region = predicted_region[sid]
        true_region_sid = merge_intervals(
            intersect_intervals(true_region.get(sid, []), eval_region)
        )

        overlap = intersect_intervals(pred_region, true_region_sid)
        overlap_bp = total_bp(overlap)
        pred_bp = total_bp(pred_region)
        true_bp = total_bp(true_region_sid)
        union_bp = pred_bp + true_bp - overlap_bp

        precision = overlap_bp / pred_bp if pred_bp > 0 else float("nan")
        recall = overlap_bp / true_bp if true_bp > 0 else float("nan")
        iou = overlap_bp / union_bp if union_bp > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
            else float("nan")
        )

        per_seq_rows.append(
            {
                "seq_id": sid,
                "evaluated_bp": total_bp(eval_region),
                "true_repeat_bp": true_bp,
                "predicted_repeat_bp": pred_bp,
                "overlap_bp": overlap_bp,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "f1": f1,
            }
        )

        for s, e in pred_region:
            predicted_interval_rows.append({"seq_id": sid, "start": s, "end": e})
        for s, e in true_region_sid:
            true_interval_rows.append({"seq_id": sid, "start": s, "end": e})

    per_seq_df = pd.DataFrame(per_seq_rows)

    total_overlap_bp = int(per_seq_df["overlap_bp"].sum())
    total_pred_bp = int(per_seq_df["predicted_repeat_bp"].sum())
    total_true_bp = int(per_seq_df["true_repeat_bp"].sum())
    total_union_bp = total_pred_bp + total_true_bp - total_overlap_bp

    overall_precision = total_overlap_bp / total_pred_bp if total_pred_bp > 0 else float("nan")
    overall_recall = total_overlap_bp / total_true_bp if total_true_bp > 0 else float("nan")
    overall_iou = total_overlap_bp / total_union_bp if total_union_bp > 0 else float("nan")
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        and not np.isnan(overall_precision)
        and not np.isnan(overall_recall)
        else float("nan")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seq_df.to_csv(output_dir / "per_sequence_interval_metrics.csv", index=False)
    pd.DataFrame(predicted_interval_rows).to_csv(
        output_dir / "predicted_repeat_intervals.csv", index=False
    )
    pd.DataFrame(true_interval_rows).to_csv(
        output_dir / "true_repeat_intervals.csv", index=False
    )

    summary = {
        "repeat_threshold": args.repeat_threshold,
        "predicted_probability_threshold": args.predicted_probability_threshold,
        "n_test_windows": int(len(idx_test)),
        "total_evaluated_bp": int(per_seq_df["evaluated_bp"].sum()),
        "total_true_repeat_bp": total_true_bp,
        "total_predicted_repeat_bp": total_pred_bp,
        "total_overlap_bp": total_overlap_bp,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_iou": overall_iou,
        "overall_f1": overall_f1,
        "runtime_seconds": time.time() - start_time,
    }

    with (output_dir / "interval_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nInterval-level evaluation complete.")
    print(f"Overall precision={overall_precision:.4f} recall={overall_recall:.4f} "
          f"IoU={overall_iou:.4f} F1={overall_f1:.4f}")
    print(f"Saved: {output_dir / 'per_sequence_interval_metrics.csv'}")
    print(f"Saved: {output_dir / 'predicted_repeat_intervals.csv'}")
    print(f"Saved: {output_dir / 'true_repeat_intervals.csv'}")
    print(f"Saved: {output_dir / 'interval_eval_summary.json'}")
    print(f"Total runtime: {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
