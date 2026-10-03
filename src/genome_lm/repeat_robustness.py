from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeat classification robustness sweeps across thresholds, seeds, "
            "and sample sizes."
        )
    )
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument(
        "--repeat-thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4",
        help="Comma-separated repeat thresholds (e.g. 0.1,0.2,0.3,0.4).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="13,21,34",
        help="Comma-separated random seeds (e.g. 13,21,34,55,89).",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="5000",
        help="Comma-separated sample sizes from the embeddings set (e.g. 5000,10000,20000).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/baseline_cnn/robustness_runs.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="outputs/baseline_cnn/robustness_summary.csv",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def stratified_subsample_indices(y: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size >= len(y):
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise RuntimeError("Cannot stratify because one class is missing.")

    pos_target = int(round(sample_size * (len(pos_idx) / len(y))))
    pos_target = max(1, min(pos_target, len(pos_idx) - 1 if len(pos_idx) > 1 else 1))
    neg_target = sample_size - pos_target

    if neg_target <= 0:
        neg_target = 1
        pos_target = sample_size - 1
    if neg_target > len(neg_idx):
        neg_target = len(neg_idx)
        pos_target = sample_size - neg_target

    pos_sel = rng.choice(pos_idx, size=pos_target, replace=False)
    neg_sel = rng.choice(neg_idx, size=neg_target, replace=False)

    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def run_single_experiment(
    x: np.ndarray,
    repeat_fraction: np.ndarray,
    threshold: float,
    sample_size: int,
    seed: int,
    test_size: float,
) -> dict[str, float | int]:
    y = (repeat_fraction >= threshold).astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            f"Single-class labels for threshold={threshold}. "
            "Try a different threshold range."
        )

    indices = stratified_subsample_indices(y, sample_size=sample_size, seed=seed)
    x_sub = x[indices]
    y_sub = y[indices]

    sub_positives = int(y_sub.sum())
    sub_negatives = int(len(y_sub) - sub_positives)

    if sub_positives < 2 or sub_negatives < 2:
        raise RuntimeError(
            "Subsample has too few examples per class. "
            "Increase sample size or change threshold."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x_sub,
        y_sub,
        test_size=test_size,
        random_state=seed,
        stratify=y_sub,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)

    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "threshold": threshold,
        "sample_size": int(len(y_sub)),
        "seed": seed,
        "full_repeats": positives,
        "full_non_repeats": negatives,
        "sub_repeats": sub_positives,
        "sub_non_repeats": sub_negatives,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()

    thresholds = parse_float_list(args.repeat_thresholds)
    seeds = parse_int_list(args.seeds)
    sample_sizes = parse_int_list(args.sample_sizes)

    print("=" * 72)
    print("Starting repeat robustness sweep")
    print(f"Embeddings file: {args.embeddings}")
    print(f"Thresholds: {thresholds}")
    print(f"Seeds: {seeds}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Test size: {args.test_size}")
    print("=" * 72)

    data = np.load(args.embeddings, allow_pickle=True)
    x = data["embeddings"]
    repeat_fraction = data["repeat_fraction"]

    print(f"Loaded embeddings shape: {x.shape}")

    total_runs = len(thresholds) * len(seeds) * len(sample_sizes)
    run_idx = 0
    rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        for sample_size in sample_sizes:
            if sample_size > len(x):
                print(
                    f"Skipping sample_size={sample_size} because dataset has only {len(x)} samples."
                )
                continue

            for seed in seeds:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] "
                    f"threshold={threshold}, sample_size={sample_size}, seed={seed}"
                )
                row = run_single_experiment(
                    x=x,
                    repeat_fraction=repeat_fraction,
                    threshold=threshold,
                    sample_size=sample_size,
                    seed=seed,
                    test_size=args.test_size,
                )
                rows.append(row)

    if not rows:
        raise RuntimeError("No successful runs were completed. Check your sweep parameters.")

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["threshold", "sample_size"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .sort_values(["threshold", "sample_size"])
    )

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSweep complete.")
    print(f"Saved run-level metrics: {output_csv}")
    print(f"Saved summary metrics:   {summary_csv}")
    print("\nTop summary rows:")
    print(summary_df.head(10).to_string(index=False))
    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
