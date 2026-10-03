from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a repeat/non-repeat classifier from gLM embeddings."
    )
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument(
        "--repeat-threshold",
        type=float,
        default=0.2,
        help="Window is labeled repeat if repeat_fraction >= threshold.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    print("=" * 72)
    print("Starting repeat baseline training")
    print(f"Embeddings file: {args.embeddings}")
    print(
        "Config: "
        f"repeat_threshold={args.repeat_threshold}, test_size={args.test_size}, seed={args.seed}"
    )
    print("=" * 72)

    data = np.load(args.embeddings, allow_pickle=True)
    x = data["embeddings"]
    repeat_fraction = data["repeat_fraction"]
    print(f"Loaded embeddings shape: {x.shape}")

    y = (repeat_fraction >= args.repeat_threshold).astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    print(f"Label distribution: repeats={positives} non_repeats={negatives}")

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            "Labels are single-class. Try another threshold or data with soft-masked repeats."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Split sizes: train={len(y_train)} test={len(y_test)}")

    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    print("Training LogisticRegression classifier...")
    clf.fit(x_train, y_train)
    print("Model training complete. Evaluating...")

    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print(f"Samples: {len(y)} | repeats={positives} non_repeats={negatives}")
    print("Repeat classification baseline metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
