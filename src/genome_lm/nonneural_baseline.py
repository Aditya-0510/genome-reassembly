from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import DNAWindow, build_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-neural repeat/non-repeat baseline using GC content and k-mer "
            "frequency features, for fair comparison against gLM embeddings."
        )
    )
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=3,
        help="K-mer length used for frequency features.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=float,
        default=0.2,
        help="Window is labeled repeat if repeat_fraction >= threshold.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def build_kmer_vocab(kmer_size: int) -> list[str]:
    return ["".join(p) for p in itertools.product("ACGT", repeat=kmer_size)]


def compute_features(sequence: str, kmer_size: int, kmer_index: dict[str, int]) -> np.ndarray:
    gc_count = sequence.count("G") + sequence.count("C")
    gc_fraction = gc_count / len(sequence) if sequence else 0.0

    counts = np.zeros(len(kmer_index), dtype=np.float64)
    valid_kmers = 0
    for i in range(len(sequence) - kmer_size + 1):
        kmer = sequence[i : i + kmer_size]
        idx = kmer_index.get(kmer)
        if idx is not None:
            counts[idx] += 1
            valid_kmers += 1

    if valid_kmers > 0:
        counts /= valid_kmers

    return np.concatenate([[gc_fraction], counts])


def build_feature_matrix(
    windows: list[DNAWindow],
    kmer_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    kmer_vocab = build_kmer_vocab(kmer_size)
    kmer_index = {kmer: i for i, kmer in enumerate(kmer_vocab)}

    features = np.stack(
        [compute_features(w.sequence, kmer_size, kmer_index) for w in windows]
    )
    repeat_fraction = np.array([w.repeat_fraction for w in windows])
    return features, repeat_fraction


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("Starting non-neural repeat baseline (GC% + k-mer frequencies)")
    print(f"FASTA: {args.fasta}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"kmer_size={args.kmer_size}, repeat_threshold={args.repeat_threshold}"
    )
    print("=" * 72)

    windows = build_windows(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )
    if not windows:
        raise RuntimeError("No windows generated. Relax filters or check FASTA path.")

    print(f"Generated windows: {len(windows)}")

    x, repeat_fraction = build_feature_matrix(windows, args.kmer_size)
    print(f"Feature matrix shape: {x.shape} (1 GC feature + {4 ** args.kmer_size} k-mer features)")

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
    print("Training LogisticRegression classifier on non-neural features...")
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
    print("Non-neural baseline metrics (compare against repeat_baseline.py gLM metrics):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
