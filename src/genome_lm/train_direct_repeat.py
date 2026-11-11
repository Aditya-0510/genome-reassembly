from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import DNAWindow, build_windows
from .direct_repeat_model import DirectRepeatCNN
from .tokenizer import DNATokenizer


class RepeatWindowDataset(Dataset):
    def __init__(
        self,
        windows: list[DNAWindow],
        tokenizer: DNATokenizer,
        repeat_label_threshold: float,
    ) -> None:
        self.windows = windows
        self.tokenizer = tokenizer
        self.repeat_label_threshold = repeat_label_threshold

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        tokens = torch.tensor(self.tokenizer.encode(window.sequence), dtype=torch.long)
        label = float(window.repeat_fraction >= self.repeat_label_threshold)
        return {
            "input_ids": tokens,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct repeat classifier on DNA windows.")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=25000)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)
    parser.add_argument("--repeat-label-threshold", type=float, default=0.2)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/direct_repeat_cnn")
    return parser.parse_args()


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    labels_i = labels.astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels_i == 1)))
    tn = int(np.sum((preds == 0) & (labels_i == 0)))
    fp = int(np.sum((preds == 1) & (labels_i == 0)))
    fn = int(np.sum((preds == 0) & (labels_i == 1)))

    total = max(1, len(labels_i))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(
    model: DirectRepeatCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    batches = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

            total_loss += float(loss.item())
            batches += 1

    if batches == 0:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return total_loss / batches, compute_metrics(probs_np, labels_np, threshold=0.5)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()

    print("=" * 72)
    print("Starting direct repeat classifier training")
    print(f"FASTA: {args.fasta}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"repeat_label_threshold={args.repeat_label_threshold}, epochs={args.epochs}"
    )
    print("=" * 72)

    tokenizer = DNATokenizer()
    windows = build_windows(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )
    if not windows:
        raise RuntimeError("No windows generated. Check FASTA path or relax filters.")

    y = np.array([float(w.repeat_fraction >= args.repeat_label_threshold) for w in windows])
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    print(f"Generated windows: {len(windows)} | repeats={positives} non_repeats={negatives}")
    if positives == 0 or negatives == 0:
        raise RuntimeError("Training labels are single-class. Adjust threshold/data.")

    indices = np.arange(len(windows))
    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_val = max(1, int(len(pos_idx) * args.val_fraction))
    neg_val = max(1, int(len(neg_idx) * args.val_fraction))
    val_idx = np.concatenate([pos_idx[:pos_val], neg_idx[:neg_val]])
    train_idx = np.concatenate([pos_idx[pos_val:], neg_idx[neg_val:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    train_windows = [windows[int(i)] for i in train_idx]
    val_windows = [windows[int(i)] for i in val_idx]

    print(f"Dataset split: train={len(train_windows)} val={len(val_windows)}")

    train_ds = RepeatWindowDataset(train_windows, tokenizer, args.repeat_label_threshold)
    val_ds = RepeatWindowDataset(val_windows, tokenizer, args.repeat_label_threshold)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectRepeatCNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(1, batches)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab,
                    "args": vars(args),
                    "best_val_f1": best_val_f1,
                },
                output_dir / "best_model.pt",
            )
            print(f"  New best checkpoint saved: {output_dir / 'best_model.pt'}")

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "num_windows": int(len(windows)),
        "num_train": int(len(train_windows)),
        "num_val": int(len(val_windows)),
        "repeat_windows": int(positives),
        "non_repeat_windows": int(negatives),
        "best_val_f1": float(best_val_f1),
        "runtime_seconds": float(time.time() - start),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Best val_f1={best_val_f1:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
