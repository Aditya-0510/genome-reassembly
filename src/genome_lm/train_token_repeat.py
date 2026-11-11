from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .token_repeat_model import TokenRepeatCNN
from .tokenizer import DNATokenizer


@dataclass(frozen=True)
class TokenLabeledWindow:
    sequence: str
    labels: list[int]


class TokenRepeatDataset(Dataset):
    def __init__(self, windows: list[TokenLabeledWindow], tokenizer: DNATokenizer) -> None:
        self.windows = windows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.windows[idx]
        input_ids = torch.tensor(self.tokenizer.encode(window.sequence), dtype=torch.long)
        labels = torch.tensor(window.labels, dtype=torch.float32)
        return {"input_ids": input_ids, "labels": labels}


def resolve_fasta_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    if len(path.parts) == 1:
        candidates = [
            Path("dataset") / "ncbi_dataset" / "data" / "GCF_000001405.40" / path.name,
            Path("dataset") / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        dataset_root = Path("dataset")
        if dataset_root.exists():
            for found in dataset_root.rglob(path.name):
                if found.is_file():
                    return found

    raise FileNotFoundError(
        f"FASTA file not found: {raw_path}\n"
        "Tip: pass workspace-relative path like:\n"
        "dataset/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna"
    )


def build_token_windows(
    fasta_path: Path,
    window_size: int,
    stride: int,
    max_windows: int | None,
    min_acgt_fraction: float,
) -> list[TokenLabeledWindow]:
    windows: list[TokenLabeledWindow] = []
    produced = 0
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        raw = str(record.seq)
        upper = raw.upper()
        for start in range(0, max(0, len(upper) - window_size + 1), stride):
            end = start + window_size
            window_upper = upper[start:end]
            if len(window_upper) != window_size:
                continue

            acgt = sum(1 for ch in window_upper if ch in {"A", "C", "G", "T"})
            if (acgt / window_size) < min_acgt_fraction:
                continue

            window_raw = raw[start:end]
            labels = [1 if ch.islower() else 0 for ch in window_raw]
            windows.append(TokenLabeledWindow(sequence=window_upper, labels=labels))
            produced += 1

            if max_windows is not None and produced >= max_windows:
                return windows
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train token-level repeat labeling model.")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=50000)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help=(
            "Positive-class weight for BCE loss. If omitted, auto-computed from train "
            "token imbalance as neg_tokens/pos_tokens (clipped to [1, 50])."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/token_repeat_cnn")
    return parser.parse_args()


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)

    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    total = max(1, len(y))
    acc = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(
    model: TokenRepeatCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    batches = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            y = labels.cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_labels.append(y)

            loss_sum += float(loss.item())
            batches += 1

    if batches == 0:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return loss_sum / batches, metrics_from_probs(probs_np, labels_np, threshold=0.5)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()

    fasta_path = resolve_fasta_path(args.fasta)
    print("=" * 72)
    print("Starting token-level repeat model training")
    print(f"FASTA: {fasta_path}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"epochs={args.epochs}"
    )
    print("=" * 72)

    windows = build_token_windows(
        fasta_path=fasta_path,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )
    if not windows:
        raise RuntimeError("No windows generated. Check FASTA path or relax filters.")

    total_tokens = len(windows) * args.window_size
    total_repeat_tokens = int(sum(sum(w.labels) for w in windows))
    print(
        f"Generated windows: {len(windows)} | repeat_token_fraction="
        f"{(total_repeat_tokens / max(1, total_tokens)):.4f}"
    )

    idx = np.arange(len(windows))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_size = max(1, int(len(idx) * args.val_fraction))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_windows = [windows[int(i)] for i in train_idx]
    val_windows = [windows[int(i)] for i in val_idx]
    print(f"Dataset split: train={len(train_windows)} val={len(val_windows)}")

    train_pos_tokens = int(sum(sum(w.labels) for w in train_windows))
    train_total_tokens = len(train_windows) * args.window_size
    train_neg_tokens = int(train_total_tokens - train_pos_tokens)

    if train_pos_tokens == 0:
        raise RuntimeError("Train split has zero positive repeat tokens. Adjust data/split.")

    if args.pos_weight is None:
        inferred_pos_weight = train_neg_tokens / max(1, train_pos_tokens)
        pos_weight_value = float(min(max(inferred_pos_weight, 1.0), 50.0))
    else:
        pos_weight_value = float(max(args.pos_weight, 1.0))

    print(
        "Train token balance: "
        f"repeat={train_pos_tokens} non_repeat={train_neg_tokens} "
        f"pos_weight={pos_weight_value:.3f}"
    )

    tokenizer = DNATokenizer()
    train_ds = TokenRepeatDataset(train_windows, tokenizer)
    val_ds = TokenRepeatDataset(val_windows, tokenizer)

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
    model = TokenRepeatCNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

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
        "repeat_token_fraction": float(total_repeat_tokens / max(1, total_tokens)),
        "train_repeat_tokens": int(train_pos_tokens),
        "train_non_repeat_tokens": int(train_neg_tokens),
        "loss_pos_weight": float(pos_weight_value),
        "best_val_f1": float(best_val_f1),
        "runtime_seconds": float(time.time() - start),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Best val_f1={best_val_f1:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
