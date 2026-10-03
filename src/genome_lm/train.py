from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import DNAMaskedLMDataset, build_windows
from .model import GenomicCNNMLM
from .tokenizer import DNATokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DNA masked language model.")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=20000)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)

    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_cnn")
    return parser.parse_args()


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).float().mean().item()
    return float(correct)


def evaluate(
    model: GenomicCNNMLM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            total_acc += masked_accuracy(logits, labels)
            batches += 1

    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_acc / batches


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DNATokenizer()
    windows = build_windows(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )

    if not windows:
        raise RuntimeError("No windows generated. Relax filters or check FASTA path.")

    dataset = DNAMaskedLMDataset(
        windows=windows,
        tokenizer=tokenizer,
        mask_probability=args.mask_probability,
        seed=args.seed,
    )

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

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
    model = GenomicCNNMLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += masked_accuracy(logits, labels)
            train_batches += 1

        train_loss = total_train_loss / max(1, train_batches)
        train_acc = total_train_acc / max(1, train_batches)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_masked_accuracy": train_acc,
            "val_loss": val_loss,
            "val_masked_accuracy": val_acc,
            "val_perplexity": val_perplexity,
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={val_perplexity:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"Saved model to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
