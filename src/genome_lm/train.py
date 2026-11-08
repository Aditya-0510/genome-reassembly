from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import DNAMaskedLMDataset, build_windows
from .model import GenomicCNNMLM
from .tokenizer import DNATokenizer


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps before cosine decay (0 disables warmup).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. Set to 0 to disable.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (requires CUDA).",
    )
    parser.add_argument(
        "--reverse-complement-prob",
        type=float,
        default=0.0,
        help="Probability of reverse-complementing a window during training (augmentation).",
    )

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

    print("=" * 72)
    print("Starting DNA MLM training")
    print(f"FASTA: {args.fasta}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"mask_prob={args.mask_probability}, batch_size={args.batch_size}, epochs={args.epochs}"
    )
    print("=" * 72)

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

    print(f"Generated windows: {len(windows)}")

    dataset = DNAMaskedLMDataset(
        windows=windows,
        tokenizer=tokenizer,
        mask_probability=args.mask_probability,
        seed=args.seed,
        reverse_complement_prob=args.reverse_complement_prob,
    )

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Dataset split: train={len(train_ds)} val={len(val_ds)}")

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
    print(f"Device: {device}")
    model = GenomicCNNMLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(
        "Model: "
        f"embedding_dim={args.embedding_dim}, channels={args.channels}, num_layers={args.num_layers}, "
        f"params={total_params:,} (trainable={trainable_params:,})"
    )
    print(f"Batches per epoch: train={len(train_loader)} val={len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_batches = 0
        print(f"\n[Epoch {epoch}/{args.epochs}] Training...")

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_acc += masked_accuracy(logits, labels)
            train_batches += 1

            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                avg_loss = total_train_loss / train_batches
                avg_acc = total_train_acc / train_batches
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  batch {batch_idx:>4}/{len(train_loader)} | "
                    f"avg_train_loss={avg_loss:.4f} avg_train_acc={avg_acc:.4f} lr={current_lr:.2e}"
                )

        train_loss = total_train_loss / max(1, train_batches)
        train_acc = total_train_acc / max(1, train_batches)

        print(f"[Epoch {epoch}/{args.epochs}] Validating...")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
        epoch_seconds = time.time() - epoch_start

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_masked_accuracy": train_acc,
            "val_loss": val_loss,
            "val_masked_accuracy": val_acc,
            "val_perplexity": val_perplexity,
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={val_perplexity:.2f} | "
            f"time={epoch_seconds:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  New best checkpoint saved: {checkpoint_path}")

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"History saved to: {output_dir / 'training_history.json'}")
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"Saved model to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
