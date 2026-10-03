from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DNAMaskedLMDataset, build_windows
from .model import GenomicCNNMLM
from .tokenizer import DNATokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract window embeddings from trained model.")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--min-acgt-fraction", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="outputs/embeddings.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    print("=" * 72)
    print("Starting embedding extraction")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FASTA: {args.fasta}")
    print(
        "Config: "
        f"window={args.window_size}, stride={args.stride}, max_windows={args.max_windows}, "
        f"batch_size={args.batch_size}"
    )
    print("=" * 72)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]

    tokenizer = DNATokenizer()

    model = GenomicCNNMLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=train_args["embedding_dim"],
        channels=train_args["channels"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        "Loaded model config: "
        f"embedding_dim={train_args['embedding_dim']}, channels={train_args['channels']}, "
        f"num_layers={train_args['num_layers']}"
    )

    windows = build_windows(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        min_acgt_fraction=args.min_acgt_fraction,
    )
    print(f"Generated windows: {len(windows)}")

    dataset = DNAMaskedLMDataset(
        windows=windows,
        tokenizer=tokenizer,
        mask_probability=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Batches to process: {len(loader)}")

    embedding_list: list[np.ndarray] = []
    repeat_fraction_list: list[float] = []
    seq_id_list: list[str] = []
    start_list: list[int] = []
    end_list: list[int] = []

    total_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"]
            logits, token_embeddings = model(input_ids, return_embeddings=True)
            _ = logits

            pooled = token_embeddings.mean(dim=1).cpu().numpy()
            embedding_list.append(pooled)

            repeat_fraction_list.extend(batch["repeat_fraction"].numpy().tolist())
            seq_id_list.extend(batch["seq_id"])
            start_list.extend(batch["start"].numpy().tolist())
            end_list.extend(batch["end"].numpy().tolist())

            total_processed += input_ids.size(0)
            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(
                    f"  batch {batch_idx:>4}/{len(loader)} | "
                    f"windows_processed={total_processed}"
                )

    embeddings = np.concatenate(embedding_list, axis=0) if embedding_list else np.empty((0, 1))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        repeat_fraction=np.array(repeat_fraction_list),
        seq_id=np.array(seq_id_list),
        start=np.array(start_list),
        end=np.array(end_list),
    )

    elapsed = time.time() - start_time
    print(f"Saved embeddings to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Extraction time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
