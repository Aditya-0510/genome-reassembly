from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .direct_repeat_model import DirectRepeatCNN
from .repeat_inference import count_subsequence, interval_total_bp, merge_intervals
from .tokenizer import DNATokenizer


@dataclass(frozen=True)
class DirectRepeatInferenceConfig:
    checkpoint_path: str
    window_size: int = 512
    stride: int = 64
    min_acgt_fraction: float = 0.9
    predicted_probability_threshold: float = 0.5
    min_repeat_fraction_positive: float = 0.05
    kmer_size: int = 9
    top_k_segments: int = 10
    batch_size: int = 128


class DirectRepeatInferenceService:
    def __init__(self, config: DirectRepeatInferenceConfig) -> None:
        self.config = config
        self.tokenizer = DNATokenizer()
        self.model = self._load_model(config.checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> DirectRepeatCNN:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        train_args = ckpt["args"]
        model = DirectRepeatCNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=train_args["embedding_dim"],
            channels=train_args["channels"],
            num_layers=train_args["num_layers"],
            dropout=train_args["dropout"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    def normalize_sequence(self, sequence: str) -> str:
        return "".join(sequence.upper().split())

    def validate_sequence(self, sequence: str) -> list[str]:
        errors: list[str] = []
        if not sequence:
            errors.append("Sequence is empty.")
            return errors

        invalid = sorted({ch for ch in sequence if ch not in {"A", "C", "G", "T", "N"}})
        if invalid:
            errors.append(
                "Sequence contains unsupported characters: " + ", ".join(invalid)
            )

        if len(sequence) < self.config.window_size:
            errors.append(
                f"Sequence length ({len(sequence)}) is smaller than window size "
                f"({self.config.window_size})."
            )
        return errors

    def _build_windows(self, sequence: str) -> list[tuple[int, int, str]]:
        windows: list[tuple[int, int, str]] = []
        for start in range(0, len(sequence) - self.config.window_size + 1, self.config.stride):
            end = start + self.config.window_size
            window_seq = sequence[start:end]

            acgt_count = sum(1 for ch in window_seq if ch in {"A", "C", "G", "T"})
            if (acgt_count / self.config.window_size) < self.config.min_acgt_fraction:
                continue
            windows.append((start, end, window_seq))
        return windows

    def _predict_window_probabilities(self, windows: list[tuple[int, int, str]]) -> list[float]:
        probabilities: list[float] = []
        with torch.no_grad():
            for batch_start in range(0, len(windows), self.config.batch_size):
                batch = windows[batch_start : batch_start + self.config.batch_size]
                tokens = [self.tokenizer.encode(window_seq) for _, _, window_seq in batch]
                input_tensor = torch.tensor(tokens, dtype=torch.long)
                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                probabilities.extend(float(p) for p in probs)
        return probabilities

    def _extract_top_segments(
        self,
        sequence: str,
        intervals: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        k = self.config.kmer_size
        counter: Counter[str] = Counter()
        first_positions: dict[str, int] = {}

        for start, end in intervals:
            segment = sequence[start:end]
            if len(segment) < k:
                continue
            for i in range(0, len(segment) - k + 1):
                kmer = segment[i : i + k]
                if "N" in kmer:
                    continue
                counter[kmer] += 1
                global_pos = start + i
                if kmer not in first_positions:
                    first_positions[kmer] = global_pos

        rows: list[dict[str, Any]] = []
        for repeat_unit, count in counter.most_common(self.config.top_k_segments):
            rows.append(
                {
                    "repeat_unit": repeat_unit,
                    "count": int(count),
                    "first_position": int(first_positions[repeat_unit]),
                }
            )
        return rows

    def predict(self, sequence: str, query_subsequence: str | None = None) -> dict[str, Any]:
        normalized_seq = self.normalize_sequence(sequence)
        errors = self.validate_sequence(normalized_seq)
        if errors:
            return {"errors": errors}

        windows = self._build_windows(normalized_seq)
        if not windows:
            return {
                "errors": [
                    "No valid windows were produced. Lower min_acgt_fraction or provide a longer sequence."
                ]
            }

        probs = self._predict_window_probabilities(windows)
        repeat_windows: list[dict[str, Any]] = []
        repeat_spans: list[tuple[int, int]] = []

        for (start, end, _), prob in zip(windows, probs):
            is_repeat = bool(prob >= self.config.predicted_probability_threshold)
            repeat_windows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "repeat_probability": float(prob),
                    "is_repeat": is_repeat,
                }
            )
            if is_repeat:
                repeat_spans.append((start, end))

        repeat_intervals = merge_intervals(repeat_spans)
        total_repeat_bp = interval_total_bp(repeat_intervals)
        repeat_fraction = total_repeat_bp / len(normalized_seq)
        repeated = repeat_fraction >= self.config.min_repeat_fraction_positive
        repeated_segments = self._extract_top_segments(normalized_seq, repeat_intervals)

        query_result: dict[str, Any] | None = None
        if query_subsequence:
            normalized_query = self.normalize_sequence(query_subsequence)
            invalid_query = [
                ch for ch in normalized_query if ch not in {"A", "C", "G", "T", "N"}
            ]
            if invalid_query:
                query_result = {
                    "query_subsequence": normalized_query,
                    "error": "Query subsequence contains unsupported characters.",
                }
            else:
                total_count = count_subsequence(normalized_seq, normalized_query)
                repeat_only_count = 0
                for start, end in repeat_intervals:
                    repeat_only_count += count_subsequence(normalized_seq[start:end], normalized_query)
                query_result = {
                    "query_subsequence": normalized_query,
                    "count_in_sequence": int(total_count),
                    "count_in_predicted_repeat_intervals": int(repeat_only_count),
                    "is_repeated": bool(total_count > 1),
                }

        return {
            "errors": [],
            "sequence_summary": {
                "length": int(len(normalized_seq)),
                "window_size": int(self.config.window_size),
                "stride": int(self.config.stride),
                "valid_windows": int(len(windows)),
                "model_type": "direct_repeat_cnn",
                "model_predicted_repeat_intervals": int(len(repeat_intervals)),
                "heuristic_repeat_intervals": 0,
                "predicted_repeat_intervals": int(len(repeat_intervals)),
                "total_repeat_bp": int(total_repeat_bp),
                "repeat_fraction": float(repeat_fraction),
                "is_repeated": bool(repeated),
            },
            "repeat_intervals": [
                {
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start),
                    "sequence": normalized_seq[start:end],
                }
                for start, end in repeat_intervals
            ],
            "repeat_windows": repeat_windows,
            "repeated_segments": repeated_segments,
            "query_result": query_result,
        }


def build_default_direct_service(
    checkpoint_path: str = "outputs/direct_repeat_cnn/best_model.pt",
    window_size: int = 512,
    stride: int = 64,
) -> DirectRepeatInferenceService:
    config = DirectRepeatInferenceConfig(
        checkpoint_path=str(Path(checkpoint_path)),
        window_size=window_size,
        stride=stride,
    )
    return DirectRepeatInferenceService(config)
