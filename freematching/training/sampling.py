from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def parse_source_ratios(spec: str, sources: Iterable[str]) -> dict[str, float]:
    source_list = list(dict.fromkeys(sources))
    if not source_list:
        raise ValueError("No weak data sources were provided.")
    if not spec:
        value = 1.0 / len(source_list)
        return {source: value for source in source_list}

    ratios: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"Invalid weak ratio entry '{item}', expected source:value")
        ratios[name.strip()] = float(value)

    unknown = set(ratios) - set(source_list)
    missing = set(source_list) - set(ratios)
    if unknown:
        raise ValueError(f"Ratios contain unknown sources: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Ratios are missing sources: {sorted(missing)}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Weak data ratios must sum to a positive value.")
    return {name: value / total for name, value in ratios.items()}


def build_source_sample_weights(
    dataset_lengths: list[int],
    dataset_sources: list[str],
    source_ratios: dict[str, float],
) -> torch.Tensor:
    if len(dataset_lengths) != len(dataset_sources):
        raise ValueError("dataset_lengths and dataset_sources must have the same length")
    totals = defaultdict(int)
    for length, source in zip(dataset_lengths, dataset_sources):
        totals[source] += int(length)
    weights: list[float] = []
    for length, source in zip(dataset_lengths, dataset_sources):
        total_length = totals[source]
        if total_length <= 0:
            raise ValueError(f"Source '{source}' has no samples")
        per_sample = source_ratios[source] / total_length
        weights.extend([per_sample] * int(length))
    return torch.tensor(weights, dtype=torch.double)

