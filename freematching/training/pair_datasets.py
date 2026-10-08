from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVF


def _read_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    for key in ("items", "pairs", "samples"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Cannot find a list of pairs in manifest: {path}")


def _first_path(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _pick_pair_paths(item: dict[str, Any]) -> tuple[str, str]:
    candidates = [
        ("image_a", "image_b"),
        ("im_A", "im_B"),
        ("source_image", "target_image"),
        ("reference_image", "target_image"),
        ("reference_images", "target_image"),
        ("img_path1", "img_path2"),
    ]
    for key_a, key_b in candidates:
        if key_a in item and key_b in item:
            return _first_path(item[key_a]), _first_path(item[key_b])
    raise KeyError(f"Pair entry has no recognized image fields: {sorted(item.keys())}")


class FrozenPairDataset(Dataset):
    """Deterministic image-pair dataset for weak adaptation."""

    def __init__(
        self,
        manifest_path: str,
        *,
        image_root: str = "",
        height: int = 288,
        width: int = 512,
        swap_pair: bool = False,
        dataset_name: str = "frozen_pairs",
    ):
        self.manifest_path = str(manifest_path)
        self.items = _read_manifest(manifest_path)
        self.image_root = Path(image_root) if image_root else None
        self.height = int(height)
        self.width = int(width)
        self.swap_pair = bool(swap_pair)
        self.dataset_name = dataset_name
        if not self.items:
            raise ValueError(f"Empty pair manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or self.image_root is None:
            return candidate
        return self.image_root / candidate

    def _load(self, path: str) -> torch.Tensor:
        with Image.open(self._resolve(path)) as image:
            image = image.convert("RGB")
            image = TVF.resize(
                image,
                [self.height, self.width],
                interpolation=TVF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            return TVF.to_tensor(image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        path_a, path_b = _pick_pair_paths(item)
        if self.swap_pair or bool(item.get("swap_pair", False)):
            path_a, path_b = path_b, path_a
        im_a = self._load(path_a)
        im_b = self._load(path_b)
        sample_id = str(
            item.get("sample_id")
            or item.get("id")
            or item.get("sample_key")
            or f"{self.dataset_name}_{index:08d}"
        )
        return {
            "im_A": im_a,
            "im_B": im_b,
            "ground_truth_flow": torch.zeros(2, self.height, self.width, dtype=torch.float32),
            "validity_mask": torch.ones(self.height, self.width, dtype=torch.float32),
            "dataset_name": self.dataset_name,
            "sample_id": sample_id,
            "sample_dir": str(item.get("sample_dir", "")),
            "source_path_a": str(self._resolve(path_a)),
            "source_path_b": str(self._resolve(path_b)),
        }

class NormalizedPairDataset(Dataset):
    """Expose a stable common sample schema for mixed weak data sources."""

    required_keys = ("im_A", "im_B", "ground_truth_flow", "validity_mask")

    def __init__(self, dataset: Dataset, dataset_name: str):
        self.dataset = dataset
        self.dataset_name = str(dataset_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        output = {key: sample[key] for key in self.required_keys}
        output.update(
            dataset_name=self.dataset_name,
            sample_id=str(sample.get("sample_id", "")),
            sample_dir=str(sample.get("sample_dir", "")),
            index=int(index),
        )
        return output


class FlowWeakPairDataset(FrozenPairDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("dataset_name", "flow_weak")
        super().__init__(*args, **kwargs)


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: str | os.PathLike[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

