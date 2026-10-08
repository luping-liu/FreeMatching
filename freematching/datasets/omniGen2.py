import hashlib
import json
import os
import os.path as osp
from glob import glob
from typing import List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


DEFAULT_OMNIGEN2_ROOT = 'data/datasets/OmniGen2'
DEFAULT_VIDEO_SHARDS_DIR = osp.join(DEFAULT_OMNIGEN2_ROOT, "merged_shards_video")
DEFAULT_IMAGE_ROOT = osp.join(DEFAULT_OMNIGEN2_ROOT, "images")
DEFAULT_CACHE_DIR = osp.join(DEFAULT_OMNIGEN2_ROOT, "cache")


def _as_first_path(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


class OmniGen2VideoWeakDataset(Dataset):
    """
    UNO-style weak-supervision dataset from OmniGen2 video shards.

    This dataset only uses JSONL metadata and image paths. `match_path_im_B` is
    deliberately not loaded or used as supervision; `match_ratio_im_B` is only a
    conservative quality filter for choosing image pairs.
    """

    def __init__(
        self,
        json_files: Sequence[str],
        image_root_dir: str = DEFAULT_IMAGE_ROOT,
        ht: int = 288,
        wt: int = 512,
        min_match_ratio: float = 0.2,
        max_match_ratio: float = 0.8,
        source_match_ratio_ranges: Optional[dict] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        debug: bool = False,
    ):
        self.json_files = sorted(str(path) for path in json_files)
        self.image_root_dir = image_root_dir
        self.ht = ht
        self.wt = wt
        self.min_match_ratio = min_match_ratio
        self.max_match_ratio = max_match_ratio
        self.source_match_ratio_ranges = source_match_ratio_ranges or {
            "video_edit": (0.4, 0.8),
            "video_icgen": (0.2, 0.8),
        }
        self.debug = debug
        self.file_handles = {}

        if not self.json_files:
            raise FileNotFoundError("No OmniGen2 video jsonl files were provided.")

        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as exc:
            print(f"Warning: cannot create cache dir {cache_dir}: {exc}")
        cache_key = self._cache_key()
        cache_path = osp.join(cache_dir, f"omnigen2_video_weak_index_{cache_key}.pt")
        self.valid_indices = self._load_or_build_index(cache_path)
        print(f">>> OmniGen2 video weak samples: {len(self.valid_indices)}")

    def _cache_key(self):
        payload = {
            "json_files": self.json_files,
            "min_match_ratio": self.min_match_ratio,
            "max_match_ratio": self.max_match_ratio,
            "source_match_ratio_ranges": self.source_match_ratio_ranges,
            "dataset": "omnigen2_video_weak_v1",
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def _load_or_build_index(self, cache_path):
        if osp.exists(cache_path):
            try:
                print(f">>> Loading OmniGen2 video weak index cache: {cache_path}")
                return torch.load(cache_path)
            except Exception as exc:
                print(f"Warning: failed to load cache {cache_path}: {exc}. Rebuilding.")

        print(">>> Building OmniGen2 video weak index from jsonl files...")
        valid_indices = self._build_index()
        try:
            torch.save(valid_indices, cache_path)
            print(f">>> Saved OmniGen2 video weak index cache: {cache_path}")
        except Exception as exc:
            print(f"Warning: failed to save cache {cache_path}: {exc}")
        return valid_indices

    def _build_index(self) -> List[Tuple[int, int]]:
        indices = []
        for file_idx, json_path in enumerate(self.json_files):
            if not osp.exists(json_path):
                print(f"Warning: missing jsonl file: {json_path}")
                continue

            if self.debug:
                print(f"Indexing {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    source = item.get("source", "")
                    min_ratio, max_ratio = self.source_match_ratio_ranges.get(
                        source, (self.min_match_ratio, self.max_match_ratio)
                    )
                    match_ratio = float(item.get("match_ratio_im_B", -1.0))
                    if match_ratio < min_ratio or match_ratio > max_ratio:
                        continue
                    if not _as_first_path(item.get("reference_images")):
                        continue
                    if not _as_first_path(item.get("target_image")):
                        continue

                    indices.append((file_idx, offset))
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def _read_item(self, idx):
        file_idx, offset = self.valid_indices[idx]
        if file_idx not in self.file_handles:
            self.file_handles[file_idx] = open(self.json_files[file_idx], "r", encoding="utf-8")

        f = self.file_handles[file_idx]
        f.seek(offset)
        return json.loads(f.readline())

    def _load_image_tensor(self, rel_path):
        image_path = osp.join(self.image_root_dir, rel_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = F.resize(image, [self.ht, self.wt], antialias=True)
            return F.to_tensor(image)

    def __getitem__(self, idx):
        try:
            item = self._read_item(idx)
            ref_path = _as_first_path(item.get("reference_images"))
            target_path = _as_first_path(item.get("target_image"))

            im_a = self._load_image_tensor(ref_path)
            im_b = self._load_image_tensor(target_path)

            return {
                "im_A": im_a,
                "im_B": im_b,
                "ground_truth_flow": torch.zeros(2, self.ht, self.wt, dtype=torch.float32),
                "validity_mask": torch.ones(self.ht, self.wt, dtype=torch.float32),
                "im_A_depth": torch.zeros(self.ht, self.wt, dtype=torch.float32),
                "sample_id": self._sample_id(item),
                "sample_dir": osp.dirname(target_path),
                "source_type": item.get("source", ""),
                "match_ratio_im_B": float(item.get("match_ratio_im_B", 0.0)),
            }
        except Exception as exc:
            if self.debug:
                print(f"Warning: failed to load OmniGen2 video sample {idx}: {exc}")
            return self.__getitem__((idx + 1) % len(self))

    def _sample_id(self, item):
        ref_path = _as_first_path(item.get("reference_images"))
        target_path = _as_first_path(item.get("target_image"))
        payload = {
            "source": item.get("source", ""),
            "reference_image": ref_path,
            "target_image": target_path,
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def __del__(self):
        if hasattr(self, "file_handles"):
            for f in self.file_handles.values():
                try:
                    f.close()
                except Exception:
                    pass


class OmniGen2VideoWeakBuilder:
    def __init__(
        self,
        data_root: str = DEFAULT_OMNIGEN2_ROOT,
        shards_dir: Optional[str] = None,
        image_root_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.shards_dir = shards_dir or osp.join(data_root, "merged_shards_video")
        self.image_root_dir = image_root_dir or osp.join(data_root, "images")

    def build_dataset(
        self,
        ht: int = 288,
        wt: int = 512,
        min_match_ratio: float = 0.2,
        max_match_ratio: float = 0.8,
        source_match_ratio_ranges: Optional[dict] = None,
        debug: bool = False,
    ) -> List[Dataset]:
        json_files = sorted(glob(osp.join(self.shards_dir, "merged_part_*.jsonl")))
        dataset = OmniGen2VideoWeakDataset(
            json_files=json_files,
            image_root_dir=self.image_root_dir,
            ht=ht,
            wt=wt,
            min_match_ratio=min_match_ratio,
            max_match_ratio=max_match_ratio,
            source_match_ratio_ranges=source_match_ratio_ranges,
            cache_dir=osp.join(self.data_root, "cache"),
            debug=debug,
        )
        return [dataset]


