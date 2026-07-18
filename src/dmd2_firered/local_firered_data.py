import json
import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    raise TypeError(f"Expected path string/list, got {type(value).__name__}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("jsonl_path", str(path))
            item.setdefault("jsonl_lineno", lineno)
            rows.append(item)
    if not rows:
        raise ValueError(f"No records found in JSONL: {path}")
    return rows


def resolve_local_path(raw_path: str, local_data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path
    else:
        resolved = local_data_root / path
    if not resolved.exists():
        raise FileNotFoundError(f"Local FireRed data path missing: raw={raw_path} resolved={resolved}")
    return resolved


def _load_firered_image_utils(project_root: Path):
    utils_path = project_root / "train" / "src" / "utils" / "image_utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"FireRed image_utils.py not found: {utils_path}")
    spec = importlib.util.spec_from_file_location("_firered_image_utils_local_dmd2", str(utils_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import FireRed image utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resize_center_crop(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.convert("RGB")
    scale = max(height / image.height, width / image.width)
    resized_h = max(height, int(round(image.height * scale)))
    resized_w = max(width, int(round(image.width * scale)))
    image = TF.resize(image, [resized_h, resized_w], interpolation=TF.InterpolationMode.BICUBIC)
    image = TF.center_crop(image, [height, width])
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5], [0.5])


def _normalise_embedding(value: Any, field_name: str, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} did not load to a Tensor: {label} -> {type(value).__name__}")
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(f"{field_name} must be rank-2 [seq, dim], got shape={tuple(value.shape)} at {label}")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{field_name} is empty: shape={tuple(value.shape)} at {label}")
    return value


def _load_embedding(path: Path, field_name: str) -> torch.Tensor:
    return _normalise_embedding(torch.load(path, map_location="cpu", weights_only=False), field_name, str(path))


class LocalFireRedEditDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        local_data_root: str,
        height: int,
        width: int,
        max_samples: Optional[int],
        source_image_field: str,
        target_image_field: str,
        instruction_field: str,
        embedding_field: str,
        uncond_embedding_field: str,
        firered_project_root: Optional[str] = None,
        allow_cos_fallback: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser()
        self.local_data_root = Path(local_data_root).expanduser()
        self.height = int(height)
        self.width = int(width)
        self.source_image_field = source_image_field
        self.target_image_field = target_image_field
        self.instruction_field = instruction_field
        self.embedding_field = embedding_field
        self.uncond_embedding_field = uncond_embedding_field
        self.allow_cos_fallback = bool(allow_cos_fallback)
        self._load_firered_image = None
        self._load_firered_tensor = None

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"JSONL missing: {self.jsonl_path}")
        if not self.local_data_root.is_dir():
            raise FileNotFoundError(f"local_data_root missing: {self.local_data_root}")
        if self.allow_cos_fallback:
            if os.environ.get("FIRERED_USE_COS", "0") != "1":
                raise RuntimeError("allow_cos_fallback=true requires FIRERED_USE_COS=1")
            if not firered_project_root:
                raise ValueError("firered_project_root is required when allow_cos_fallback=true")
            project_root = Path(str(firered_project_root)).expanduser()
            if not project_root.is_dir():
                raise FileNotFoundError(f"firered_project_root missing: {project_root}")
            image_utils = _load_firered_image_utils(project_root)
            self._load_firered_image = image_utils.load_image
            self._load_firered_tensor = image_utils.load_tensor

        records = _read_jsonl(self.jsonl_path)
        if max_samples is not None:
            records = records[: int(max_samples)]
        if not records:
            raise ValueError(f"No records remain after max_samples={max_samples}")
        self.records = records

        required = [
            self.source_image_field,
            self.target_image_field,
            self.embedding_field,
            self.uncond_embedding_field,
        ]
        for idx, item in enumerate(self.records):
            missing = [key for key in required if key not in item or item[key] in (None, "")]
            if missing:
                raise KeyError(f"Record {idx} missing fields {missing} at {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        source_paths = _as_list(item[self.source_image_field])
        if len(source_paths) != 1:
            raise ValueError(
                f"Exactly one source image is supported, got {len(source_paths)} "
                f"at {item.get('jsonl_path')}:{item.get('jsonl_lineno')}"
            )

        raw_source_path = source_paths[0]
        raw_target_path = str(item[self.target_image_field])
        raw_prompt_path = str(item[self.embedding_field])
        raw_uncond_path = str(item[self.uncond_embedding_field])

        if self.allow_cos_fallback:
            assert self._load_firered_image is not None
            assert self._load_firered_tensor is not None
            source = _resize_center_crop(self._load_firered_image(raw_source_path), self.height, self.width)
            target = _resize_center_crop(self._load_firered_image(raw_target_path), self.height, self.width)
            prompt_embeds = _normalise_embedding(
                self._load_firered_tensor(raw_prompt_path),
                self.embedding_field,
                raw_prompt_path,
            )
            uncond_embeds = _normalise_embedding(
                self._load_firered_tensor(raw_uncond_path),
                self.uncond_embedding_field,
                raw_uncond_path,
            )
            source_path = raw_source_path
            target_path = raw_target_path
            prompt_path = raw_prompt_path
            uncond_path = raw_uncond_path
        else:
            source_path = resolve_local_path(raw_source_path, self.local_data_root)
            target_path = resolve_local_path(raw_target_path, self.local_data_root)
            prompt_path = resolve_local_path(raw_prompt_path, self.local_data_root)
            uncond_path = resolve_local_path(raw_uncond_path, self.local_data_root)
            source = _resize_center_crop(Image.open(source_path), self.height, self.width)
            target = _resize_center_crop(Image.open(target_path), self.height, self.width)
            prompt_embeds = _load_embedding(prompt_path, self.embedding_field)
            uncond_embeds = _load_embedding(uncond_path, self.uncond_embedding_field)

        return {
            "uid": str(item.get("uid", index)),
            "text": str(item.get(self.instruction_field, "")),
            "jsonl_path": str(item.get("jsonl_path", self.jsonl_path)),
            "jsonl_lineno": int(item.get("jsonl_lineno", -1)),
            "source_image_path": str(source_path),
            "target_image_path": str(target_path),
            "embedding_path": str(prompt_path),
            "uncond_embedding_path": str(uncond_path),
            "source_image": source,
            "target_image": target,
            "prompt_embeds": prompt_embeds,
            "uncond_prompt_embeds": uncond_embeds,
        }


def _pad_embeddings(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("Cannot pad an empty embedding list")
    max_len = max(int(t.shape[0]) for t in tensors)
    dim = int(tensors[0].shape[1])
    for idx, tensor in enumerate(tensors):
        if tensor.dim() != 2 or int(tensor.shape[1]) != dim:
            raise ValueError(f"Embedding {idx} shape mismatch: got {tuple(tensor.shape)}, expected dim={dim}")

    padded = torch.stack(
        [
            torch.cat([t, t.new_zeros(max_len - t.shape[0], t.shape[1])], dim=0)
            for t in tensors
        ],
        dim=0,
    )
    mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(t.shape[0], dtype=torch.long),
                    torch.zeros(max_len - t.shape[0], dtype=torch.long),
                ],
                dim=0,
            )
            for t in tensors
        ],
        dim=0,
    )
    return padded, mask


def collate_local_firered_edit(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    prompt_embeds, prompt_mask = _pad_embeddings([item["prompt_embeds"] for item in batch])
    uncond_embeds, uncond_mask = _pad_embeddings([item["uncond_prompt_embeds"] for item in batch])
    return {
        "uid": [item["uid"] for item in batch],
        "text": [item["text"] for item in batch],
        "jsonl_path": [item["jsonl_path"] for item in batch],
        "jsonl_lineno": [item["jsonl_lineno"] for item in batch],
        "source_image_path": [item["source_image_path"] for item in batch],
        "target_image_path": [item["target_image_path"] for item in batch],
        "embedding_path": [item["embedding_path"] for item in batch],
        "uncond_embedding_path": [item["uncond_embedding_path"] for item in batch],
        "source_image": torch.stack([item["source_image"] for item in batch], dim=0),
        "target_image": torch.stack([item["target_image"] for item in batch], dim=0),
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_mask,
        "uncond_prompt_embeds": uncond_embeds,
        "uncond_prompt_attention_mask": uncond_mask,
    }
