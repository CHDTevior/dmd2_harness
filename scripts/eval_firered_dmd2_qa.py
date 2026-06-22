#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("FIRERED_DISABLE_FLASH_ATTN", "1")

try:
    import diffusers.utils.import_utils as _diffusers_import_utils

    _diffusers_import_utils._flash_attn_available = False
    _diffusers_import_utils._flash_attn_version = None
    _diffusers_import_utils._flash_attn_3_available = False
    _diffusers_import_utils._flash_attn_3_version = None
except Exception:
    pass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.amp import autocast as torch_autocast
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TWINFLOW_SRC = Path("/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TWINFLOW_SRC) not in sys.path:
    sys.path.insert(0, str(TWINFLOW_SRC))

from dmd2_firered.local_firered_data import (  # noqa: E402
    LocalFireRedEditDataset,
    collate_local_firered_edit,
)
from networks import MODELS  # noqa: E402


TEACHER_ADAPTER = "teacher_gray"
STUDENT_ADAPTER = "student"
QA_LABELS = [
    "input",
    "orig_lora_vanilla_40",
    "orig_lora_few_1nfe",
    "dmd2_1nfe",
    "dmd2_2nfe",
    "dmd2_4nfe",
    "target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA eval for FireRed DMD2 LoRA checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to global_step_* checkpoint dir")
    parser.add_argument("--output-dir", default="", help="Defaults to <run_dir>/offline_eval_qa/<checkpoint>")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--thumb-size", type=int, default=256)
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return cfg


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")


def as_path(path: str, label: str, must_dir: bool = False, must_file: bool = False) -> Path:
    value = Path(path).expanduser()
    if not value.exists():
        raise FileNotFoundError(f"{label} missing: {value}")
    if must_dir and not value.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {value}")
    if must_file and not value.is_file():
        raise FileNotFoundError(f"{label} must be a file: {value}")
    return value


def adapter_file(path: Path) -> Path:
    if path.is_dir():
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No adapter_model.safetensors/bin found in {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Adapter path missing: {path}")


def load_adapter_into(peft_model, adapter_path: Path, adapter_name: str) -> None:
    path = adapter_file(adapter_path)
    if path.suffix == ".safetensors":
        state_dict = load_file(str(path))
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
    result = set_peft_model_state_dict(peft_model, state_dict, adapter_name=adapter_name)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading adapter={adapter_name} from {path}: {unexpected[:20]}"
        )


def get_inner_peft_model(wrapped_model):
    module = wrapped_model.transformer.module if hasattr(wrapped_model.transformer, "module") else wrapped_model.transformer
    return module.transformer


def configure_eval_adapters(wrapped_model, model_cfg: Dict, checkpoint: Path):
    target_modules = [item.strip() for item in str(model_cfg["lora_target_modules"]).split(",") if item.strip()]
    if not target_modules:
        raise ValueError("model.lora_target_modules is empty")
    lora_config = LoraConfig(
        r=int(model_cfg["student_lora_rank"]),
        lora_alpha=int(model_cfg["student_lora_alpha"]),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        init_lora_weights=str(model_cfg.get("init_lora_weights", "gaussian")),
        target_modules=target_modules,
        bias="none",
    )

    wrapped_model.transformer.requires_grad_(False)
    wrapped_model.transformer.transformer = get_peft_model(
        wrapped_model.transformer.transformer,
        lora_config,
        adapter_name=TEACHER_ADAPTER,
    )
    inner_peft = get_inner_peft_model(wrapped_model)
    inner_peft.add_adapter(STUDENT_ADAPTER, lora_config)

    teacher_path = as_path(model_cfg["teacher_adapter_path"], "model.teacher_adapter_path", must_dir=True)
    student_path = checkpoint / "student_adapter" / STUDENT_ADAPTER
    as_path(str(student_path), "student adapter checkpoint", must_dir=True)
    load_adapter_into(inner_peft, teacher_path, TEACHER_ADAPTER)
    load_adapter_into(inner_peft, student_path, STUDENT_ADAPTER)

    wrapped_model.transformer.requires_grad_(False)
    wrapped_model.transformer.eval()
    return inner_peft


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    return TF.to_pil_image(tensor)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


def load_font(size: int = 14):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_contact_sheet(rows: List[List[Tuple[str, Image.Image]]], output_path: Path, thumb_size: int) -> None:
    if not rows:
        raise RuntimeError("No rows to render")
    font = load_font(14)
    caption_h = 24
    cell_w = int(thumb_size)
    cell_h = int(thumb_size) + caption_h
    ncols = len(QA_LABELS)
    canvas = Image.new("RGB", (ncols * cell_w, len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(f"Row {row_idx} has {len(row)} columns, expected {ncols}")
        for col_idx, (label, image) in enumerate(row):
            x0 = col_idx * cell_w
            y0 = row_idx * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(0, 0, 0), width=1)
            draw.text((x0 + 3, y0 + 4), label, fill=(0, 0, 0), font=font)
            thumb = image.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x0, y0 + caption_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (x.dim() - 1)))


def forward_adapter(wrapped_model, inner_peft, adapter_name: str, x_t, t, prompt_embeds, prompt_mask, source_latents):
    inner_peft.set_adapter(adapter_name)
    model_dtype = prompt_embeds.dtype
    return wrapped_model.transformer(
        x_t.to(dtype=model_dtype),
        t,
        [prompt_embeds, prompt_mask, source_latents.to(dtype=model_dtype)],
    )


@torch.no_grad()
def flow_sample_pixels(
    wrapped_model,
    inner_peft,
    adapter_name: str,
    noise: torch.Tensor,
    steps: int,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    source_latents: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    latents = noise.clone().to(dtype=dtype)
    batch_size = int(latents.shape[0])
    for idx in range(steps):
        t_curr = 1.0 - float(idx) / float(steps)
        t_next = 1.0 - float(idx + 1) / float(steps)
        t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
        with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
            velocity = forward_adapter(
                wrapped_model,
                inner_peft,
                adapter_name,
                latents,
                t,
                prompt_embeds,
                prompt_mask,
                source_latents,
            )
        latents = (latents + (t_next - t_curr) * velocity).to(dtype=dtype)
    with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
        return wrapped_model.latents_to_pixels(latents).detach()


@torch.no_grad()
def vanilla_teacher_pixels(wrapped_model, inner_peft, batch: Dict, device: torch.device, seed: int) -> torch.Tensor:
    pipe = wrapped_model.model
    if getattr(pipe, "text_encoder", None) is None:
        raise RuntimeError("Vanilla eval requires text_encoder")
    inner_peft.set_adapter(TEACHER_ADAPTER)

    source_pil = tensor_to_pil(batch["source_image"][0])
    prompt = str(batch["text"][0])

    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as _qpp_mod

    original_transformer = pipe.transformer
    original_vae_image_size = _qpp_mod.VAE_IMAGE_SIZE
    pipe.transformer = inner_peft
    _qpp_mod.VAE_IMAGE_SIZE = source_pil.width * source_pil.height
    try:
        result = pipe(
            image=[source_pil],
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(seed),
            true_cfg_scale=0.0,
            negative_prompt=" ",
            num_inference_steps=40,
            num_images_per_prompt=1,
            height=source_pil.height,
            width=source_pil.width,
        )
    finally:
        _qpp_mod.VAE_IMAGE_SIZE = original_vae_image_size
        pipe.transformer = original_transformer
    return pil_to_tensor(result.images[0]).unsqueeze(0).to(device=device)


def make_dataset(cfg: Dict) -> LocalFireRedEditDataset:
    data_cfg = cfg["data"]
    return LocalFireRedEditDataset(
        jsonl_path=data_cfg["jsonl"],
        local_data_root=data_cfg["local_data_root"],
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        max_samples=data_cfg.get("max_samples"),
        source_image_field=data_cfg["source_image_field"],
        target_image_field=data_cfg["target_image_field"],
        instruction_field=data_cfg["instruction_field"],
        embedding_field=data_cfg["embedding_field"],
        uncond_embedding_field=data_cfg["uncond_embedding_field"],
    )


def main() -> None:
    args = parse_args()
    cfg_path = as_path(args.config, "config", must_file=True)
    checkpoint = as_path(args.checkpoint, "checkpoint", must_dir=True)
    cfg = load_config(cfg_path)
    if float(cfg["dmd2"].get("cfg_scale", -1)) != 0.0:
        raise ValueError(f"dmd2.cfg_scale must be 0, got {cfg['dmd2'].get('cfg_scale')}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    model_cfg = cfg["model"]
    as_path(model_cfg["base_model_path"], "model.base_model_path", must_dir=True)
    dataset = make_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_local_firered_edit,
    )

    wrapped_model = MODELS[model_cfg.get("type", "QwenImageEdit")](
        model_cfg["base_model_path"],
        aux_time_embed=bool(model_cfg.get("aux_time_embed", False)),
        text_dtype=dtype,
        imgs_dtype=dtype,
        device=str(device),
    )
    wrapped_model.transformer.to(device)
    wrapped_model.transformer.eval()
    inner_peft = configure_eval_adapters(wrapped_model, model_cfg, checkpoint)

    if args.output_dir:
        eval_dir = Path(args.output_dir).expanduser() / checkpoint.name
    else:
        eval_dir = checkpoint.parents[1] / "offline_eval_qa" / checkpoint.name
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: List[List[Tuple[str, Image.Image]]] = []
    entries = []
    produced = 0
    for batch in loader:
        if produced >= int(args.max_samples):
            break
        source = batch["source_image"].to(device=device)
        target = batch["target_image"].to(device=device)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
        prompt_mask = batch["prompt_attention_mask"].to(device=device)
        sample_seed = int(args.seed) + produced
        with torch.no_grad():
            source_latents = wrapped_model.pixels_to_latents(source).to(device=device, dtype=dtype)
        noise = torch.randn(
            target.shape[0],
            wrapped_model.transformer.in_channels,
            target.shape[-2] // wrapped_model.model.vae_scale_factor,
            target.shape[-1] // wrapped_model.model.vae_scale_factor,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(sample_seed),
        )

        outputs = {}
        vanilla = vanilla_teacher_pixels(wrapped_model, inner_peft, batch, device, sample_seed)
        outputs["orig_lora_vanilla_40"] = vanilla
        outputs["orig_lora_few_1nfe"] = flow_sample_pixels(
            wrapped_model, inner_peft, TEACHER_ADAPTER, noise, 1, prompt_embeds, prompt_mask, source_latents, dtype
        )
        outputs["dmd2_1nfe"] = flow_sample_pixels(
            wrapped_model, inner_peft, STUDENT_ADAPTER, noise, 1, prompt_embeds, prompt_mask, source_latents, dtype
        )
        outputs["dmd2_2nfe"] = flow_sample_pixels(
            wrapped_model, inner_peft, STUDENT_ADAPTER, noise, 2, prompt_embeds, prompt_mask, source_latents, dtype
        )
        outputs["dmd2_4nfe"] = flow_sample_pixels(
            wrapped_model, inner_peft, STUDENT_ADAPTER, noise, 4, prompt_embeds, prompt_mask, source_latents, dtype
        )

        uid = str(batch["uid"][0])
        sample_name = f"{produced:04d}_{uid[:32].replace('/', '_')}"
        row: List[Tuple[str, Image.Image]] = [("input", tensor_to_pil(source[0]))]
        generated_paths = {}
        for label in QA_LABELS[1:-1]:
            image = tensor_to_pil(outputs[label][0])
            out_path = eval_dir / f"{sample_name}_{label}.png"
            image.save(out_path)
            generated_paths[label] = str(out_path)
            row.append((label, image))
        row.append(("target", tensor_to_pil(target[0])))
        rows.append(row)
        entries.append(
            {
                "index": produced,
                "uid": uid,
                "seed": sample_seed,
                "source_image": str(batch["source_image_path"][0]),
                "target_image": str(batch["target_image_path"][0]),
                "generated": generated_paths,
            }
        )
        produced += 1

    if produced == 0:
        raise RuntimeError("QA eval produced no samples")

    contact_sheet = eval_dir / "contact_sheet.png"
    make_contact_sheet(rows, contact_sheet, int(args.thumb_size))
    manifest = {
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "contact_sheet": str(contact_sheet),
        "qa_labels": QA_LABELS,
        "cfg_scale": 0.0,
        "num_samples": produced,
        "entries": entries,
    }
    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"contact_sheet": str(contact_sheet), "manifest": str(eval_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
