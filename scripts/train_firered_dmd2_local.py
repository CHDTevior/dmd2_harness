#!/usr/bin/env python
import argparse
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.amp import autocast as torch_autocast
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

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


STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher_gray"
FAKE_ADAPTER = "fake_critic"


def as_path(value: str, label: str, must_dir: bool | None = None, must_file: bool | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if must_dir is True and not path.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {path}")
    if must_file is True and not path.is_file():
        raise FileNotFoundError(f"{label} must be a file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FireRed/QwenImageEdit DMD2 dryrun")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--fake-updates-per-step", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cuda_flags() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def dtype_from_precision(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision={name}; expected bf16/fp16/fp32")


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


def lora_param_matches(name: str, adapter_name: str) -> bool:
    return "lora_" in name and f".{adapter_name}." in name


def collect_adapter_params(module: nn.Module, adapter_name: str) -> List[Tuple[str, nn.Parameter]]:
    params = [(name, param) for name, param in module.named_parameters() if lora_param_matches(name, adapter_name)]
    if not params:
        samples = [name for name, _ in list(module.named_parameters())[:30]]
        raise RuntimeError(f"No LoRA params found for adapter={adapter_name}; sample params={samples}")
    return params


def configure_adapters(wrapped_model, model_cfg: Dict) -> tuple[nn.Module, List[nn.Parameter], List[nn.Parameter]]:
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
        adapter_name=STUDENT_ADAPTER,
    )
    inner_peft = get_inner_peft_model(wrapped_model)
    inner_peft.add_adapter(TEACHER_ADAPTER, lora_config)
    inner_peft.add_adapter(FAKE_ADAPTER, lora_config)

    teacher_path = as_path(model_cfg["teacher_adapter_path"], "model.teacher_adapter_path", must_dir=True)
    load_adapter_into(inner_peft, teacher_path, TEACHER_ADAPTER)
    load_adapter_into(inner_peft, teacher_path, STUDENT_ADAPTER)
    load_adapter_into(inner_peft, teacher_path, FAKE_ADAPTER)

    for name, param in wrapped_model.transformer.named_parameters():
        param.requires_grad = lora_param_matches(name, STUDENT_ADAPTER) or lora_param_matches(name, FAKE_ADAPTER)
        if "lora_" in name:
            param.data = param.data.to(torch.float32)

    student_named = collect_adapter_params(wrapped_model.transformer, STUDENT_ADAPTER)
    fake_named = collect_adapter_params(wrapped_model.transformer, FAKE_ADAPTER)
    teacher_named = collect_adapter_params(wrapped_model.transformer, TEACHER_ADAPTER)
    for _, param in teacher_named:
        param.requires_grad = False

    student_params = [param for _, param in student_named]
    fake_params = [param for _, param in fake_named]
    print(
        json.dumps(
            {
                "event": "lora_configured",
                "student_trainable_m": sum(p.numel() for p in student_params) / 1e6,
                "fake_critic_trainable_m": sum(p.numel() for p in fake_params) / 1e6,
                "teacher_params_m": sum(p.numel() for _, p in teacher_named) / 1e6,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return inner_peft, student_params, fake_params


class LatentRealismHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(32, channels * 4)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        pooled = latents.float().mean(dim=(2, 3))
        return self.net(pooled).squeeze(-1)


def make_optimizer(params: Iterable[nn.Parameter], lr: float, fused: bool) -> torch.optim.Optimizer:
    params = list(params)
    if not params:
        raise ValueError("Optimizer received no parameters")
    kwargs = {"lr": float(lr), "betas": (0.9, 0.999), "weight_decay": 0.0}
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def resolve_steps(cfg: Dict, override_steps: int | None) -> int:
    if override_steps is not None:
        return int(override_steps)
    train_cfg = cfg["train"]
    return int(train_cfg.get("dryrun_steps") or 1)


def forward_adapter(wrapped_model, inner_peft, adapter_name: str, x_t, t, prompt_embeds, prompt_mask, source_latents):
    inner_peft.set_adapter(adapter_name)
    model_dtype = prompt_embeds.dtype
    return wrapped_model.transformer(
        x_t.to(dtype=model_dtype),
        t,
        [prompt_embeds, prompt_mask, source_latents.to(dtype=model_dtype)],
    )


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (x.dim() - 1)))


def sample_t(batch_size: int, device: torch.device, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    return torch.rand(batch_size, device=device).mul(high - low).add(low)


def flow_x0(x_t: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return x_t - expand_time(t, x_t) * velocity


@contextmanager
def head_grad(head: nn.Module, enabled: bool):
    old = [param.requires_grad for param in head.parameters()]
    for param in head.parameters():
        param.requires_grad = enabled
    try:
        yield
    finally:
        for param, value in zip(head.parameters(), old):
            param.requires_grad = value


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} is not finite: {tensor.detach().float().cpu()}")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    return TF.to_pil_image(tensor)


def make_contact_sheet(rows: List[List[Image.Image]], labels: List[str], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows for contact sheet")
    thumb_w, thumb_h = rows[0][0].size
    label_h = 18
    width = len(labels) * thumb_w
    height = len(rows) * (thumb_h + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        if len(row) != len(labels):
            raise ValueError(f"Row {row_idx} has {len(row)} columns, expected {len(labels)}")
        y0 = row_idx * (thumb_h + label_h)
        for col_idx, image in enumerate(row):
            x0 = col_idx * thumb_w
            draw.text((x0 + 2, y0 + 2), labels[col_idx], fill=(0, 0, 0))
            canvas.paste(image, (x0, y0 + label_h))
            draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + thumb_h + label_h - 1], outline=(180, 180, 180))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@torch.no_grad()
def run_eval_contact_sheet(
    wrapped_model,
    inner_peft,
    loader,
    device: torch.device,
    dtype: torch.dtype,
    out_path: Path,
    max_samples: int,
) -> None:
    was_training = wrapped_model.training
    wrapped_model.eval()
    rows: List[List[Image.Image]] = []
    for batch in loader:
        if len(rows) >= max_samples:
            break
        source = batch["source_image"].to(device=device)
        target = batch["target_image"].to(device=device)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
        prompt_mask = batch["prompt_attention_mask"].to(device=device)
        source_latents = wrapped_model.pixels_to_latents(source).to(device=device, dtype=dtype)
        noise = torch.randn(
            target.shape[0],
            wrapped_model.transformer.in_channels,
            target.shape[-2] // wrapped_model.model.vae_scale_factor,
            target.shape[-1] // wrapped_model.model.vae_scale_factor,
            device=device,
            dtype=dtype,
        )
        t = torch.ones(target.shape[0], device=device, dtype=torch.float32)
        with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
            teacher_v = forward_adapter(wrapped_model, inner_peft, TEACHER_ADAPTER, noise, t, prompt_embeds, prompt_mask, source_latents)
            student_v = forward_adapter(wrapped_model, inner_peft, STUDENT_ADAPTER, noise, t, prompt_embeds, prompt_mask, source_latents)
            teacher_x = flow_x0(noise, t, teacher_v).to(dtype=dtype)
            student_x = flow_x0(noise, t, student_v).to(dtype=dtype)
            teacher_img = wrapped_model.latents_to_pixels(teacher_x).detach()
            student_img = wrapped_model.latents_to_pixels(student_x).detach()
        for i in range(target.shape[0]):
            rows.append(
                [
                    tensor_to_pil(source[i]).resize((256, 256)),
                    tensor_to_pil(teacher_img[i]).resize((256, 256)),
                    tensor_to_pil(student_img[i]).resize((256, 256)),
                    tensor_to_pil(target[i]).resize((256, 256)),
                ]
            )
            if len(rows) >= max_samples:
                break
    make_contact_sheet(rows, ["input", "teacher_1nfe", "student_1nfe", "target"], out_path)
    if was_training:
        wrapped_model.train()


def save_checkpoint(inner_peft, head: nn.Module, cfg: Dict, run_dir: Path, step: int) -> Path:
    ckpt_dir = run_dir / "checkpoints" / f"global_step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    inner_peft.save_pretrained(ckpt_dir / "student_adapter", selected_adapters=[STUDENT_ADAPTER])
    inner_peft.save_pretrained(ckpt_dir / "fake_critic_adapter", selected_adapters=[FAKE_ADAPTER])
    torch.save({"state_dict": head.state_dict(), "config": {"channels": head.net[0].in_features}}, ckpt_dir / "latent_realism_head.pt")
    (ckpt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "step": step,
                "student_adapter": STUDENT_ADAPTER,
                "fake_critic_adapter": FAKE_ADAPTER,
                "teacher_adapter": TEACHER_ADAPTER,
                "config_project": cfg["project"]["name"],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ckpt_dir


def main() -> None:
    args = parse_args()
    cfg_path = as_path(args.config, "config", must_file=True)
    cfg = load_config(cfg_path)
    if float(cfg["dmd2"].get("cfg_scale", -1)) != 0.0:
        raise ValueError(f"dmd2.cfg_scale must be 0, got {cfg['dmd2'].get('cfg_scale')}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    set_cuda_flags()
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device(args.device)
    dtype = dtype_from_precision(str(cfg["train"].get("precision", "bf16")))
    steps = resolve_steps(cfg, args.steps)
    output_root = Path(cfg["project"]["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("local_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data_cfg = cfg["data"]
    dataset = LocalFireRedEditDataset(
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
    if len(dataset) < 1:
        raise RuntimeError("Dataset is empty")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("micro_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True,
        collate_fn=collate_local_firered_edit,
        drop_last=True,
    )
    if len(loader) < 1:
        raise RuntimeError("DataLoader produced zero batches")

    print(
        json.dumps(
            {
                "event": "startup",
                "config": str(cfg_path),
                "run_dir": str(run_dir),
                "dataset_size": len(dataset),
                "steps": steps,
                "device": str(device),
                "dtype": str(dtype),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "fused_adamw": bool(cfg["train"].get("fused_adamw", True)),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    model_cfg = cfg["model"]
    as_path(model_cfg["base_model_path"], "model.base_model_path", must_dir=True)
    wrapped_model = MODELS[model_cfg.get("type", "QwenImageEdit")](
        model_cfg["base_model_path"],
        aux_time_embed=bool(model_cfg.get("aux_time_embed", False)),
        text_dtype=dtype,
        imgs_dtype=dtype,
        device=str(device),
    )
    text_encoder = getattr(wrapped_model.model, "text_encoder", None)
    if text_encoder is not None:
        wrapped_model.model.text_encoder = None
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
    wrapped_model.transformer.to(device)
    wrapped_model.transformer.train()
    if bool(cfg["train"].get("gradient_checkpointing", True)):
        wrapped_model.transformer.enable_gradient_checkpointing()

    inner_peft, student_params, fake_params = configure_adapters(wrapped_model, model_cfg)
    student_opt = make_optimizer(
        student_params,
        lr=float(cfg["train"]["learning_rate_student"]),
        fused=bool(cfg["train"].get("fused_adamw", True)),
    )

    first_batch = next(iter(loader))
    with torch.no_grad():
        first_target = first_batch["target_image"].to(device=device)
        first_latents = wrapped_model.pixels_to_latents(first_target)
    head = LatentRealismHead(channels=int(first_latents.shape[1])).to(device=device)
    fake_opt = make_optimizer(
        list(fake_params) + list(head.parameters()),
        lr=float(cfg["train"]["learning_rate_fake_critic"]),
        fused=bool(cfg["train"].get("fused_adamw", True)),
    )
    del first_batch, first_target, first_latents
    torch.cuda.empty_cache()

    log_path = run_dir / "train_log.jsonl"
    data_iter = iter(loader)
    fake_updates = args.fake_updates_per_step
    if fake_updates is None:
        fake_updates = int(cfg["dmd2"].get("dfake_gen_update_ratio", 1))
    fake_updates = max(1, fake_updates)

    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    teacher_match_weight = float(cfg["dmd2"].get("teacher_match_loss_weight", 0.1))
    dm_weight = float(cfg["dmd2"].get("dm_loss_weight", 1.0))
    gen_cls_weight = float(cfg["dmd2"].get("gen_cls_loss_weight", 0.0))
    guidance_cls_weight = float(cfg["dmd2"].get("guidance_cls_loss_weight", 0.0))
    use_classifier = bool(cfg["dmd2"].get("use_gan_classifier", True))

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        source = batch["source_image"].to(device=device, non_blocking=True)
        target = batch["target_image"].to(device=device, non_blocking=True)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        prompt_mask = batch["prompt_attention_mask"].to(device=device, non_blocking=True)

        with torch.no_grad():
            target_latents = wrapped_model.pixels_to_latents(target).to(device=device, dtype=dtype)
            source_latents = wrapped_model.pixels_to_latents(source).to(device=device, dtype=dtype)

        batch_size = int(target_latents.shape[0])

        student_opt.zero_grad(set_to_none=True)
        fake_opt.zero_grad(set_to_none=True)
        t = sample_t(batch_size, device)
        noise = torch.randn_like(target_latents)
        x_t = expand_time(t, target_latents) * noise + (1.0 - expand_time(t, target_latents)) * target_latents
        with head_grad(head, enabled=False):
            with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                student_v = forward_adapter(wrapped_model, inner_peft, STUDENT_ADAPTER, x_t, t, prompt_embeds, prompt_mask, source_latents)
                student_x0 = flow_x0(x_t, t, student_v)
                with torch.no_grad():
                    teacher_v = forward_adapter(wrapped_model, inner_peft, TEACHER_ADAPTER, x_t, t, prompt_embeds, prompt_mask, source_latents)
                    t_dm = sample_t(batch_size, device)
                    dm_noise = torch.randn_like(student_x0)
                    dm_x_t = expand_time(t_dm, student_x0) * dm_noise + (1.0 - expand_time(t_dm, student_x0)) * student_x0.detach()
                    real_v = forward_adapter(wrapped_model, inner_peft, TEACHER_ADAPTER, dm_x_t, t_dm, prompt_embeds, prompt_mask, source_latents)
                    fake_v = forward_adapter(wrapped_model, inner_peft, FAKE_ADAPTER, dm_x_t, t_dm, prompt_embeds, prompt_mask, source_latents)
                    pred_real_x0 = flow_x0(dm_x_t, t_dm, real_v)
                    pred_fake_x0 = flow_x0(dm_x_t, t_dm, fake_v)
                    dm_grad = (pred_fake_x0 - pred_real_x0).detach()
                    dm_norm = dm_grad.abs().mean().clamp_min(1e-6)
                    dm_grad = torch.nan_to_num(dm_grad / dm_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
                    dm_target = (student_x0 - dm_grad).detach()
                loss_teacher = F.mse_loss(student_v.float(), teacher_v.float())
                loss_dm = 0.5 * F.mse_loss(student_x0.float(), dm_target.float())
                if use_classifier and gen_cls_weight > 0:
                    gen_cls_loss = F.softplus(-head(student_x0).float()).mean()
                else:
                    gen_cls_loss = student_x0.float().new_zeros(())
                loss_student = teacher_match_weight * loss_teacher + dm_weight * loss_dm + gen_cls_weight * gen_cls_loss
        assert_finite("loss_student", loss_student)
        loss_student.backward()
        torch.nn.utils.clip_grad_norm_(student_params, max_grad_norm)
        student_opt.step()

        last_loss_fake = None
        last_loss_cls = None
        for _ in range(fake_updates):
            try:
                fake_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                fake_batch = next(data_iter)
            source_f = fake_batch["source_image"].to(device=device, non_blocking=True)
            target_f = fake_batch["target_image"].to(device=device, non_blocking=True)
            prompt_embeds_f = fake_batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
            prompt_mask_f = fake_batch["prompt_attention_mask"].to(device=device, non_blocking=True)
            with torch.no_grad():
                target_f_latents = wrapped_model.pixels_to_latents(target_f).to(device=device, dtype=dtype)
                source_f_latents = wrapped_model.pixels_to_latents(source_f).to(device=device, dtype=dtype)
                bsz_f = int(target_f_latents.shape[0])
                t_f = sample_t(bsz_f, device)
                noise_f = torch.randn_like(target_f_latents)
                x_t_f = expand_time(t_f, target_f_latents) * noise_f + (1.0 - expand_time(t_f, target_f_latents)) * target_f_latents
                with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                    student_v_f = forward_adapter(wrapped_model, inner_peft, STUDENT_ADAPTER, x_t_f, t_f, prompt_embeds_f, prompt_mask_f, source_f_latents)
                generated_latents = flow_x0(x_t_f, t_f, student_v_f).detach()

            fake_opt.zero_grad(set_to_none=True)
            with head_grad(head, enabled=True):
                t_g = sample_t(int(generated_latents.shape[0]), device)
                noise_g = torch.randn_like(generated_latents)
                fake_x_t = expand_time(t_g, generated_latents) * noise_g + (1.0 - expand_time(t_g, generated_latents)) * generated_latents
                fake_target_v = noise_g - generated_latents
                with torch_autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                    fake_pred_v = forward_adapter(wrapped_model, inner_peft, FAKE_ADAPTER, fake_x_t, t_g, prompt_embeds_f, prompt_mask_f, source_f_latents)
                    loss_fake = F.mse_loss(fake_pred_v.float(), fake_target_v.float())
                    if use_classifier and guidance_cls_weight > 0:
                        real_logits = head(target_f_latents)
                        fake_logits = head(generated_latents.detach())
                        loss_cls = F.softplus(-real_logits.float()).mean() + F.softplus(fake_logits.float()).mean()
                    else:
                        loss_cls = generated_latents.float().new_zeros(())
                    loss_fake_total = loss_fake + guidance_cls_weight * loss_cls
            assert_finite("loss_fake_total", loss_fake_total)
            loss_fake_total.backward()
            torch.nn.utils.clip_grad_norm_(list(fake_params) + list(head.parameters()), max_grad_norm)
            fake_opt.step()
            last_loss_fake = loss_fake.detach()
            last_loss_cls = loss_cls.detach()

        mem_gb = torch.cuda.memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
        record = {
            "step": step,
            "loss_student": float(loss_student.detach().float().cpu()),
            "loss_teacher": float(loss_teacher.detach().float().cpu()),
            "loss_dm": float(loss_dm.detach().float().cpu()),
            "loss_gen_cls": float(gen_cls_loss.detach().float().cpu()),
            "loss_fake": float(last_loss_fake.float().cpu()) if last_loss_fake is not None else math.nan,
            "loss_guidance_cls": float(last_loss_cls.float().cpu()) if last_loss_cls is not None else math.nan,
            "fake_updates": fake_updates,
            "memory_reserved_gib": mem_gb,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(json.dumps({"event": "train_step", **record}, ensure_ascii=True), flush=True)

    eval_samples = args.eval_samples if args.eval_samples is not None else int(cfg["eval"].get("max_samples", 4))
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_local_firered_edit,
    )
    eval_path = run_dir / "eval" / f"global_step_{steps:06d}" / "contact_sheet.png"
    run_eval_contact_sheet(wrapped_model, inner_peft, eval_loader, device, dtype, eval_path, max_samples=eval_samples)
    ckpt_dir = save_checkpoint(inner_peft, head, cfg, run_dir, steps)
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "eval_contact_sheet": str(eval_path),
        "steps": steps,
        "status": "ok",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **manifest}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
