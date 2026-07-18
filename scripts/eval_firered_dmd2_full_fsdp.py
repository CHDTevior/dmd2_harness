#!/usr/bin/env python
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path

if os.environ.get("FIRERED_DISABLE_FLASH_ATTN", "0") == "1":
    import diffusers.utils.import_utils as _diffusers_import_utils

    _diffusers_import_utils._flash_attn_available = False
    _diffusers_import_utils._flash_attn_version = None
    _diffusers_import_utils._flash_attn_3_available = False
    _diffusers_import_utils._flash_attn_3_version = None

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader

HARNESS_ROOT = Path(__file__).resolve().parents[1]
TWINFLOW_SRC = Path(
    os.environ.get("TWINFLOW_SRC", "/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")
).expanduser()
for path in (HARNESS_ROOT, TWINFLOW_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_firered_dmd2_full_fsdp import (  # noqa: E402
    DMD2FullOfficialMethod,
    DMD2FullSharedMethod,
    DMD2_RENOISE_SAMPLING_STYLE,
    SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE,
)
from data.firered_edit_jsonl_dataset import FireRedEditJsonlDataset, collate_firered_edit  # noqa: E402
from networks import MODELS  # noqa: E402
from steerers.qwenimage.sft_ddp_lora_firered_edit import (  # noqa: E402
    maybe_drop_text_encoder,
    optional_field,
    safe_filename,
    slice_batch,
    write_json_atomic,
)
from steerers.qwenimage.sft_fsdp_firered_edit import (  # noqa: E402
    cleanup_distributed,
    get_conditions_from_batch,
    get_fsdp_use_orig_params,
    get_in_channels,
    get_sampling_model,
    load_fsdp_model_dcp,
    load_eval_reference_map,
    make_labeled_contact_sheet,
    set_seed,
    setup_distributed,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval FireRed DMD2 full FSDP checkpoint only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to global_step_* checkpoint dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--eval-jsonl", default="", help="Optional eval jsonl override.")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-manifest", default="")
    parser.add_argument("--reference-label", action="append", default=[])
    parser.add_argument(
        "--sampling-style",
        default=DMD2_RENOISE_SAMPLING_STYLE,
        choices=[DMD2_RENOISE_SAMPLING_STYLE, SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE],
    )
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--cfg-scales",
        type=float,
        nargs="+",
        default=[0.0],
        help="Student CFG sweep. 0 means the existing conditional-only student path.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="Per-image denoising repeats per NFE after warm-up; 0 disables timing.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Unmeasured denoising warm-up passes per NFE before timing.",
    )
    parser.add_argument(
        "--benchmark-output-repeats",
        type=int,
        default=0,
        help="Per-image decoded-output repeats per NFE after warm-up; 0 disables timing.",
    )
    parser.add_argument(
        "--benchmark-output-warmup",
        type=int,
        default=1,
        help="Unmeasured decoded-output warm-up passes per NFE before timing.",
    )
    parser.add_argument(
        "--benchmark-vae-decode-repeats",
        type=int,
        default=0,
        help="Per-image VAE decode repeats after warm-up; 0 disables timing.",
    )
    parser.add_argument(
        "--benchmark-vae-decode-warmup",
        type=int,
        default=1,
        help="Unmeasured VAE decode warm-up passes before timing.",
    )
    parser.add_argument(
        "--benchmark-vae-decode-source-nfe",
        type=int,
        default=1,
        help="NFE used only to prepare representative latents for VAE decode timing.",
    )
    return parser.parse_args()


def exp_step(checkpoint: str) -> int:
    name = Path(checkpoint).name
    if not name.startswith("global_step_"):
        raise ValueError(f"Checkpoint path must end with global_step_*, got {checkpoint}")
    return int(name.split("_")[-1])


def raise_collective_checkpoint_error(*, local_error: str, device: torch.device, phase: str) -> None:
    valid = torch.tensor(0 if local_error else 1, device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if int(valid.item()) == 1:
        return

    errors = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(errors, local_error)
    else:
        errors[0] = local_error
    detail = "; ".join(error for error in errors if error) or "unknown rank-local failure"
    raise RuntimeError(f"Checkpoint evaluation {phase} failed on at least one rank: {detail}")


def load_student_checkpoint_for_eval(checkpoint_dir: str, wrapped_model, device: torch.device) -> dict:
    """Load the student weights required for evaluation, not training-resume state.

    Final DMD2 archives intentionally retain only ``model_dcp`` and provenance.
    Requiring optimizer or per-rank RNG files here would make those valid
    inference-only checkpoints unusable, while the training resume path remains
    responsible for validating its complete mutable state.
    """
    ckpt_dir = Path(checkpoint_dir)
    local_error = ""
    train_state = None
    model_dcp = ckpt_dir / "model_dcp"
    try:
        complete = ckpt_dir / ".save_complete"
        failed = ckpt_dir / ".save_failed"
        metadata_path = ckpt_dir / "checkpoint_meta.json"
        train_state_path = ckpt_dir / "train_state.json"
        if failed.is_file():
            raise RuntimeError(f"Checkpoint has failure marker: {failed.read_text().strip()}")
        if not complete.is_file():
            raise FileNotFoundError(f"Checkpoint is missing .save_complete: {complete}")
        if not model_dcp.is_dir():
            raise FileNotFoundError(f"Checkpoint is missing model_dcp directory: {model_dcp}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing checkpoint_meta.json: {metadata_path}")
        if not train_state_path.is_file():
            raise FileNotFoundError(
                "Checkpoint is missing train_state.json required for evaluation provenance: "
                f"{train_state_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("checkpoint_format"), str):
            raise ValueError(f"Checkpoint metadata is invalid: {metadata_path}")
        with train_state_path.open("r", encoding="utf-8") as handle:
            train_state = json.load(handle)
        if not isinstance(train_state, dict):
            raise ValueError(f"Checkpoint train_state must be a JSON object: {train_state_path}")
        if int(train_state.get("global_step", -1)) != exp_step(str(ckpt_dir)):
            raise ValueError(
                "Checkpoint train_state global_step does not match checkpoint directory: "
                f"{train_state_path}"
            )
    except Exception as exc:
        local_error = f"rank {dist.get_rank() if dist.is_initialized() else 0}: {type(exc).__name__}: {exc}"

    raise_collective_checkpoint_error(
        local_error=local_error,
        device=device,
        phase="preflight",
    )

    load_error = ""
    try:
        load_fsdp_model_dcp(wrapped_model.transformer, model_dcp)
    except Exception as exc:
        load_error = f"rank {dist.get_rank() if dist.is_initialized() else 0}: {type(exc).__name__}: {exc}"
    raise_collective_checkpoint_error(
        local_error=load_error,
        device=device,
        phase="model_dcp load",
    )
    return train_state


def build_fsdp_model(config: dict, local_rank: int, dtype: torch.dtype):
    wrapped_model = MODELS[config["model"]["model_name"]](
        model_id=config["model"]["model_path"],
        aux_time_embed=bool(config["model"].get("aux_time_embed", False)),
        text_dtype=dtype,
        imgs_dtype=dtype,
    )
    no_split_modules = [m for m in wrapped_model.model.transformer._no_split_modules]
    maybe_drop_text_encoder(wrapped_model, bool(config["model"].get("drop_text_encoder", True)))
    wrapped_model.transformer.requires_grad_(False)
    wrapped_model.transformer.eval()

    wrapped_model.transformer = FSDP(
        wrapped_model.transformer,
        device_id=local_rank,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda inner: inner.__class__.__name__ in no_split_modules,
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=get_fsdp_use_orig_params(config),
    )
    wrapped_model.transformer.eval()
    return wrapped_model


def make_eval_dataloader(config: dict, args: argparse.Namespace) -> DataLoader:
    eval_config = dict(config.get("eval", {}) or {})
    if args.eval_jsonl:
        eval_jsonl = [args.eval_jsonl]
    else:
        eval_jsonl_value = eval_config.get("eval_jsonl", config["data"]["train_jsonl"])
        eval_jsonl = [eval_jsonl_value] if isinstance(eval_jsonl_value, str) else list(eval_jsonl_value)
    dataset = FireRedEditJsonlDataset(
        jsonl_files=eval_jsonl,
        firered_project_root=str(config["data"]["firered_project_root"]),
        height=int(args.height),
        width=int(args.width),
        max_samples=int(args.max_samples),
        repeat=1,
        instruction_field=str(config["data"].get("instruction_field", "instruction")),
        embedding_field=optional_field(config["data"].get("embedding_field", "embeddings_tensor_en")),
        uncond_embedding_field=optional_field(config["data"].get("uncond_embedding_field", "embeddings_tensor_droptext")),
    )
    return DataLoader(
        dataset,
        num_workers=int(eval_config.get("num_workers", 0)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=int(args.batch_size),
        collate_fn=collate_firered_edit,
    )


def require_cached_positive_conditions(batch: dict, *, benchmark_name: str) -> None:
    """Keep text encoding outside timing and make the benchmark contract explicit."""
    missing = [
        key
        for key in ("prompt_embeds", "prompt_attention_mask")
        if batch.get(key) is None
    ]
    if missing:
        raise RuntimeError(
            f"{benchmark_name} requires cached positive conditions ({', '.join(missing)} missing); "
            "refusing to include text-encoder work in the reported latency"
        )


def validate_decoded_output_shape(
    *,
    pixels: torch.Tensor,
    expected_shape: tuple[int, ...],
    device: torch.device,
    benchmark_name: str,
) -> None:
    """Validate once outside timing, with a collective verdict for FSDP safety."""
    local_valid = int(tuple(pixels.shape) == expected_shape)
    globally_valid = torch.tensor(local_valid, device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(globally_valid, op=dist.ReduceOp.MIN)
    if int(globally_valid.item()) != 1:
        raise RuntimeError(
            f"{benchmark_name} decoded shape preflight failed on at least one rank; "
            f"local output={tuple(pixels.shape)}, expected={expected_shape}"
        )


@torch.no_grad()
def prepare_benchmark_inputs(
    *,
    wrapped_model,
    eval_dataloader: DataLoader,
    max_samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """Precompute all non-denoising inputs so timed regions contain sampling only."""
    model_fn = get_sampling_model(wrapped_model)
    prepared = []
    for batch_list in eval_dataloader:
        if len(prepared) >= max_samples:
            break
        batch = batch_list[0]
        if int(batch["image"].shape[0]) != 1:
            raise ValueError("Timing benchmark requires --batch-size=1 for per-image latency")
        require_cached_positive_conditions(batch, benchmark_name="Denoising/VAE timing")
        source = batch["source_image"].to(device=device)
        image = batch["image"].to(device=device)
        prompt_embeds, prompt_mask, _, _ = get_conditions_from_batch(
            wrapped_model,
            batch,
            source,
            device,
            need_uncond=False,
        )
        source_latents = wrapped_model.pixels_to_latents(source).to(dtype=dtype)
        noise_generator = torch.Generator(device=device).manual_seed(int(seed) + len(prepared))
        noise = torch.randn(
            [
                1,
                get_in_channels(model_fn),
                image.shape[-2] // wrapped_model.model.vae_scale_factor,
                image.shape[-1] // wrapped_model.model.vae_scale_factor,
            ],
            dtype=dtype,
            device=device,
            generator=noise_generator,
        )
        prepared.append(
            {
                "noise": noise,
                "prompt_embeds": prompt_embeds.to(dtype=dtype),
                "prompt_mask": prompt_mask,
                "source_latents": source_latents,
                "output_shape": tuple(source.shape),
            }
        )
    if len(prepared) != max_samples:
        raise RuntimeError(f"Expected {max_samples} timing samples, prepared {len(prepared)}")
    return prepared


@torch.no_grad()
def dmd2_renoise_sampling_loop_for_timing(
    *,
    method,
    noise: torch.Tensor,
    model_fn,
    sampling_steps: int,
    condition: list[torch.Tensor],
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Official DMD2 re-noise loop without per-call argument validation for timing only."""
    latents = noise
    trajectory = []
    batch_size = int(latents.shape[0])
    for idx in range(sampling_steps):
        t_curr = 1.0 - float(idx) / float(sampling_steps)
        t_next = 1.0 - float(idx + 1) / float(sampling_steps)
        timestep = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
        velocity = model_fn(latents, timestep, condition)
        x0 = method.flow_x0(latents, timestep, velocity)
        trajectory.append(x0)
        latents = method.renoise_x0(x0, t_next, generator=generator)
    return trajectory


@torch.no_grad()
def sample_latents_for_timing(
    *,
    method,
    model_fn,
    sample: dict,
    noise: torch.Tensor,
    generator: torch.Generator,
    sampling_steps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
        trajectory = dmd2_renoise_sampling_loop_for_timing(
            method=method,
            noise=noise,
            model_fn=model_fn,
            sampling_steps=int(sampling_steps),
            generator=generator,
            condition=[sample["prompt_embeds"], sample["prompt_mask"], sample["source_latents"]],
        )
    return trajectory[-1]


def synchronize_benchmark_ranks(device: torch.device) -> None:
    """Align ranks outside the timed region before an FSDP sampling call."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        except TypeError:
            dist.barrier()
    torch.cuda.synchronize(device)


def cfg_label(cfg_scale: float) -> str:
    value = f"{float(cfg_scale):.17g}".replace("-", "m").replace(".", "p")
    return f"cfg{value}"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_fingerprint(path: Path) -> str:
    """Hash whole small files and both ends of large files without rescanning model shards."""
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    sample_bytes = 1024 * 1024
    with path.open("rb") as file_obj:
        if stat.st_size <= 2 * sample_bytes:
            for chunk in iter(lambda: file_obj.read(sample_bytes), b""):
                hasher.update(chunk)
        else:
            hasher.update(file_obj.read(sample_bytes))
            file_obj.seek(-sample_bytes, os.SEEK_END)
            hasher.update(file_obj.read(sample_bytes))
    return hasher.hexdigest()


def path_fingerprint(path_value: str) -> str:
    """Produce a stable, content-sensitive identifier for a file or checkpoint tree."""
    path = Path(path_value)
    if path.is_file():
        return file_fingerprint(path)
    if not path.is_dir():
        return f"missing:{path}"
    entries = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append((str(file_path.relative_to(path)), file_fingerprint(file_path)))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def cfg_sweep_checkpoint_identity(config_path: str, checkpoint_path: str) -> dict[str, str]:
    """Identify the exact config and model checkpoint without a full shard rescan."""
    config = Path(config_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    identity = {
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_fingerprint": path_fingerprint(str(checkpoint)),
    }
    for name in ("train_state.json", "checkpoint_meta.json"):
        metadata_path = checkpoint / name
        identity[name] = file_sha256(metadata_path) if metadata_path.is_file() else ""
    return identity


def cfg_sweep_output_namespace(
    *,
    sampling_style: str,
    sampling_steps: list[int],
    cfg_scales: list[float],
    eval_config: dict,
) -> str:
    """Give every CFG sweep an invocation-specific directory below one checkpoint."""
    eval_jsonl = eval_config.get("eval_jsonl", [])
    if isinstance(eval_jsonl, str):
        eval_jsonl = [eval_jsonl]
    reference_manifest = str(eval_config.get("reference_manifest", "") or "")
    signature_payload = {
        "sampling_style": str(sampling_style),
        "sampling_steps": [int(step) for step in sampling_steps],
        "cfg_scales": [float(scale) for scale in cfg_scales],
        "height": int(eval_config.get("height", 0)),
        "width": int(eval_config.get("width", 0)),
        "max_samples": int(eval_config.get("max_samples", 0)),
        "batch_size": int(eval_config.get("batch_size", 0)),
        "seed": int(eval_config.get("seed", 0)),
        "eval_jsonl": [str(path) for path in eval_jsonl],
        "eval_jsonl_fingerprints": [path_fingerprint(str(path)) for path in eval_jsonl],
        "reference_manifest": reference_manifest,
        "reference_manifest_fingerprint": path_fingerprint(reference_manifest) if reference_manifest else "",
        "reference_labels": [str(label) for label in eval_config.get("reference_labels", [])],
        "checkpoint_identity": dict(eval_config.get("_cfg_sweep_checkpoint_identity", {}) or {}),
    }
    digest = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    step_text = "-".join(str(int(step)) for step in sampling_steps)
    cfg_text = "-".join(cfg_label(float(scale)) for scale in cfg_scales)
    return (
        f"cfg_sweep_{sampling_style}_steps-{step_text}_{cfg_text}_"
        f"h{signature_payload['height']}w{signature_payload['width']}_"
        f"seed{signature_payload['seed']}_{digest}"
    )


def preload_reference_images(
    *,
    batch: dict,
    batch_size: int,
    reference_by_uid: dict,
    reference_labels: list[str],
) -> dict[tuple[str, str], Image.Image]:
    """Read references before FSDP sampling so rank-zero image I/O cannot desynchronize ranks."""
    preloaded = {}
    for index in range(batch_size):
        uid = str(batch["uid"][index])
        reference_entry = reference_by_uid.get(uid)
        if reference_entry is None:
            raise KeyError(f"reference manifest has no entry for eval uid={uid}")
        for label in reference_labels:
            reference_path = reference_entry[label]
            try:
                with Image.open(reference_path) as reference_image:
                    preloaded[(uid, label)] = reference_image.convert("RGB").copy()
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"reference image is unreadable for uid={uid} label={label}: {reference_path}"
                ) from exc
    return preloaded


def make_student_cfg_model_fn(
    *,
    model_fn,
    uncond_embeds: torch.Tensor | None,
    uncond_mask: torch.Tensor | None,
    cfg_scale: float,
    dtype: torch.dtype,
):
    """Apply the same uncond + scale * (cond - uncond) rule as the DMD2 teacher."""
    if cfg_scale in {0.0, 1.0}:
        return model_fn
    if cfg_scale < 0.0:
        raise ValueError(f"cfg scale must be non-negative, got {cfg_scale}")
    if uncond_embeds is None or uncond_mask is None:
        raise RuntimeError("Student CFG requires unconditional prompt embeddings")

    def guided_model_fn(x_t: torch.Tensor, t: torch.Tensor, condition):
        cond_v = model_fn(x_t.to(dtype=dtype), t, condition)
        uncond_v = model_fn(
            x_t.to(dtype=dtype),
            t,
            [uncond_embeds, uncond_mask, condition[2]],
        )
        return uncond_v + cfg_scale * (cond_v - uncond_v)

    return guided_model_fn


@torch.no_grad()
def pixels_from_batch_cfg_sweep(
    *,
    wrapped_model,
    method,
    batch: dict,
    sampling_style: str,
    sampling_steps: list[int],
    cfg_scales: list[float],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[str, int, float, torch.Tensor]]:
    """Generate fixed-noise NFE/CFG variants for a single evaluation batch."""
    if any(scale < 0.0 for scale in cfg_scales):
        raise ValueError(f"cfg scales must be non-negative, got {cfg_scales}")
    image = batch["image"].to(device=device)
    source = batch["source_image"].to(device=device)
    prompt_embeds, prompt_mask, _, _ = get_conditions_from_batch(
        wrapped_model,
        batch,
        source,
        device,
        need_uncond=False,
    )
    uncond_embeds = None
    uncond_mask = None
    if any(scale not in {0.0, 1.0} for scale in cfg_scales):
        _, _, uncond_embeds, uncond_mask = get_conditions_from_batch(
            wrapped_model,
            batch,
            source,
            device,
            need_uncond=True,
        )
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    if uncond_embeds is not None:
        uncond_embeds = uncond_embeds.to(dtype=dtype)
    source_latents = wrapped_model.pixels_to_latents(source).to(dtype=dtype)
    model_fn = get_sampling_model(wrapped_model)
    condition = [prompt_embeds, prompt_mask, source_latents]
    variants = []
    for steps in sampling_steps:
        for cfg_scale in cfg_scales:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            noise = torch.randn(
                [
                    image.shape[0],
                    get_in_channels(model_fn),
                    image.shape[-2] // wrapped_model.model.vae_scale_factor,
                    image.shape[-1] // wrapped_model.model.vae_scale_factor,
                ],
                dtype=dtype,
                device=device,
                generator=generator,
            )
            student_model_fn = make_student_cfg_model_fn(
                model_fn=model_fn,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
                cfg_scale=float(cfg_scale),
                dtype=dtype,
            )
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                samples = method.sampling_loop(
                    noise,
                    student_model_fn,
                    sampling_steps=int(steps),
                    stochast_ratio=1.0,
                    extrapol_ratio=0.0,
                    sampling_order=1,
                    time_dist_ctrl=[1.0, 1.0, 1.0],
                    rfba_gap_steps=[0.001, 0.7],
                    sampling_style=str(sampling_style),
                    generator=generator,
                    c=condition,
                )
            label = f"{sampling_style}_{int(steps)}nfe_{cfg_label(float(cfg_scale))}"
            variants.append((label, int(steps), float(cfg_scale), wrapped_model.latents_to_pixels(samples[-1])))
    return variants


@torch.no_grad()
def run_offline_eval_cfg_sweep(
    *,
    wrapped_model,
    method,
    eval_dataloader: DataLoader,
    output_dir: str,
    global_step: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    eval_config: dict,
    sampling_style: str,
    sampling_steps: list[int],
    cfg_scales: list[float],
    seed: int,
) -> str | None:
    """Offline evaluation for explicit student CFG variants; CFG=0 stays conditional-only."""
    if not cfg_scales:
        raise ValueError("cfg sweep requires at least one cfg scale")
    model_fn = get_sampling_model(wrapped_model)
    was_training = bool(model_fn.training)
    model_fn.eval()
    eval_dir = (
        Path(output_dir)
        / "offline_eval"
        / f"step_{global_step:08d}"
        / cfg_sweep_output_namespace(
            sampling_style=sampling_style,
            sampling_steps=sampling_steps,
            cfg_scales=cfg_scales,
            eval_config=eval_config,
        )
    )
    if rank == 0:
        eval_dir.mkdir(parents=True, exist_ok=True)
    # Every rank validates the manifest and upcoming UIDs before any FSDP forward.
    # A rank-0-only failure here would otherwise leave the other ranks in a collective.
    reference_by_uid, reference_labels = load_eval_reference_map(eval_config)

    rows = []
    entries = []
    produced = 0
    max_samples = int(eval_config.get("max_samples", 0) or 0)
    try:
        for batch_list in eval_dataloader:
            batch = batch_list[0]
            batch_size = int(batch["image"].shape[0])
            if max_samples > 0:
                remaining = max_samples - produced
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)
                batch = slice_batch(batch, batch_size)
            if reference_labels:
                missing_uids = [
                    str(batch["uid"][index])
                    for index in range(batch_size)
                    if str(batch["uid"][index]) not in reference_by_uid
                ]
                if missing_uids:
                    raise KeyError(f"reference manifest has no entry for eval uids={missing_uids}")
            preloaded_reference_images = {}
            if reference_labels:
                preloaded_reference_images = preload_reference_images(
                    batch=batch,
                    batch_size=batch_size,
                    reference_by_uid=reference_by_uid,
                    reference_labels=reference_labels,
                )
            predictions = pixels_from_batch_cfg_sweep(
                wrapped_model=wrapped_model,
                method=method,
                batch=batch,
                sampling_style=sampling_style,
                sampling_steps=sampling_steps,
                cfg_scales=cfg_scales,
                seed=int(seed) + produced,
                device=device,
                dtype=dtype,
            )
            if rank == 0:
                for index in range(batch_size):
                    uid = str(batch["uid"][index])
                    sample_name = f"{produced:04d}_{safe_filename(uid)}"
                    generated = {}
                    row = [("input", tensor_to_pil(batch["source_image"][index]))]
                    for label in reference_labels:
                        row.append((label, preloaded_reference_images[(uid, label)]))
                    for label, steps, cfg_scale, prediction in predictions:
                        output_path = eval_dir / f"{sample_name}_{label}.png"
                        prediction_image = tensor_to_pil(prediction[index])
                        prediction_image.save(output_path)
                        generated[label] = str(output_path)
                        row.append((label, prediction_image))
                    row.append(("target", tensor_to_pil(batch["image"][index])))
                    rows.append(row)
                    entries.append(
                        {
                            "index": produced,
                            "uid": uid,
                            "generated_image": next(iter(generated.values())),
                            "generated": generated,
                            "source_image": str(batch.get("source_image_path", [""])[index]),
                            "edit_image": str(batch.get("edit_image_path", [""])[index]),
                            "embedding": str(batch.get("embedding_path", [""])[index]),
                            "uncond_embedding": str(batch.get("uncond_embedding_path", [""])[index]),
                            "jsonl_path": str(batch.get("jsonl_path", [""])[index]),
                            "jsonl_lineno": int(batch.get("jsonl_lineno", [-1])[index]),
                        }
                    )
                    produced += 1
            else:
                produced += batch_size

        if rank != 0:
            return None
        if not rows:
            raise RuntimeError("CFG sweep offline eval produced no samples")
        contact_sheet = eval_dir / "contact_sheet.png"
        make_labeled_contact_sheet(rows, contact_sheet)
        variant_metadata = [
            {
                "label": f"{sampling_style}_{int(steps)}nfe_{cfg_label(float(cfg_scale))}",
                "sampling_steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "cfg_mode": "conditional_only"
                if float(cfg_scale) in {0.0, 1.0}
                else "uncond_plus_scale_cond_delta",
            }
            for steps in sampling_steps
            for cfg_scale in cfg_scales
        ]
        write_json_atomic(
            eval_dir / "manifest.json",
            {
                "global_step": int(global_step),
                "num_samples": len(entries),
                "sampling_style": str(sampling_style),
                "cfg_scales": [float(scale) for scale in cfg_scales],
                "cfg_formula": "uncond + scale * (cond - uncond) except scales 0 and 1, which are conditional-only",
                "variants": variant_metadata,
                "reference_manifest": str(eval_config.get("reference_manifest", "") or ""),
                "reference_labels": reference_labels,
                "contact_sheet": str(contact_sheet),
                "entries": entries,
            },
        )
        return str(contact_sheet)
    finally:
        model_fn.train(was_training)


def benchmark_denoising_only(
    *,
    wrapped_model,
    method,
    eval_dataloader: DataLoader,
    sampling_steps: list[int],
    sampling_style: str,
    max_samples: int,
    seed: int,
    repeats: int,
    warmup: int,
    cfg_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Measure only the student denoising loop, excluding loading and VAE/data work."""
    if repeats <= 0:
        raise ValueError(f"benchmark repeats must be positive, got {repeats}")
    if warmup < 0:
        raise ValueError(f"benchmark warmup must be non-negative, got {warmup}")
    if any(int(steps) <= 0 for steps in sampling_steps):
        raise ValueError(f"benchmark NFE values must be positive, got {sampling_steps}")
    if sampling_style != DMD2_RENOISE_SAMPLING_STYLE:
        raise ValueError(
            "Pure denoising timing currently supports only "
            f"sampling_style={DMD2_RENOISE_SAMPLING_STYLE!r}; "
            "source_flowmatch_euler includes scheduler setup that must be benchmarked separately"
        )
    if cfg_scale not in {0.0, 1.0}:
        raise ValueError(
            "Pure denoising timing supports only the direct-conditional student path "
            f"(cfg_scale 0 or 1), got {cfg_scale}"
        )

    model_fn = get_sampling_model(wrapped_model)
    prepared = prepare_benchmark_inputs(
        wrapped_model=wrapped_model,
        eval_dataloader=eval_dataloader,
        max_samples=max_samples,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    model_fn.eval()
    results = []
    for steps in sampling_steps:
        for warmup_idx in range(warmup):
            sample = prepared[warmup_idx % len(prepared)]
            sample_latents_for_timing(
                method=method,
                model_fn=model_fn,
                sample=sample,
                noise=sample["noise"].clone(),
                generator=torch.Generator(device=device).manual_seed(seed + 100000 + int(steps) * 100 + warmup_idx),
                sampling_steps=int(steps),
                dtype=dtype,
                device=device,
            )
        torch.cuda.synchronize(device)

        durations = []
        for repeat_idx in range(repeats):
            for sample_idx, sample in enumerate(prepared):
                call_seed = seed + 200000 + int(steps) * 10000 + repeat_idx * len(prepared) + sample_idx
                # These setup operations are intentionally outside the timed denoising region.
                noise = sample["noise"].clone()
                generator = torch.Generator(device=device).manual_seed(call_seed)
                synchronize_benchmark_ranks(device)
                started_at = time.perf_counter()
                sample_latents_for_timing(
                    method=method,
                    model_fn=model_fn,
                    sample=sample,
                    noise=noise,
                    generator=generator,
                    sampling_steps=int(steps),
                    dtype=dtype,
                    device=device,
                )
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started_at, device=device, dtype=torch.float64)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                durations.append(float(elapsed.item()))
        results.append(
            {
                "nfe": int(steps),
                "timed_images": len(durations),
                "per_image_seconds_mean": statistics.fmean(durations),
                "per_image_seconds_median": statistics.median(durations),
                "per_image_seconds_min": min(durations),
                "per_image_seconds_max": max(durations),
                "sample_set_seconds_mean": statistics.fmean(durations) * len(prepared),
            }
        )

    return {
        "timing_scope": "student denoising sampling loop only",
        "cfg_scale_requested": float(cfg_scale),
        "cfg_mode": "conditional_only",
        "excluded": [
            "checkpoint/model loading",
            "data loading",
            "prompt/source preparation",
            "VAE encode/decode",
            "image saving",
            "contact-sheet creation",
            "input latent cloning and RNG setup",
        ],
        "rank_aggregation": "max across ranks per measured sample",
        "sampling_style": sampling_style,
        "height": int(prepared[0]["noise"].shape[-2] * wrapped_model.model.vae_scale_factor),
        "width": int(prepared[0]["noise"].shape[-1] * wrapped_model.model.vae_scale_factor),
        "batch_size": 1,
        "sample_count": len(prepared),
        "warmup_per_nfe": int(warmup),
        "repeats_per_image": int(repeats),
        "gpu_name": torch.cuda.get_device_name(device),
        "results": results,
    }


def benchmark_vae_decode_only(
    *,
    wrapped_model,
    method,
    eval_dataloader: DataLoader,
    source_sampling_steps: int,
    sampling_style: str,
    cfg_scale: float,
    max_samples: int,
    seed: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Measure only VAE decode using representative student output latents."""
    if repeats <= 0:
        raise ValueError(f"VAE decode benchmark repeats must be positive, got {repeats}")
    if warmup < 0:
        raise ValueError(f"VAE decode benchmark warmup must be non-negative, got {warmup}")
    if source_sampling_steps <= 0:
        raise ValueError(f"VAE decode source NFE must be positive, got {source_sampling_steps}")
    if sampling_style != DMD2_RENOISE_SAMPLING_STYLE:
        raise ValueError(
            "VAE decode timing currently supports only "
            f"sampling_style={DMD2_RENOISE_SAMPLING_STYLE!r}"
        )
    if cfg_scale not in {0.0, 1.0}:
        raise ValueError(
            "VAE decode timing supports only the direct-conditional student path "
            f"(cfg_scale 0 or 1), got {cfg_scale}"
        )

    model_fn = get_sampling_model(wrapped_model)
    prepared = prepare_benchmark_inputs(
        wrapped_model=wrapped_model,
        eval_dataloader=eval_dataloader,
        max_samples=max_samples,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    model_fn.eval()
    latents = []
    for sample_idx, sample in enumerate(prepared):
        latents.append(
            sample_latents_for_timing(
                method=method,
                model_fn=model_fn,
                sample=sample,
                noise=sample["noise"].clone(),
                generator=torch.Generator(device=device).manual_seed(seed + 500000 + sample_idx),
                sampling_steps=int(source_sampling_steps),
                dtype=dtype,
                device=device,
            )
        )
    torch.cuda.synchronize(device)

    # Decode once outside timing so both shape validation and any first-use work
    # cannot contaminate the decoder-only latency.
    synchronize_benchmark_ranks(device)
    preflight_pixels = wrapped_model.latents_to_pixels(latents[0])
    torch.cuda.synchronize(device)
    validate_decoded_output_shape(
        pixels=preflight_pixels,
        expected_shape=tuple(prepared[0]["output_shape"]),
        device=device,
        benchmark_name="VAE decode timing",
    )
    del preflight_pixels

    for warmup_idx in range(warmup):
        wrapped_model.latents_to_pixels(latents[warmup_idx % len(latents)])
    torch.cuda.synchronize(device)

    durations = []
    for repeat_idx in range(repeats):
        for latent in latents:
            synchronize_benchmark_ranks(device)
            started_at = time.perf_counter()
            pixels = wrapped_model.latents_to_pixels(latent)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started_at, device=device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            durations.append(float(elapsed.item()))

    return {
        "timing_scope": "VAE decode only",
        "cfg_scale_requested": float(cfg_scale),
        "cfg_mode": "conditional_only",
        "representative_latent_source_nfe": int(source_sampling_steps),
        "included": ["output VAE decode"],
        "excluded": [
            "checkpoint/model loading",
            "data and prompt/source preparation",
            "source VAE encode",
            "student sampling loop used to prepare representative latents",
            "unconditional/negative prompt branch",
            "image saving and contact-sheet creation",
        ],
        "rank_aggregation": "max across ranks per measured sample",
        "height": int(latents[0].shape[-2] * wrapped_model.model.vae_scale_factor),
        "width": int(latents[0].shape[-1] * wrapped_model.model.vae_scale_factor),
        "batch_size": 1,
        "sample_count": len(latents),
        "warmup": int(warmup),
        "repeats_per_image": int(repeats),
        "gpu_name": torch.cuda.get_device_name(device),
        "result": {
            "timed_images": len(durations),
            "per_image_seconds_mean": statistics.fmean(durations),
            "per_image_seconds_median": statistics.median(durations),
            "per_image_seconds_min": min(durations),
            "per_image_seconds_max": max(durations),
        },
    }


@torch.no_grad()
def sample_decoded_output_for_timing(
    *,
    wrapped_model,
    method,
    model_fn,
    batch: dict,
    generator: torch.Generator,
    sampling_steps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Run the direct-conditional FireRed path from cached conditions to decoded pixels."""
    source = batch["source_image"].to(device=device)
    prompt_embeds, prompt_mask, _, _ = get_conditions_from_batch(
        wrapped_model,
        batch,
        source,
        device,
        need_uncond=False,
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    source_latents = wrapped_model.pixels_to_latents(source).to(dtype=dtype)
    noise = torch.randn(
        [
            1,
            get_in_channels(model_fn),
            source.shape[-2] // wrapped_model.model.vae_scale_factor,
            source.shape[-1] // wrapped_model.model.vae_scale_factor,
        ],
        dtype=dtype,
        device=device,
        generator=generator,
    )
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
        trajectory = dmd2_renoise_sampling_loop_for_timing(
            method=method,
            noise=noise,
            model_fn=model_fn,
            sampling_steps=int(sampling_steps),
            generator=generator,
            condition=[prompt_embeds, prompt_mask, source_latents],
        )
    return wrapped_model.latents_to_pixels(trajectory[-1])


def collect_output_timing_batches(*, eval_dataloader: DataLoader, max_samples: int) -> list[dict]:
    batches = []
    for batch_list in eval_dataloader:
        if len(batches) >= max_samples:
            break
        batch = batch_list[0]
        if int(batch["source_image"].shape[0]) != 1:
            raise ValueError("Decoded-output timing requires --batch-size=1 for per-image latency")
        require_cached_positive_conditions(batch, benchmark_name="Decoded-output timing")
        batches.append(batch)
    if len(batches) != max_samples:
        raise RuntimeError(f"Expected {max_samples} output-timing samples, collected {len(batches)}")
    return batches


def benchmark_decoded_output(
    *,
    wrapped_model,
    method,
    eval_dataloader: DataLoader,
    sampling_steps: list[int],
    sampling_style: str,
    cfg_scale: float,
    max_samples: int,
    seed: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Measure cached-condition input through VAE-decoded output pixels, excluding I/O/loading."""
    if repeats <= 0:
        raise ValueError(f"output benchmark repeats must be positive, got {repeats}")
    if warmup < 0:
        raise ValueError(f"output benchmark warmup must be non-negative, got {warmup}")
    if any(int(steps) <= 0 for steps in sampling_steps):
        raise ValueError(f"output benchmark NFE values must be positive, got {sampling_steps}")
    if cfg_scale not in {0.0, 1.0}:
        raise ValueError(
            "Decoded-output timing supports only the direct-conditional student path "
            f"(cfg_scale 0 or 1), got {cfg_scale}"
        )
    if sampling_style != DMD2_RENOISE_SAMPLING_STYLE:
        raise ValueError(
            "Decoded-output timing currently supports only "
            f"sampling_style={DMD2_RENOISE_SAMPLING_STYLE!r}"
        )

    model_fn = get_sampling_model(wrapped_model)
    batches = collect_output_timing_batches(eval_dataloader=eval_dataloader, max_samples=max_samples)
    model_fn.eval()
    results = []
    for steps in sampling_steps:
        # Validate the full decoded tensor shape before warm-up/timing. Every rank
        # participates in the collective verdict so FSDP peers cannot hang later.
        preflight_batch = batches[0]
        synchronize_benchmark_ranks(device)
        preflight_pixels = sample_decoded_output_for_timing(
            wrapped_model=wrapped_model,
            method=method,
            model_fn=model_fn,
            batch=preflight_batch,
            generator=torch.Generator(device=device).manual_seed(seed + 250000 + int(steps)),
            sampling_steps=int(steps),
            dtype=dtype,
            device=device,
        )
        torch.cuda.synchronize(device)
        validate_decoded_output_shape(
            pixels=preflight_pixels,
            expected_shape=tuple(preflight_batch["source_image"].shape),
            device=device,
            benchmark_name=f"Decoded-output timing ({steps} NFE)",
        )
        del preflight_pixels

        for warmup_idx in range(warmup):
            batch = batches[warmup_idx % len(batches)]
            sample_decoded_output_for_timing(
                wrapped_model=wrapped_model,
                method=method,
                model_fn=model_fn,
                batch=batch,
                generator=torch.Generator(device=device).manual_seed(
                    seed + 300000 + int(steps) * 100 + warmup_idx
                ),
                sampling_steps=int(steps),
                dtype=dtype,
                device=device,
            )
        torch.cuda.synchronize(device)

        durations = []
        for repeat_idx in range(repeats):
            for sample_idx, batch in enumerate(batches):
                call_seed = seed + 400000 + int(steps) * 10000 + repeat_idx * len(batches) + sample_idx
                synchronize_benchmark_ranks(device)
                started_at = time.perf_counter()
                pixels = sample_decoded_output_for_timing(
                    wrapped_model=wrapped_model,
                    method=method,
                    model_fn=model_fn,
                    batch=batch,
                    generator=torch.Generator(device=device).manual_seed(call_seed),
                    sampling_steps=int(steps),
                    dtype=dtype,
                    device=device,
                )
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - started_at, device=device, dtype=torch.float64)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                durations.append(float(elapsed.item()))
        results.append(
            {
                "nfe": int(steps),
                "timed_images": len(durations),
                "per_image_seconds_mean": statistics.fmean(durations),
                "per_image_seconds_median": statistics.median(durations),
                "per_image_seconds_min": min(durations),
                "per_image_seconds_max": max(durations),
                "sample_set_seconds_mean": statistics.fmean(durations) * len(batches),
            }
        )

    return {
        "timing_scope": "cached-condition input through decoded output pixels",
        "cfg_scale_requested": float(cfg_scale),
        "cfg_mode": "conditional_only",
        "included": [
            "source and cached prompt embedding host-to-device transfer",
            "source VAE encode",
            "noise/RNG creation",
            "student DMD2 re-noise sampling loop",
            "output VAE decode",
        ],
        "excluded": [
            "checkpoint/model loading",
            "JSONL/image/embedding file I/O and CPU image preprocessing",
            "text encoder prompt encoding (evaluation uses cached prompt embeddings)",
            "unconditional/negative prompt branch (cfg=1 is direct conditional-only)",
            "image saving and contact-sheet creation",
        ],
        "rank_aggregation": "max across ranks per measured sample",
        "sampling_style": sampling_style,
        "height": int(batches[0]["source_image"].shape[-2]),
        "width": int(batches[0]["source_image"].shape[-1]),
        "batch_size": 1,
        "sample_count": len(batches),
        "warmup_per_nfe": int(warmup),
        "repeats_per_image": int(repeats),
        "gpu_name": torch.cuda.get_device_name(device),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    method_type = str(config["method"].get("method_type"))
    if method_type not in {"DMD2FullShared", "DMD2FullOfficial"}:
        raise ValueError("This eval script only supports DMD2FullShared or DMD2FullOfficial")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(int(args.seed), rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = torch.bfloat16
    method_cfg = dict(config["method"])
    method_cfg.pop("method_type", None)
    method_cls = DMD2FullOfficialMethod if method_type == "DMD2FullOfficial" else DMD2FullSharedMethod
    method = method_cls(method_cfg)
    sampling_steps = [int(step) for step in args.sampling_steps]
    if any(step <= 0 for step in sampling_steps):
        raise ValueError(f"--sampling-steps must be positive, got {sampling_steps}")
    cfg_scales = [float(scale) for scale in args.cfg_scales]
    if any(not math.isfinite(scale) or scale < 0.0 for scale in cfg_scales):
        raise ValueError(f"--cfg-scales must be finite and non-negative, got {cfg_scales}")
    if len({cfg_label(scale) for scale in cfg_scales}) != len(cfg_scales):
        raise ValueError(f"--cfg-scales must produce unique variant labels, got {cfg_scales}")
    variant_labels = {
        f"{args.sampling_style}_{step}nfe_{cfg_label(scale)}"
        for step in sampling_steps
        for scale in cfg_scales
    }
    if len(variant_labels) != len(sampling_steps) * len(cfg_scales):
        raise ValueError(
            "--sampling-steps and --cfg-scales must produce unique variant labels, "
            f"got steps={sampling_steps} cfg_scales={cfg_scales}"
        )

    wrapped_model = build_fsdp_model(config, local_rank, dtype)
    load_student_checkpoint_for_eval(args.checkpoint, wrapped_model, device)
    wrapped_model.transformer.eval()

    eval_config = dict(config.get("eval", {}) or {})
    eval_config["_cfg_sweep_checkpoint_identity"] = cfg_sweep_checkpoint_identity(
        args.config,
        args.checkpoint,
    )
    if args.eval_jsonl:
        eval_config["eval_jsonl"] = [str(Path(args.eval_jsonl).resolve())]
    eval_config.update(
        {
            "enabled": True,
            "height": int(args.height),
            "width": int(args.width),
            "max_samples": int(args.max_samples),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "cfg_scale": 0.0,
            "sampling_style": str(args.sampling_style),
            "variants": [
                {
                    "label": f"{args.sampling_style}_{steps}nfe_1024",
                    "sampling_style": str(args.sampling_style),
                    "sampling_steps": int(steps),
                }
                for steps in sampling_steps
            ],
        }
    )
    if args.reference_manifest:
        eval_config["reference_manifest"] = args.reference_manifest
        eval_config["reference_labels"] = args.reference_label or ["orig_lora_source_40_cfg4"]
    else:
        eval_config.pop("reference_manifest", None)
        eval_config.pop("reference_labels", None)

    eval_dataloader = make_eval_dataloader(config, args)
    try:
        # CFG=0 is still the historical conditional-only path, but uses the same
        # rank-wide reference validation and invocation namespace as CFG sweeps.
        contact_sheet = run_offline_eval_cfg_sweep(
            wrapped_model=wrapped_model,
            method=method,
            eval_dataloader=eval_dataloader,
            output_dir=args.output_dir,
            global_step=exp_step(args.checkpoint),
            rank=rank,
            device=device,
            dtype=dtype,
            eval_config=eval_config,
            sampling_style=str(args.sampling_style),
            sampling_steps=sampling_steps,
            cfg_scales=cfg_scales,
            seed=int(args.seed),
        )
        if rank == 0:
            invocation_dir = (
                Path(contact_sheet).parent
                if contact_sheet is not None
                else Path(args.output_dir) / "offline_eval" / f"step_{exp_step(args.checkpoint):08d}"
            )
            manifest_path = invocation_dir / "eval_1024_invocation.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "config": str(Path(args.config).resolve()),
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "output_dir": str(Path(args.output_dir).resolve()),
                        "height": int(args.height),
                        "width": int(args.width),
                        "eval_jsonl": str(Path(args.eval_jsonl).resolve()) if args.eval_jsonl else "",
                        "max_samples": int(args.max_samples),
                        "batch_size": int(args.batch_size),
                        "seed": int(args.seed),
                        "world_size": int(world_size),
                        "sampling_style": str(args.sampling_style),
                        "sampling_steps": sampling_steps,
                        "cfg_scales": cfg_scales,
                        "reference_manifest": str(args.reference_manifest or ""),
                        "reference_labels": args.reference_label,
                        "contact_sheet": str(contact_sheet),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"contact_sheet": contact_sheet, "invocation": str(manifest_path)}, indent=2))

        benchmark_cfg = None
        if (
            args.benchmark_repeats > 0
            or args.benchmark_output_repeats > 0
            or args.benchmark_vae_decode_repeats > 0
        ):
            if len(cfg_scales) != 1 or cfg_scales[0] not in {0.0, 1.0}:
                raise ValueError(
                    "Timing requires exactly one direct-conditional cfg scale: --cfg-scales 0 or --cfg-scales 1"
                )
            benchmark_cfg = float(cfg_scales[0])

        if args.benchmark_repeats > 0:
            timing = benchmark_denoising_only(
                wrapped_model=wrapped_model,
                method=method,
                eval_dataloader=eval_dataloader,
                sampling_steps=sampling_steps,
                sampling_style=str(args.sampling_style),
                max_samples=int(args.max_samples),
                seed=int(args.seed),
                repeats=int(args.benchmark_repeats),
                warmup=int(args.benchmark_warmup),
                cfg_scale=benchmark_cfg,
                device=device,
                dtype=dtype,
            )
            if rank == 0:
                timing_path = Path(contact_sheet).parent / "denoising_timing.json"
                timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
                print(json.dumps({"denoising_timing": str(timing_path), "results": timing["results"]}, indent=2))

        if args.benchmark_vae_decode_repeats > 0:
            timing = benchmark_vae_decode_only(
                wrapped_model=wrapped_model,
                method=method,
                eval_dataloader=eval_dataloader,
                source_sampling_steps=int(args.benchmark_vae_decode_source_nfe),
                sampling_style=str(args.sampling_style),
                cfg_scale=benchmark_cfg,
                max_samples=int(args.max_samples),
                seed=int(args.seed),
                repeats=int(args.benchmark_vae_decode_repeats),
                warmup=int(args.benchmark_vae_decode_warmup),
                device=device,
                dtype=dtype,
            )
            if rank == 0:
                timing_path = Path(contact_sheet).parent / "vae_decode_timing.json"
                timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
                print(json.dumps({"vae_decode_timing": str(timing_path), "result": timing["result"]}, indent=2))

        if args.benchmark_output_repeats > 0:
            timing = benchmark_decoded_output(
                wrapped_model=wrapped_model,
                method=method,
                eval_dataloader=eval_dataloader,
                sampling_steps=sampling_steps,
                sampling_style=str(args.sampling_style),
                cfg_scale=benchmark_cfg,
                max_samples=int(args.max_samples),
                seed=int(args.seed),
                repeats=int(args.benchmark_output_repeats),
                warmup=int(args.benchmark_output_warmup),
                device=device,
                dtype=dtype,
            )
            if rank == 0:
                timing_path = Path(contact_sheet).parent / "decoded_output_timing.json"
                timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
                print(json.dumps({"decoded_output_timing": str(timing_path), "results": timing["results"]}, indent=2))
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_distributed()


if __name__ == "__main__":
    main()
