#!/usr/bin/env python
import gc
import json
import os
import random
import sys
import time
from contextlib import nullcontext
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
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import autocast as torch_autocast
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

TWINFLOW_SRC = Path("/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")
if str(TWINFLOW_SRC) not in sys.path:
    sys.path.insert(0, str(TWINFLOW_SRC))

from data.firered_edit_jsonl_dataset import FireRedEditJsonlDataset, collate_firered_edit  # noqa: E402
from networks import MODELS  # noqa: E402
from services.tools import create_logger  # noqa: E402
from steerers.qwenimage.sft_ddp_lora_firered_edit import (  # noqa: E402
    SkipFirstBatchSampler,
    get_conditions_from_batch,
    make_train_state,
    maybe_drop_text_encoder,
    optional_field,
    resolve_resume_path,
    should_run_offline_eval,
)
from steerers.qwenimage.sft_fsdp_firered_edit import (  # noqa: E402
    cleanup_distributed,
    get_fsdp_backend,
    get_fsdp_use_orig_params,
    is_main_process,
    load_full_checkpoint,
    run_offline_eval,
    save_full_checkpoint,
    set_seed,
    setup_distributed,
)


class DMD2FullSharedMethod:
    """LoRA-free full-student DMD2 variant with a shared full fake-score model.

    Upstream DMD2 trains a student, frozen real teacher, and trainable fake critic.
    Three full Qwen/FireRed transformers are not practical here, so this experiment
    trains a full student transformer and uses that same full transformer as the
    fake-score estimator under a no-grad boundary for the distribution-matching
    surrogate. The output checkpoint is a full FSDP transformer checkpoint, not a
    LoRA adapter.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.fm_loss_weight = float(cfg.get("fm_loss_weight", 1.0))
        self.dm_loss_weight = float(cfg.get("dm_loss_weight", 0.5))
        self.fake_loss_weight = float(cfg.get("fake_loss_weight", 0.25))
        self.cfg_bake_loss_weight = float(cfg.get("cfg_bake_loss_weight", 0.0))
        self.dm_grad_clip = float(cfg.get("dm_grad_clip", 10.0))
        self.dm_grad_eps = float(cfg.get("dm_grad_eps", 1.0e-6))
        self.train_cfg_scale = float(cfg.get("train_cfg_scale", 0.0))
        self.train_cfg_mode = str(cfg.get("train_cfg_mode", "guided_grad"))
        self.student_train_sampling_steps = int(cfg.get("student_train_sampling_steps", 1))
        self.student_train_backprop_mode = str(cfg.get("student_train_backprop_mode", "single_step"))
        self.sequential_backward = bool(cfg.get("sequential_backward", False))
        self.debug_timing = bool(cfg.get("debug_timing", False))
        if str(cfg.get("critic_mode", "shared_full")) != "shared_full":
            raise ValueError("Full DMD2 currently supports method.critic_mode=shared_full only")
        if self.train_cfg_mode not in {"guided_grad", "teacher_detached"}:
            raise ValueError(
                f"method.train_cfg_mode must be guided_grad or teacher_detached, got {self.train_cfg_mode!r}"
            )
        if self.student_train_sampling_steps <= 0:
            raise ValueError(
                f"method.student_train_sampling_steps must be positive, got {self.student_train_sampling_steps}"
            )
        if self.student_train_backprop_mode not in {"single_step", "full_rollout"}:
            raise ValueError(
                "method.student_train_backprop_mode must be one of "
                f"single_step/full_rollout, got {self.student_train_backprop_mode!r}"
            )

    @property
    def requires_uncond(self) -> bool:
        return self.train_cfg_scale > 0.0 and (
            self.train_cfg_mode == "guided_grad" or self.cfg_bake_loss_weight > 0.0
        )

    @staticmethod
    def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], *([1] * (x.dim() - 1)))

    @staticmethod
    def sample_t(batch_size: int, device: torch.device, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
        return torch.rand(batch_size, device=device).mul(high - low).add(low)

    def flow_x0(self, x_t: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return x_t - self.expand_time(t, x_t) * velocity

    def predict_velocity(
        self,
        *,
        model_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cond_v = model_fn(
            x_t.to(dtype=dtype),
            t,
            [prompt_embeds, prompt_mask, source_latents.to(dtype=dtype)],
        )
        if self.train_cfg_scale <= 0.0:
            return cond_v, cond_v, None
        if self.train_cfg_mode == "teacher_detached":
            return cond_v, cond_v, None
        if uncond_embeds is None or uncond_mask is None:
            raise RuntimeError("method.train_cfg_scale > 0 requires unconditional prompt embeddings")
        uncond_v = model_fn(
            x_t.to(dtype=dtype),
            t,
            [uncond_embeds, uncond_mask, source_latents.to(dtype=dtype)],
        )
        guided_v = uncond_v + self.train_cfg_scale * (cond_v - uncond_v)
        return guided_v, cond_v, uncond_v

    def cfg_teacher_target(
        self,
        *,
        model_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_v: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.train_cfg_scale <= 0.0:
            return cond_v.detach()
        if uncond_embeds is None or uncond_mask is None:
            raise RuntimeError("method.train_cfg_scale > 0 requires unconditional prompt embeddings")
        with torch.no_grad():
            uncond_v = model_fn(
                x_t.to(dtype=dtype),
                t,
                [uncond_embeds, uncond_mask, source_latents.to(dtype=dtype)],
            )
            guided_v = uncond_v + self.train_cfg_scale * (cond_v.detach() - uncond_v)
        return guided_v

    def student_rollout(
        self,
        *,
        initial_latents: torch.Tensor,
        model_fn,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        latents = initial_latents
        batch_size = int(latents.shape[0])
        for idx in range(self.student_train_sampling_steps):
            t_curr = 1.0 - float(idx) / float(self.student_train_sampling_steps)
            t_next = 1.0 - float(idx + 1) / float(self.student_train_sampling_steps)
            t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
            velocity, _, _ = self.predict_velocity(
                model_fn=model_fn,
                x_t=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )
            latents = latents + (t_next - t_curr) * velocity
        return latents

    def student_train_sample(
        self,
        *,
        initial_latents: torch.Tensor,
        model_fn,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.student_train_backprop_mode == "full_rollout":
            return self.student_rollout(
                initial_latents=initial_latents,
                model_fn=model_fn,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )

        latents = initial_latents
        batch_size = int(latents.shape[0])
        selected = torch.randint(
            low=0,
            high=self.student_train_sampling_steps,
            size=(1,),
            device=latents.device,
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(selected, src=0)
        selected_idx = int(selected.item())

        with torch.no_grad():
            for idx in range(selected_idx):
                t_curr = 1.0 - float(idx) / float(self.student_train_sampling_steps)
                t_next = 1.0 - float(idx + 1) / float(self.student_train_sampling_steps)
                t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
                velocity, _, _ = self.predict_velocity(
                    model_fn=model_fn,
                    x_t=latents,
                    t=t,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                    uncond_embeds=uncond_embeds,
                    uncond_mask=uncond_mask,
                )
                latents = latents + (t_next - t_curr) * velocity

        t_curr = 1.0 - float(selected_idx) / float(self.student_train_sampling_steps)
        t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
        velocity, _, _ = self.predict_velocity(
            model_fn=model_fn,
            x_t=latents,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
        )
        return self.flow_x0(latents, t, velocity)

    def training_step(
        self,
        *,
        model_fn,
        target_latents: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict]:
        batch_size = int(target_latents.shape[0])
        device = target_latents.device

        t = self.sample_t(batch_size, device)
        noise = torch.randn_like(target_latents)
        x_t = self.expand_time(t, target_latents) * noise + (1.0 - self.expand_time(t, target_latents)) * target_latents
        target_v = noise - target_latents

        student_v, cond_v, _ = self.predict_velocity(
            model_fn=model_fn,
            x_t=x_t,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
        )
        loss_fm = F.mse_loss(student_v.float(), target_v.float())
        if self.cfg_bake_loss_weight > 0.0 and self.train_cfg_scale > 0.0:
            if self.train_cfg_mode == "teacher_detached":
                cfg_target_v = self.cfg_teacher_target(
                    model_fn=model_fn,
                    x_t=x_t,
                    t=t,
                    cond_v=cond_v,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                    uncond_embeds=uncond_embeds,
                    uncond_mask=uncond_mask,
                )
            else:
                cfg_target_v = student_v.detach()
            loss_cfg_bake = F.mse_loss(cond_v.float(), cfg_target_v.float())
        else:
            loss_cfg_bake = student_v.float().new_zeros(())

        gen_noise = torch.randn_like(target_latents)
        student_x0 = self.student_train_sample(
            initial_latents=gen_noise,
            model_fn=model_fn,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
        )

        with torch.no_grad():
            t_dm = self.sample_t(batch_size, device)
            dm_noise = torch.randn_like(student_x0)
            dm_x_t = self.expand_time(t_dm, student_x0) * dm_noise + (
                1.0 - self.expand_time(t_dm, student_x0)
            ) * student_x0.detach()
            fake_v, _, _ = self.predict_velocity(
                model_fn=model_fn,
                x_t=dm_x_t,
                t=t_dm,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )
            pred_fake_x0 = self.flow_x0(dm_x_t, t_dm, fake_v)
            pred_real_x0 = target_latents
            dm_grad = pred_fake_x0 - pred_real_x0
            dm_norm = dm_grad.abs().mean().clamp_min(self.dm_grad_eps)
            dm_grad = torch.nan_to_num(dm_grad / dm_norm, nan=0.0, posinf=self.dm_grad_clip, neginf=-self.dm_grad_clip)
            dm_grad = dm_grad.clamp(-self.dm_grad_clip, self.dm_grad_clip)
            dm_target = (student_x0 - dm_grad).detach()
        loss_dm = 0.5 * F.mse_loss(student_x0.float(), dm_target.float())

        if self.fake_loss_weight > 0.0:
            fake_t = self.sample_t(batch_size, device)
            fake_noise = torch.randn_like(student_x0)
            fake_latents = student_x0.detach()
            fake_x_t = self.expand_time(fake_t, fake_latents) * fake_noise + (
                1.0 - self.expand_time(fake_t, fake_latents)
            ) * fake_latents
            fake_target_v = fake_noise - fake_latents
            fake_pred_v, _, _ = self.predict_velocity(
                model_fn=model_fn,
                x_t=fake_x_t,
                t=fake_t,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )
            loss_fake = F.mse_loss(fake_pred_v.float(), fake_target_v.float())
        else:
            loss_fake = student_x0.float().new_zeros(())

        loss = (
            self.fm_loss_weight * loss_fm
            + self.dm_loss_weight * loss_dm
            + self.fake_loss_weight * loss_fake
            + self.cfg_bake_loss_weight * loss_cfg_bake
        )
        metrics = {
            "loss": loss.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_dm": loss_dm.detach(),
            "loss_fake": loss_fake.detach(),
            "loss_cfg_bake": loss_cfg_bake.detach(),
        }
        return loss, metrics

    @staticmethod
    def require_finite(loss: torch.Tensor, name: str) -> None:
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError(f"Non-finite {name}")

    def training_step_backward(
        self,
        *,
        model_fn,
        target_latents: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
        dtype: torch.dtype,
        backward_scale: float,
        debug_prefix: str = "",
    ) -> tuple[torch.Tensor, dict]:
        batch_size = int(target_latents.shape[0])
        device = target_latents.device
        debug_enabled = self.debug_timing and bool(debug_prefix)
        debug_last = time.time()

        def debug_begin(name: str) -> None:
            if debug_enabled:
                print(f"[debug_timing] {debug_prefix} begin={name}", flush=True)

        def debug_end(name: str) -> None:
            nonlocal debug_last
            if debug_enabled:
                torch.cuda.synchronize(device)
                now = time.time()
                print(f"[debug_timing] {debug_prefix} end={name} dt={now - debug_last:.2f}s", flush=True)
                debug_last = now

        debug_begin("entry")
        debug_end("entry")
        debug_begin("sample_t_noise")
        t = self.sample_t(batch_size, device)
        noise = torch.randn_like(target_latents)
        debug_end("sample_t_noise")
        debug_begin("make_x_t")
        x_t = self.expand_time(t, target_latents) * noise + (1.0 - self.expand_time(t, target_latents)) * target_latents
        target_v = noise - target_latents
        debug_end("make_x_t")

        debug_begin("fm_predict")
        student_v, cond_v, _ = self.predict_velocity(
            model_fn=model_fn,
            x_t=x_t,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
        )
        debug_end("fm_predict")
        loss_fm = F.mse_loss(student_v.float(), target_v.float())
        if self.cfg_bake_loss_weight > 0.0 and self.train_cfg_scale > 0.0:
            if self.train_cfg_mode == "teacher_detached":
                debug_begin("cfg_teacher_target")
                cfg_target_v = self.cfg_teacher_target(
                    model_fn=model_fn,
                    x_t=x_t,
                    t=t,
                    cond_v=cond_v,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                    uncond_embeds=uncond_embeds,
                    uncond_mask=uncond_mask,
                )
                debug_end("cfg_teacher_target")
            else:
                cfg_target_v = student_v.detach()
            loss_cfg_bake = F.mse_loss(cond_v.float(), cfg_target_v.float())
        else:
            loss_cfg_bake = student_v.float().new_zeros(())

        fm_phase = self.fm_loss_weight * loss_fm + self.cfg_bake_loss_weight * loss_cfg_bake
        self.require_finite(fm_phase, "fm_phase")
        if fm_phase.requires_grad:
            debug_begin("fm_backward")
            (fm_phase * backward_scale).backward()
            debug_end("fm_backward")
        loss_fm_detached = loss_fm.detach()
        loss_cfg_bake_detached = loss_cfg_bake.detach()
        del t, noise, x_t, target_v, student_v, cond_v, loss_fm, loss_cfg_bake, fm_phase

        gen_noise = torch.randn_like(target_latents)
        debug_begin("student_train_sample")
        student_x0 = self.student_train_sample(
            initial_latents=gen_noise,
            model_fn=model_fn,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
        )
        debug_end("student_train_sample")

        with torch.no_grad():
            t_dm = self.sample_t(batch_size, device)
            dm_noise = torch.randn_like(student_x0)
            dm_x_t = self.expand_time(t_dm, student_x0) * dm_noise + (
                1.0 - self.expand_time(t_dm, student_x0)
            ) * student_x0.detach()
            debug_begin("dm_predict")
            fake_v, _, _ = self.predict_velocity(
                model_fn=model_fn,
                x_t=dm_x_t,
                t=t_dm,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )
            debug_end("dm_predict")
            pred_fake_x0 = self.flow_x0(dm_x_t, t_dm, fake_v)
            pred_real_x0 = target_latents
            dm_grad = pred_fake_x0 - pred_real_x0
            dm_norm = dm_grad.abs().mean().clamp_min(self.dm_grad_eps)
            dm_grad = torch.nan_to_num(dm_grad / dm_norm, nan=0.0, posinf=self.dm_grad_clip, neginf=-self.dm_grad_clip)
            dm_grad = dm_grad.clamp(-self.dm_grad_clip, self.dm_grad_clip)
            dm_target = (student_x0 - dm_grad).detach()
        loss_dm = 0.5 * F.mse_loss(student_x0.float(), dm_target.float())

        if self.fake_loss_weight > 0.0:
            fake_t = self.sample_t(batch_size, device)
            fake_noise = torch.randn_like(student_x0)
            fake_latents = student_x0.detach()
            fake_x_t = self.expand_time(fake_t, fake_latents) * fake_noise + (
                1.0 - self.expand_time(fake_t, fake_latents)
            ) * fake_latents
            fake_target_v = fake_noise - fake_latents
            fake_pred_v, _, _ = self.predict_velocity(
                model_fn=model_fn,
                x_t=fake_x_t,
                t=fake_t,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
                uncond_embeds=uncond_embeds,
                uncond_mask=uncond_mask,
            )
            loss_fake = F.mse_loss(fake_pred_v.float(), fake_target_v.float())
        else:
            loss_fake = student_x0.float().new_zeros(())

        dm_phase = self.dm_loss_weight * loss_dm + self.fake_loss_weight * loss_fake
        self.require_finite(dm_phase, "dm_phase")
        if dm_phase.requires_grad:
            debug_begin("dm_backward")
            (dm_phase * backward_scale).backward()
            debug_end("dm_backward")

        loss = (
            self.fm_loss_weight * loss_fm_detached
            + self.dm_loss_weight * loss_dm.detach()
            + self.fake_loss_weight * loss_fake.detach()
            + self.cfg_bake_loss_weight * loss_cfg_bake_detached
        )
        metrics = {
            "loss": loss.detach(),
            "loss_fm": loss_fm_detached,
            "loss_dm": loss_dm.detach(),
            "loss_fake": loss_fake.detach(),
            "loss_cfg_bake": loss_cfg_bake_detached,
        }
        return loss.detach(), metrics

    @torch.no_grad()
    def sampling_loop(
        self,
        noise: torch.Tensor,
        model_fn,
        *,
        sampling_steps: int,
        sampling_style: str,
        c,
        **_,
    ):
        if sampling_style != "few":
            raise ValueError(f"DMD2FullShared eval only supports sampling_style='few', got {sampling_style!r}")
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
        latents = noise
        trajectory = []
        batch_size = int(latents.shape[0])
        for idx in range(int(sampling_steps)):
            t_curr = 1.0 - float(idx) / float(sampling_steps)
            t_next = 1.0 - float(idx + 1) / float(sampling_steps)
            t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
            velocity = model_fn(latents, t, c)
            latents = latents + (t_next - t_curr) * velocity
            trajectory.append(latents)
        return trajectory


def all_reduce_mean(value: torch.Tensor, world_size: int) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced.item() / world_size)


def cuda_barrier(local_rank: int) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def validate_config(config: dict) -> None:
    if config["method"].get("method_type") != "DMD2FullShared":
        raise ValueError(f"method.method_type must be DMD2FullShared, got {config['method'].get('method_type')}")
    if config["model"].get("model_name") != "QwenImageEdit":
        raise ValueError(f"model.model_name must be QwenImageEdit, got {config['model'].get('model_name')}")
    if float((config.get("sample") or {}).get("cfg_scale", 0.0)) != 0.0:
        raise ValueError("sample.cfg_scale must be 0 for FireRed gray DMD2")
    if float((config.get("eval") or {}).get("cfg_scale", 0.0)) != 0.0:
        raise ValueError("eval.cfg_scale must be 0 for FireRed gray DMD2")
    if int(config["train"].get("checkpoints_total_limit", 0)) != 1:
        raise ValueError("train.checkpoints_total_limit must be 1 for this run")
    if int(config["train"].get("save_every", 0)) != 500:
        raise ValueError("train.save_every must be 500")
    method_cfg = config.get("method") or {}
    if int(method_cfg.get("student_train_sampling_steps", 1)) <= 0:
        raise ValueError("method.student_train_sampling_steps must be positive")
    if str(method_cfg.get("train_cfg_mode", "guided_grad")) not in {"guided_grad", "teacher_detached"}:
        raise ValueError("method.train_cfg_mode must be guided_grad or teacher_detached")
    if str(method_cfg.get("student_train_backprop_mode", "single_step")) not in {"single_step", "full_rollout"}:
        raise ValueError("method.student_train_backprop_mode must be single_step or full_rollout")
    if float(method_cfg.get("train_cfg_scale", 0.0)) > 0.0 and not config["data"].get("uncond_embedding_field"):
        raise ValueError("method.train_cfg_scale > 0 requires data.uncond_embedding_field")
    eval_cfg = config.get("eval") or {}
    if int(eval_cfg.get("every_steps", 0)) != 500:
        raise ValueError("eval.every_steps must be 500")
    for item in eval_cfg.get("variants", []):
        if str(item.get("sampling_style", "few")) != "few":
            raise ValueError(f"DMD2 full eval variants must use sampling_style=few: {item}")


def maybe_enable_qwen_block_timing(wrapped_model, enabled: bool, rank: int, local_rank: int) -> None:
    if not enabled:
        return
    transformer = getattr(getattr(wrapped_model, "model", None), "transformer", None)
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        print(f"[block_timing] rank={rank:02d} local={local_rank} no transformer_blocks found", flush=True)
        return

    for block_idx, block in enumerate(blocks):
        original_forward = block.forward

        def timed_forward(*args, __forward=original_forward, __block_idx=block_idx, **kwargs):
            start = time.time()
            print(f"[block_timing] rank={rank:02d} local={local_rank} block={__block_idx:02d} begin", flush=True)
            output = __forward(*args, **kwargs)
            torch.cuda.synchronize()
            print(
                f"[block_timing] rank={rank:02d} local={local_rank} block={__block_idx:02d} end dt={time.time() - start:.2f}s",
                flush=True,
            )
            return output

        block.forward = timed_forward
    print(f"[block_timing] rank={rank:02d} local={local_rank} enabled blocks={len(blocks)}", flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/train_firered_dmd2_full_fsdp.py <config.yaml>")
    config_path = sys.argv[1]
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    validate_config(config)

    fsdp_backend = get_fsdp_backend(config)
    fsdp_use_orig_params = get_fsdp_use_orig_params(config)
    method_cfg = dict(config["method"])
    method_cfg.pop("method_type")
    method = DMD2FullSharedMethod(method_cfg)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(int(config["train"].get("seed", 42)), rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parent_path = Path(config_path)
    exp_name = str(Path(parent_path.parent.name) / parent_path.stem)
    config["train"]["output_dir"] = os.path.join(config["train"]["output_dir"], exp_name)
    os.makedirs(config["train"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["train"]["output_dir"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config["train"]["output_dir"], "offline_eval"), exist_ok=True)
    cuda_barrier(local_rank)
    logger = create_logger(__name__, config["train"]["output_dir"])

    dtype = torch.bfloat16
    checkpoint_metadata = {
        "checkpoint_format": "torch_distributed_checkpoint_sharded_with_optimizer",
        "config_path": os.path.abspath(config_path),
        "fsdp_backend": fsdp_backend,
        "fsdp_use_orig_params": fsdp_use_orig_params,
        "model_name": config["model"]["model_name"],
        "model_path": config["model"]["model_path"],
        "method_type": "DMD2FullShared",
        "critic_mode": method_cfg.get("critic_mode", "shared_full"),
        "student_train_sampling_steps": int(method_cfg.get("student_train_sampling_steps", 1)),
        "student_train_backprop_mode": str(method_cfg.get("student_train_backprop_mode", "single_step")),
        "train_cfg_scale": float(method_cfg.get("train_cfg_scale", 0.0)),
        "train_cfg_mode": str(method_cfg.get("train_cfg_mode", "guided_grad")),
        "cfg_bake_loss_weight": float(method_cfg.get("cfg_bake_loss_weight", 0.0)),
    }

    wrapped_model = MODELS[config["model"]["model_name"]](
        model_id=config["model"]["model_path"],
        aux_time_embed=bool(config["model"].get("aux_time_embed", False)),
        text_dtype=dtype,
        imgs_dtype=dtype,
    )
    no_split_modules = [m for m in wrapped_model.model.transformer._no_split_modules]
    maybe_drop_text_encoder(wrapped_model, bool(config["model"].get("drop_text_encoder", True)))
    wrapped_model.transformer.requires_grad_(True)
    if bool(config["train"].get("gradient_checkpointing", True)):
        use_reentrant = bool(config["train"].get("gradient_checkpointing_use_reentrant", False))
        wrapped_model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
        if is_main_process(rank):
            logger.info(f"Gradient checkpointing enabled: use_reentrant={use_reentrant}")
    elif is_main_process(rank):
        logger.info("Gradient checkpointing disabled")
    maybe_enable_qwen_block_timing(wrapped_model, method.debug_timing, rank, local_rank)

    def wrap_fsdp1(module):
        module.float()
        backward_prefetch = (
            BackwardPrefetch.BACKWARD_PRE
            if bool(config["train"].get("fsdp_backward_prefetch", True))
            else None
        )
        return FSDP(
            module,
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
            backward_prefetch=backward_prefetch,
            forward_prefetch=bool(config["train"].get("fsdp_forward_prefetch", True)),
            limit_all_gathers=bool(config["train"].get("fsdp_limit_all_gathers", True)),
            use_orig_params=fsdp_use_orig_params,
        )

    wrapped_model.transformer = wrap_fsdp1(wrapped_model.transformer)
    transformer_ddp = wrapped_model.transformer
    trainable_params = [p for p in transformer_ddp.module.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable full transformer parameters found")
    transformer_ddp.train()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["train"]["lr"]),
        betas=tuple(config["train"].get("betas", [0.9, 0.99])),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        foreach=True,
    )

    resume_path = resolve_resume_path(
        config["train"]["output_dir"],
        str(config["train"].get("load_checkpoint_path", "") or ""),
    )
    initial_global_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    resume_micro_step = 0
    if resume_path:
        resume_state = load_full_checkpoint(
            checkpoint_dir=resume_path,
            wrapped_model=wrapped_model,
            optimizer=optimizer,
            rank=rank,
            require_optimizer=bool(config["train"].get("resume_require_optimizer", True)),
        )
        initial_global_step = int(resume_state["global_step"])
        start_epoch = int(resume_state.get("epoch", 0))
        resume_step_in_epoch = int(resume_state.get("next_step_in_epoch", 0))
        resume_micro_step = int(resume_state.get("micro_step", 0))
        saved_world_size = int(resume_state.get("world_size", -1))
        if bool(config["train"].get("resume_strict_world_size", True)) and saved_world_size != world_size:
            raise RuntimeError(f"Checkpoint world_size={saved_world_size} != current world_size={world_size}")

    micro_batch_size = int(config["train"]["micro_batch_size"])
    dataset = FireRedEditJsonlDataset(
        jsonl_files=list(config["data"]["train_jsonl"]),
        firered_project_root=str(config["data"]["firered_project_root"]),
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        max_samples=config["data"].get("max_samples"),
        repeat=int(config["data"].get("repeat", 1)),
        instruction_field=str(config["data"].get("instruction_field", "instruction")),
        embedding_field=optional_field(config["data"].get("embedding_field", "embeddings_tensor_en")),
        uncond_embedding_field=optional_field(config["data"].get("uncond_embedding_field", "embeddings_tensor_droptext")),
    )
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    base_steps_per_epoch = len(sampler) // micro_batch_size
    if base_steps_per_epoch == 0:
        raise RuntimeError("Dataloader is empty after distributed sampling")
    while resume_step_in_epoch >= base_steps_per_epoch:
        start_epoch += 1
        resume_step_in_epoch -= base_steps_per_epoch

    def make_train_dataloader(skip_batches: int = 0):
        batch_sampler = SkipFirstBatchSampler(
            sampler,
            batch_size=micro_batch_size,
            drop_last=True,
            skip_batches=skip_batches,
        )
        return DataLoader(
            dataset,
            num_workers=int(config["train"].get("num_workers", 0)),
            batch_sampler=batch_sampler,
            pin_memory=True,
            collate_fn=collate_firered_edit,
        )

    eval_config = dict(config.get("eval", {}) or {})
    eval_dataloader = None
    if bool(eval_config.get("enabled", False)):
        eval_jsonl_value = eval_config.get("eval_jsonl", config["data"]["train_jsonl"])
        eval_jsonl = [eval_jsonl_value] if isinstance(eval_jsonl_value, str) else list(eval_jsonl_value)
        eval_dataset = FireRedEditJsonlDataset(
            jsonl_files=eval_jsonl,
            firered_project_root=str(config["data"]["firered_project_root"]),
            height=int(eval_config.get("height", config["data"]["height"])),
            width=int(eval_config.get("width", config["data"]["width"])),
            max_samples=eval_config.get("max_samples"),
            repeat=1,
            instruction_field=str(config["data"].get("instruction_field", "instruction")),
            embedding_field=optional_field(config["data"].get("embedding_field", "embeddings_tensor_en")),
            uncond_embedding_field=optional_field(config["data"].get("uncond_embedding_field", "embeddings_tensor_droptext")),
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            num_workers=int(eval_config.get("num_workers", 0)),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            batch_size=int(eval_config.get("batch_size", 1)),
            collate_fn=collate_firered_edit,
        )

    max_train_steps = int(config["train"].get("max_train_steps", base_steps_per_epoch))
    grad_accum_steps = int(config["train"].get("grad_accumulation_steps", 1))
    save_every = int(config["train"].get("save_every", 500))
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    checkpoints_total_limit = int(config["train"].get("checkpoints_total_limit", 1))
    checkpoint_expected_size_gb = float(config["train"].get("checkpoint_expected_size_gb", 0.0) or 0.0)
    checkpoint_preclean_before_save = bool(config["train"].get("checkpoint_preclean_before_save", True))
    save_final_checkpoint = bool(config["train"].get("save_final_checkpoint", True))
    num_train_epochs = int(config["train"].get("num_train_epochs", 1))
    condition_mode = str(config["data"].get("condition_mode", "offline"))
    if condition_mode != "offline":
        raise ValueError("DMD2 full requires data.condition_mode=offline")

    if is_main_process(rank):
        logger.info(
            "Running FireRed DMD2 full FSDP: records=%d world=%d micro_batch=%d steps_per_epoch=%d "
            "max_train_steps=%d output=%s method=%s student_nfe=%d backprop=%s train_cfg=%.3f cfg_mode=%s cfg_bake=%.3f",
            len(dataset),
            world_size,
            micro_batch_size,
            base_steps_per_epoch,
            max_train_steps,
            config["train"]["output_dir"],
            checkpoint_metadata["critic_mode"],
            checkpoint_metadata["student_train_sampling_steps"],
            checkpoint_metadata["student_train_backprop_mode"],
            checkpoint_metadata["train_cfg_scale"],
            checkpoint_metadata["train_cfg_mode"],
            checkpoint_metadata["cfg_bake_loss_weight"],
        )

    global_step = initial_global_step
    micro_step = resume_micro_step
    current_epoch = start_epoch
    next_step_in_epoch = resume_step_in_epoch
    last_checkpoint_step = initial_global_step if global_step >= max_train_steps else -1
    optimizer.zero_grad(set_to_none=True)

    def debug_msg(phase: str, *, started_at: float | None = None, extra: str = "") -> float:
        now = time.time()
        if method.debug_timing:
            dt = "" if started_at is None else f" dt={now - started_at:.2f}s"
            suffix = f" {extra}" if extra else ""
            print(
                f"[debug_timing][rank={rank:02d} local={local_rank}] "
                f"next_step={global_step + 1:08d} phase={phase}{dt}{suffix}",
                flush=True,
            )
        return now

    def current_train_state():
        return make_train_state(
            global_step=global_step,
            epoch=current_epoch,
            next_step_in_epoch=next_step_in_epoch,
            micro_step=micro_step,
            config_path=config_path,
            world_size=world_size,
            train_config=config["train"],
        )

    try:
        for epoch in range(start_epoch, num_train_epochs):
            if global_step >= max_train_steps:
                break
            current_epoch = epoch
            sampler.set_epoch(epoch)
            skip_batches = resume_step_in_epoch if epoch == start_epoch else 0
            debug_t0 = debug_msg("make_train_dataloader_begin", extra=f"epoch={epoch} skip={skip_batches}")
            train_dataloader = make_train_dataloader(skip_batches=skip_batches)
            debug_msg("make_train_dataloader_end", started_at=debug_t0, extra=f"epoch={epoch}")
            train_iter = iter(train_dataloader)
            local_step_in_epoch = 0
            while True:
                debug_t0 = debug_msg(
                    "dataloader_next_begin",
                    extra=f"epoch={epoch} local_step={local_step_in_epoch}",
                )
                try:
                    batch_list = next(train_iter)
                except StopIteration:
                    debug_msg("dataloader_stop", started_at=debug_t0, extra=f"epoch={epoch}")
                    break
                step_in_epoch = skip_batches + local_step_in_epoch
                batch = batch_list[0]
                debug_msg(
                    "dataloader_next_end",
                    started_at=debug_t0,
                    extra=f"epoch={epoch} local_step={local_step_in_epoch} uid={batch.get('uid', [])[:1]}",
                )
                local_step_in_epoch += 1
                micro_step += 1
                is_sync_step = micro_step % grad_accum_steps == 0
                sync_context = nullcontext() if is_sync_step else transformer_ddp.no_sync()

                debug_t0 = debug_msg("h2d_begin", extra=f"uid={batch.get('uid', [])[:1]}")
                image = batch["image"].to(device=device)
                source = batch["source_image"].to(device=device)
                debug_msg("h2d_end", started_at=debug_t0)
                start_time = time.time()
                with sync_context:
                    with torch.no_grad():
                        debug_t0 = debug_msg("vae_encode_begin")
                        target_latents = wrapped_model.pixels_to_latents(image).to(dtype)
                        source_latents = wrapped_model.pixels_to_latents(source).to(dtype)
                        debug_msg("vae_encode_end", started_at=debug_t0)
                        debug_t0 = debug_msg("conditions_begin")
                        prompt_embeds, prompt_mask, uncond_embeds, uncond_mask = get_conditions_from_batch(
                            wrapped_model,
                            batch,
                            source,
                            device,
                            need_uncond=method.requires_uncond,
                        )
                        if condition_mode == "offline" and batch.get("prompt_embeds") is None:
                            raise RuntimeError("data.condition_mode=offline but batch has no prompt_embeds")
                        prompt_embeds = prompt_embeds.to(dtype=dtype)
                        if method.requires_uncond:
                            uncond_embeds = uncond_embeds.to(dtype=dtype)
                        debug_msg("conditions_end", started_at=debug_t0)

                    with torch_autocast(enabled=True, dtype=dtype, device_type="cuda", cache_enabled=False):
                        debug_t0 = debug_msg("method_step_begin")
                        if method.sequential_backward:
                            loss, metrics = method.training_step_backward(
                                model_fn=transformer_ddp,
                                target_latents=target_latents,
                                source_latents=source_latents,
                                prompt_embeds=prompt_embeds,
                                prompt_mask=prompt_mask,
                                uncond_embeds=uncond_embeds,
                                uncond_mask=uncond_mask,
                                dtype=dtype,
                                backward_scale=1.0 / float(grad_accum_steps),
                                debug_prefix=(
                                    f"rank={rank:02d} local={local_rank} step={global_step + 1:08d}"
                                    if method.debug_timing
                                    else ""
                                ),
                            )
                        else:
                            loss, metrics = method.training_step(
                                model_fn=transformer_ddp,
                                target_latents=target_latents,
                                source_latents=source_latents,
                                prompt_embeds=prompt_embeds,
                                prompt_mask=prompt_mask,
                                uncond_embeds=uncond_embeds,
                                uncond_mask=uncond_mask,
                                dtype=dtype,
                            )
                        debug_msg("method_step_end", started_at=debug_t0)
                    finite_flag = torch.tensor(
                        0 if torch.isfinite(loss.detach()).all().item() else 1,
                        device=device,
                        dtype=torch.int,
                    )
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MAX)
                    if int(finite_flag.item()) != 0:
                        raise FloatingPointError(f"Non-finite loss at global_step={global_step}")
                    if not method.sequential_backward:
                        (loss / grad_accum_steps).backward()

                if is_sync_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    next_step_in_epoch = step_in_epoch + 1

                    avg = {name: all_reduce_mean(value, world_size) for name, value in metrics.items()}
                    if is_main_process(rank):
                        logger.info(
                            "step=%08d loss=%.6f fm=%.6f dm=%.6f fake=%.6f cfg_bake=%.6f grad_norm=%.4f step_time=%.2fs uid=%s",
                            global_step,
                            avg["loss"],
                            avg["loss_fm"],
                            avg["loss_dm"],
                            avg["loss_fake"],
                            avg["loss_cfg_bake"],
                            float(grad_norm),
                            time.time() - start_time,
                            batch.get("uid", [])[:2],
                        )

                    if should_run_offline_eval(global_step, eval_config):
                        eval_failed = torch.zeros((), dtype=torch.int, device=device)
                        try:
                            contact_sheet = run_offline_eval(
                                wrapped_model,
                                method,
                                eval_dataloader,
                                config.get("sample", {}),
                                eval_config,
                                config["train"]["output_dir"],
                                global_step,
                                rank,
                                device,
                                dtype,
                            )
                            if is_main_process(rank):
                                logger.info("offline_eval step=%08d contact_sheet=%s", global_step, contact_sheet)
                        except Exception:
                            if is_main_process(rank):
                                logger.exception("offline_eval failed at step=%08d", global_step)
                            eval_failed.fill_(1)
                        dist.all_reduce(eval_failed, op=dist.ReduceOp.MAX)
                        if int(eval_failed.item()) != 0:
                            raise RuntimeError("Offline eval failed; see rank0 log for details")
                        cuda_barrier(local_rank)

                    if save_every > 0 and global_step % save_every == 0:
                        ckpt_path = save_full_checkpoint(
                            output_dir=config["train"]["output_dir"],
                            wrapped_model=wrapped_model,
                            optimizer=optimizer,
                            global_step=global_step,
                            rank=rank,
                            train_state=current_train_state(),
                            checkpoint_metadata=checkpoint_metadata,
                            checkpoints_total_limit=checkpoints_total_limit,
                            checkpoint_expected_size_gb=checkpoint_expected_size_gb,
                            checkpoint_preclean_before_save=checkpoint_preclean_before_save,
                        )
                        last_checkpoint_step = global_step
                        if is_main_process(rank):
                            logger.info("Saved full checkpoint to %s", ckpt_path)

                    if global_step >= max_train_steps:
                        break

                del batch, image, source

            resume_step_in_epoch = 0
            next_step_in_epoch = 0

        if global_step < max_train_steps:
            raise RuntimeError(
                f"Training ended at global_step={global_step}, below max_train_steps={max_train_steps}. "
                "Increase train.num_train_epochs or data.repeat."
            )

        if save_final_checkpoint and last_checkpoint_step != global_step:
            ckpt_path = save_full_checkpoint(
                output_dir=config["train"]["output_dir"],
                wrapped_model=wrapped_model,
                optimizer=optimizer,
                global_step=global_step,
                rank=rank,
                train_state=current_train_state(),
                checkpoint_metadata=checkpoint_metadata,
                checkpoints_total_limit=checkpoints_total_limit,
                checkpoint_expected_size_gb=checkpoint_expected_size_gb,
                checkpoint_preclean_before_save=checkpoint_preclean_before_save,
            )
        elif last_checkpoint_step == global_step:
            ckpt_path = str(Path(config["train"]["output_dir"]) / "checkpoints" / f"global_step_{global_step}")
        else:
            ckpt_path = ""

        if is_main_process(rank):
            manifest = {
                "run_dir": str(config["train"]["output_dir"]),
                "checkpoint_dir": str(ckpt_path),
                "global_step": int(global_step),
                "status": "ok",
                "method": checkpoint_metadata,
            }
            with open(Path(config["train"]["output_dir"]) / "manifest.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
            logger.info("completed %s", json.dumps(manifest, ensure_ascii=True))
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_distributed()


if __name__ == "__main__":
    main()
