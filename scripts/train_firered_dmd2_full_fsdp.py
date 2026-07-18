#!/usr/bin/env python
import gc
import hashlib
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
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torch import nn
from torch.amp import autocast as torch_autocast
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

TWINFLOW_SRC = Path(
    os.environ.get("TWINFLOW_SRC", "/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")
).expanduser()
if str(TWINFLOW_SRC) not in sys.path:
    sys.path.insert(0, str(TWINFLOW_SRC))

from data.firered_edit_jsonl_dataset import FireRedEditJsonlDataset, collate_firered_edit  # noqa: E402
from networks import MODELS  # noqa: E402
from services.tools import create_logger  # noqa: E402
from steerers.qwenimage.sft_ddp_lora_firered_edit import (  # noqa: E402
    SkipFirstBatchSampler,
    cleanup_old_checkpoints,
    get_conditions_from_batch,
    load_rng_state,
    load_train_state,
    make_train_state,
    maybe_drop_text_encoder,
    optional_field,
    resolve_resume_path,
    should_run_offline_eval,
    write_json_atomic,
)
from steerers.qwenimage.sft_fsdp_firered_edit import (  # noqa: E402
    assert_checkpoint_free_space,
    cleanup_distributed,
    fsdp_model_state_dict,
    fsdp_optimizer_state_dict,
    get_fsdp_backend,
    get_fsdp_use_orig_params,
    get_ckpt_paths,
    is_main_process,
    load_fsdp_model_dcp,
    load_fsdp_optimizer_dcp,
    maybe_preclean_checkpoints_for_save,
    reset_checkpoint_artifacts,
    run_offline_eval,
    set_seed,
    setup_distributed,
    write_failure_marker,
)

DMD2_RENOISE_SAMPLING_STYLE = "dmd2_renoise"
SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE = "source_flowmatch_euler"


class QwenHiddenStateGANClassifier(nn.Module):
    """DMD2-style real/fake head on a Qwen transformer middle hidden state.

    Upstream DMD2 runs real/fake latents through the fake UNet and attaches a
    classifier to the bottleneck feature. For FireRed/QwenImageEdit we hook a
    configured transformer block on the fake critic, split the edit/source image
    token grids, and condition the head on pooled prompt embeddings plus the
    diffusion time. The head is training-only and is not saved for inference.
    """

    def __init__(self, transformer_hidden_dim: int, prompt_dim: int, hidden_channels: int = 128) -> None:
        super().__init__()
        if transformer_hidden_dim <= 0:
            raise ValueError(f"transformer_hidden_dim must be positive, got {transformer_hidden_dim}")
        if prompt_dim <= 0:
            raise ValueError(f"prompt_dim must be positive, got {prompt_dim}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        final_channels = hidden_channels * 4
        self.transformer_hidden_dim = int(transformer_hidden_dim)
        self.prompt_dim = int(prompt_dim)
        self.hidden_channels = int(hidden_channels)
        self.spatial_head = nn.Sequential(
            nn.Conv2d(transformer_hidden_dim * 2, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, hidden_channels * 2), num_channels=hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, final_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, final_channels), num_channels=final_channels),
            nn.SiLU(),
        )
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, final_channels),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, final_channels),
            nn.SiLU(),
            nn.Linear(final_channels, final_channels),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(final_channels * 4),
            nn.Linear(final_channels * 4, final_channels),
            nn.SiLU(),
            nn.Linear(final_channels, 1),
        )

    @staticmethod
    def _masked_prompt_mean(prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None) -> torch.Tensor:
        if prompt_embeds.dim() != 3:
            raise ValueError(f"prompt_embeds must be [B, T, C], got shape={tuple(prompt_embeds.shape)}")
        if prompt_mask is None:
            return prompt_embeds.float().mean(dim=1)
        if prompt_mask.dim() != 2 or prompt_mask.shape[:2] != prompt_embeds.shape[:2]:
            raise ValueError(
                "prompt_mask must be [B, T] and match prompt_embeds, "
                f"got mask={tuple(prompt_mask.shape)} embeds={tuple(prompt_embeds.shape)}"
            )
        weights = prompt_mask.to(device=prompt_embeds.device, dtype=torch.float32)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (prompt_embeds.float() * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        hooked_hidden_states: torch.Tensor,
        candidate_latents: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if hooked_hidden_states.dim() != 3:
            raise ValueError(f"hooked_hidden_states must be [B, S, C], got {tuple(hooked_hidden_states.shape)}")
        if candidate_latents.dim() != 4:
            raise ValueError(f"candidate_latents must be BCHW, got shape={tuple(candidate_latents.shape)}")
        if source_latents.shape != candidate_latents.shape:
            raise ValueError(
                "source_latents shape must match candidate_latents for GAN classifier: "
                f"source={tuple(source_latents.shape)} candidate={tuple(candidate_latents.shape)}"
            )
        if hooked_hidden_states.shape[0] != candidate_latents.shape[0]:
            raise ValueError(
                "hooked hidden batch mismatch: "
                f"hidden={hooked_hidden_states.shape[0]} latents={candidate_latents.shape[0]}"
            )
        if hooked_hidden_states.shape[-1] != self.transformer_hidden_dim:
            raise ValueError(
                "hooked hidden dim mismatch: "
                f"expected={self.transformer_hidden_dim} got={hooked_hidden_states.shape[-1]}"
            )
        if t.dim() != 1 or int(t.shape[0]) != int(candidate_latents.shape[0]):
            raise ValueError(f"t must be [B], got shape={tuple(t.shape)} for B={candidate_latents.shape[0]}")

        grid_h = int(candidate_latents.shape[-2]) // 2
        grid_w = int(candidate_latents.shape[-1]) // 2
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"invalid latent grid for classifier: {tuple(candidate_latents.shape)}")
        tokens_per_image = grid_h * grid_w
        required_tokens = tokens_per_image * 2
        if int(hooked_hidden_states.shape[1]) < required_tokens:
            raise ValueError(
                "hooked hidden state does not contain edit+source token grids: "
                f"seq={hooked_hidden_states.shape[1]} required={required_tokens} "
                f"latent_shape={tuple(candidate_latents.shape)}"
            )

        edit_tokens = hooked_hidden_states[:, :tokens_per_image, :].float()
        source_tokens = hooked_hidden_states[:, tokens_per_image:required_tokens, :].float()
        edit_grid = edit_tokens.transpose(1, 2).contiguous().view(
            candidate_latents.shape[0], self.transformer_hidden_dim, grid_h, grid_w
        )
        source_grid = source_tokens.transpose(1, 2).contiguous().view(
            candidate_latents.shape[0], self.transformer_hidden_dim, grid_h, grid_w
        )
        spatial = self.spatial_head(torch.cat([edit_grid, source_grid], dim=1))
        spatial_avg = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        spatial_max = F.adaptive_max_pool2d(spatial, 1).flatten(1)
        prompt_context = self.prompt_proj(self._masked_prompt_mean(prompt_embeds, prompt_mask))
        time_context = self.time_proj(t.to(device=spatial.device, dtype=torch.float32).view(-1, 1))
        logits = self.head(torch.cat([spatial_avg, spatial_max, prompt_context, time_context], dim=1))
        return logits.squeeze(1)


class QwenHiddenStateGANForwarder:
    """Runs fake Qwen critic and feeds a returned block feature to the GAN head."""

    def __init__(self, model_fn, classifier, feature_layer_idx: int) -> None:
        self.model_fn = model_fn
        self.classifier = classifier
        self.feature_layer_idx = int(feature_layer_idx)
        blocks = self._resolve_blocks(model_fn)
        if not blocks:
            raise RuntimeError("Could not find Qwen transformer_blocks for GAN feature extraction")
        if self.feature_layer_idx < 0:
            self.feature_layer_idx += len(blocks)
        if self.feature_layer_idx < 0 or self.feature_layer_idx >= len(blocks):
            raise ValueError(
                f"GAN feature layer index out of range: {self.feature_layer_idx}, num_layers={len(blocks)}"
            )
        self.num_layers = len(blocks)

    @staticmethod
    def _resolve_blocks(model_fn):
        module = getattr(model_fn, "module", model_fn)
        transformer = getattr(module, "transformer", None)
        blocks = getattr(transformer, "transformer_blocks", None)
        return list(blocks) if blocks is not None else []

    def parameters(self):
        yield from self.model_fn.parameters()
        yield from self.classifier.parameters()

    def __call__(
        self,
        candidate_latents: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        t: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        output = self.model_fn(
            candidate_latents.to(dtype=dtype),
            t,
            [prompt_embeds.to(dtype=dtype), prompt_mask, source_latents.to(dtype=dtype)],
            return_hidden_state_layer=self.feature_layer_idx,
        )
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "GAN feature was not returned by Qwen forward. "
                f"layer={self.feature_layer_idx} num_layers={self.num_layers}"
            )
        feature = output[1]
        if not torch.is_tensor(feature) or feature.dim() != 3:
            raise RuntimeError(f"GAN feature expected [B, S, C] tensor, got {type(feature)}")
        return self.classifier(feature, candidate_latents, source_latents, prompt_embeds, prompt_mask, t)


def calculate_qwen_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    m = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    b = base_shift - m * float(base_seq_len)
    return float(image_seq_len) * m + b


@torch.no_grad()
def source_flowmatch_euler_sampling_loop(
    noise: torch.Tensor,
    model_fn,
    *,
    sampling_steps: int,
    c,
):
    """Eval-only sampler matching FireRed/Qwen source FlowMatchEuler updates."""
    if sampling_steps <= 0:
        raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
    from diffusers import FlowMatchEulerDiscreteScheduler

    latents = noise
    trajectory = []
    image_seq_len = int(latents.shape[-2] // 2) * int(latents.shape[-1] // 2)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        invert_sigmas=False,
        shift_terminal=0.02,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        time_shift_type="exponential",
        stochastic_sampling=False,
    )
    sigmas = np.linspace(1.0, 1.0 / float(sampling_steps), int(sampling_steps), dtype=np.float32)
    mu = calculate_qwen_shift(image_seq_len)
    scheduler.set_timesteps(int(sampling_steps), device=latents.device, sigmas=sigmas, mu=mu)
    batch_size = int(latents.shape[0])
    for timestep in scheduler.timesteps:
        t_model = (timestep / 1000.0).expand(batch_size).to(device=latents.device, dtype=torch.float32)
        velocity = model_fn(latents, t_model, c)
        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
        trajectory.append(latents)
    return trajectory


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
        self.clear_cuda_cache_between_phases = bool(cfg.get("clear_cuda_cache_between_phases", False))
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

    def renoise_x0(
        self,
        x0: torch.Tensor,
        t_next: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if t_next <= 0.0:
            return x0
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        t_tensor = torch.full((x0.shape[0],), t_next, device=x0.device, dtype=torch.float32)
        return self.expand_time(t_tensor, x0) * noise + (1.0 - self.expand_time(t_tensor, x0)) * x0

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
            x0 = self.flow_x0(latents, t, velocity)
            latents = self.renoise_x0(x0, t_next)
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
                x0 = self.flow_x0(latents, t, velocity)
                latents = self.renoise_x0(x0, t_next)

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
        self.require_finite(loss_dm, "loss_dm")
        if loss_dm.requires_grad:
            debug_begin("dm_backward")
            (self.dm_loss_weight * loss_dm * backward_scale).backward()
            debug_end("dm_backward")
        loss_dm_detached = loss_dm.detach()
        fake_latents = student_x0.detach() if self.fake_loss_weight > 0.0 else None
        del gen_noise, student_x0, t_dm, dm_noise, dm_x_t, fake_v, pred_fake_x0, pred_real_x0
        del dm_grad, dm_norm, dm_target, loss_dm
        if fake_latents is not None and self.clear_cuda_cache_between_phases:
            torch.cuda.empty_cache()

        if self.fake_loss_weight > 0.0:
            fake_t = self.sample_t(batch_size, device)
            fake_noise = torch.randn_like(fake_latents)
            fake_x_t = self.expand_time(fake_t, fake_latents) * fake_noise + (
                1.0 - self.expand_time(fake_t, fake_latents)
            ) * fake_latents
            fake_target_v = fake_noise - fake_latents
            debug_begin("fake_predict")
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
            debug_end("fake_predict")
            loss_fake = F.mse_loss(fake_pred_v.float(), fake_target_v.float())
            self.require_finite(loss_fake, "loss_fake")
            if loss_fake.requires_grad:
                debug_begin("fake_backward")
                (self.fake_loss_weight * loss_fake * backward_scale).backward()
                debug_end("fake_backward")
        else:
            loss_fake = target_latents.float().new_zeros(())

        loss = (
            self.fm_loss_weight * loss_fm_detached
            + self.dm_loss_weight * loss_dm_detached
            + self.fake_loss_weight * loss_fake.detach()
            + self.cfg_bake_loss_weight * loss_cfg_bake_detached
        )
        metrics = {
            "loss": loss.detach(),
            "loss_fm": loss_fm_detached,
            "loss_dm": loss_dm_detached,
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
        if sampling_style == SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE:
            return source_flowmatch_euler_sampling_loop(
                noise,
                model_fn,
                sampling_steps=int(sampling_steps),
                c=c,
            )
        if sampling_style != DMD2_RENOISE_SAMPLING_STYLE:
            raise ValueError(
                "DMD2FullShared eval only supports official DMD2 re-noise sampling "
                f"({DMD2_RENOISE_SAMPLING_STYLE!r}) or source FlowMatchEuler "
                f"({SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE!r}), got {sampling_style!r}"
            )
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
        latents = noise
        trajectory = []
        batch_size = int(latents.shape[0])
        generator = _.get("generator")
        for idx in range(int(sampling_steps)):
            t_curr = 1.0 - float(idx) / float(sampling_steps)
            t_next = 1.0 - float(idx + 1) / float(sampling_steps)
            t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
            velocity = model_fn(latents, t, c)
            x0 = self.flow_x0(latents, t, velocity)
            trajectory.append(x0)
            latents = self.renoise_x0(x0, t_next, generator=generator)
        return trajectory


class DMD2FullOfficialMethod:
    """Official-style full-model DMD2 for FireRed/QwenImageEdit.

    This variant uses three full transformers:
    - student/generator: trainable and saved for inference
    - real teacher: frozen merged gray FireRed model, queried with CFG
    - fake critic: trainable model that learns the fake distribution
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.teacher_match_loss_weight = float(cfg.get("teacher_match_loss_weight", 0.0))
        self.dm_loss_weight = float(cfg.get("dm_loss_weight", 1.0))
        self.fake_loss_weight = float(cfg.get("fake_loss_weight", 1.0))
        self.gan_loss_weight = float(cfg.get("gan_loss_weight", 0.0))
        self.gan_classifier_loss_weight = float(cfg.get("gan_classifier_loss_weight", 0.0))
        self.gan_classifier_type = str(cfg.get("gan_classifier_type", "qwen_hidden_state"))
        self.gan_feature_layer = cfg.get("gan_feature_layer", "middle")
        self.gan_noise_t_min = float(cfg.get("gan_noise_t_min", 0.0))
        self.gan_noise_t_max = float(cfg.get("gan_noise_t_max", 0.98))
        self.dm_grad_clip = float(cfg.get("dm_grad_clip", 10.0))
        self.dm_grad_eps = float(cfg.get("dm_grad_eps", 1.0e-6))
        self.real_guidance_scale = float(cfg.get("real_guidance_scale", cfg.get("train_cfg_scale", 1.0)))
        self.fake_guidance_scale = float(cfg.get("fake_guidance_scale", 1.0))
        self.student_train_sampling_steps = int(cfg.get("student_train_sampling_steps", 1))
        self.student_train_backprop_mode = str(cfg.get("student_train_backprop_mode", "single_step"))
        self.dfake_gen_update_ratio = int(cfg.get("dfake_gen_update_ratio", 1))
        self.sequential_backward = True
        self.clear_cuda_cache_between_phases = bool(cfg.get("clear_cuda_cache_between_phases", False))
        self.debug_timing = bool(cfg.get("debug_timing", False))
        if str(cfg.get("critic_mode", "separate_full")) != "separate_full":
            raise ValueError("DMD2FullOfficial requires method.critic_mode=separate_full")
        if self.fake_guidance_scale != 1.0:
            raise ValueError(f"Official DMD2 requires fake_guidance_scale=1.0, got {self.fake_guidance_scale}")
        if self.real_guidance_scale <= 1.0:
            raise ValueError(
                "DMD2FullOfficial needs real_guidance_scale > 1.0 to distill CFG into the student; "
                f"got {self.real_guidance_scale}"
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
        if self.dfake_gen_update_ratio <= 0:
            raise ValueError(f"method.dfake_gen_update_ratio must be positive, got {self.dfake_gen_update_ratio}")
        if self.gan_loss_weight < 0.0:
            raise ValueError(f"method.gan_loss_weight must be non-negative, got {self.gan_loss_weight}")
        if self.gan_classifier_loss_weight < 0.0:
            raise ValueError(
                "method.gan_classifier_loss_weight must be non-negative, "
                f"got {self.gan_classifier_loss_weight}"
            )
        if self.uses_gan and self.gan_classifier_type != "qwen_hidden_state":
            raise ValueError(
                "DMD2FullOfficial GAN currently supports method.gan_classifier_type=qwen_hidden_state only, "
                f"got {self.gan_classifier_type!r}"
            )
        if self.gan_noise_t_min < 0.0 or self.gan_noise_t_max > 1.0 or self.gan_noise_t_min > self.gan_noise_t_max:
            raise ValueError(
                "method.gan_noise_t_min/max must satisfy 0 <= min <= max <= 1, "
                f"got min={self.gan_noise_t_min} max={self.gan_noise_t_max}"
            )

    @property
    def uses_gan(self) -> bool:
        return self.gan_loss_weight > 0.0 or self.gan_classifier_loss_weight > 0.0

    @property
    def requires_uncond(self) -> bool:
        return self.real_guidance_scale > 1.0

    @staticmethod
    def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], *([1] * (x.dim() - 1)))

    @staticmethod
    def sample_t(batch_size: int, device: torch.device, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
        return torch.rand(batch_size, device=device).mul(high - low).add(low)

    def sample_gan_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.gan_noise_t_min == self.gan_noise_t_max:
            return torch.full((batch_size,), self.gan_noise_t_min, device=device, dtype=torch.float32)
        return torch.rand(batch_size, device=device).mul(self.gan_noise_t_max - self.gan_noise_t_min).add(
            self.gan_noise_t_min
        )

    def flow_x0(self, x_t: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return x_t - self.expand_time(t, x_t) * velocity

    def noised_latents(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        return self.expand_time(t, x0) * noise + (1.0 - self.expand_time(t, x0)) * x0

    @staticmethod
    def set_requires_grad(module, enabled: bool) -> list[bool]:
        previous = []
        for param in module.parameters():
            previous.append(bool(param.requires_grad))
            param.requires_grad_(enabled)
        return previous

    @staticmethod
    def restore_requires_grad(module, previous: list[bool]) -> None:
        for param, enabled in zip(module.parameters(), previous):
            param.requires_grad_(enabled)

    @staticmethod
    def clear_param_grads(module) -> None:
        for param in module.parameters():
            param.grad = None

    def renoise_x0(
        self,
        x0: torch.Tensor,
        t_next: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if t_next <= 0.0:
            return x0
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        t_tensor = torch.full((x0.shape[0],), t_next, device=x0.device, dtype=torch.float32)
        return self.expand_time(t_tensor, x0) * noise + (1.0 - self.expand_time(t_tensor, x0)) * x0

    def predict_cond_velocity(
        self,
        *,
        model_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return model_fn(
            x_t.to(dtype=dtype),
            t,
            [prompt_embeds, prompt_mask, source_latents.to(dtype=dtype)],
        )

    def predict_guided_velocity(
        self,
        *,
        model_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self.predict_cond_velocity(
                model_fn=model_fn,
                x_t=x_t,
                t=t,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
            )
        if uncond_embeds is None or uncond_mask is None:
            raise RuntimeError("real_guidance_scale > 1 requires unconditional prompt embeddings")
        uncond_v = model_fn(
            x_t.to(dtype=dtype),
            t,
            [uncond_embeds, uncond_mask, source_latents.to(dtype=dtype)],
        )
        cond_v = self.predict_cond_velocity(
            model_fn=model_fn,
            x_t=x_t,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
        )
        return uncond_v + guidance_scale * (cond_v - uncond_v)

    def student_rollout(
        self,
        *,
        initial_latents: torch.Tensor,
        student_model_fn,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        latents = initial_latents
        batch_size = int(latents.shape[0])
        for idx in range(self.student_train_sampling_steps):
            t_curr = 1.0 - float(idx) / float(self.student_train_sampling_steps)
            t_next = 1.0 - float(idx + 1) / float(self.student_train_sampling_steps)
            t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
            velocity = self.predict_cond_velocity(
                model_fn=student_model_fn,
                x_t=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
            )
            x0 = self.flow_x0(latents, t, velocity)
            latents = self.renoise_x0(x0, t_next)
        return latents

    def student_train_sample(
        self,
        *,
        initial_latents: torch.Tensor,
        student_model_fn,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_latents: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.student_train_backprop_mode == "full_rollout":
            return self.student_rollout(
                initial_latents=initial_latents,
                student_model_fn=student_model_fn,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
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
                velocity = self.predict_cond_velocity(
                    model_fn=student_model_fn,
                    x_t=latents,
                    t=t,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                )
                x0 = self.flow_x0(latents, t, velocity)
                latents = self.renoise_x0(x0, t_next)

        t_curr = 1.0 - float(selected_idx) / float(self.student_train_sampling_steps)
        t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
        velocity = self.predict_cond_velocity(
            model_fn=student_model_fn,
            x_t=latents,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
        )
        return self.flow_x0(latents, t, velocity)

    @staticmethod
    def require_finite(loss: torch.Tensor, name: str) -> None:
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError(f"Non-finite {name}")

    def training_step_backward(
        self,
        *,
        student_model_fn,
        fake_model_fn,
        teacher_model_fn,
        gan_classifier_gen_fn,
        gan_classifier_train_fn,
        target_latents: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        uncond_mask: torch.Tensor | None,
        dtype: torch.dtype,
        backward_scale: float,
        compute_student_gradient: bool,
        debug_prefix: str = "",
    ) -> tuple[torch.Tensor, dict]:
        batch_size = int(target_latents.shape[0])
        device = target_latents.device
        if self.uses_gan and (gan_classifier_gen_fn is None or gan_classifier_train_fn is None):
            raise RuntimeError("GAN loss is enabled but GAN classifier module was not provided")

        if compute_student_gradient:
            gen_noise = torch.randn_like(target_latents)
            student_x0 = self.student_train_sample(
                initial_latents=gen_noise,
                student_model_fn=student_model_fn,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                source_latents=source_latents,
                dtype=dtype,
            )
            with torch.no_grad():
                t_dm = self.sample_t(batch_size, device)
                dm_noise = torch.randn_like(student_x0)
                dm_x_t = self.expand_time(t_dm, student_x0) * dm_noise + (
                    1.0 - self.expand_time(t_dm, student_x0)
                ) * student_x0.detach()
                real_v = self.predict_guided_velocity(
                    model_fn=teacher_model_fn,
                    x_t=dm_x_t,
                    t=t_dm,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    uncond_embeds=uncond_embeds,
                    uncond_mask=uncond_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                    guidance_scale=self.real_guidance_scale,
                )
                fake_v = self.predict_cond_velocity(
                    model_fn=fake_model_fn,
                    x_t=dm_x_t,
                    t=t_dm,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                )
                pred_real_x0 = self.flow_x0(dm_x_t, t_dm, real_v)
                pred_fake_x0 = self.flow_x0(dm_x_t, t_dm, fake_v)
                dm_grad = (pred_fake_x0 - pred_real_x0).detach()
                dm_norm = dm_grad.abs().mean().clamp_min(self.dm_grad_eps)
                dm_grad = torch.nan_to_num(
                    dm_grad / dm_norm,
                    nan=0.0,
                    posinf=self.dm_grad_clip,
                    neginf=-self.dm_grad_clip,
                ).clamp(-self.dm_grad_clip, self.dm_grad_clip)
                dm_target = (student_x0 - dm_grad).detach()
            loss_teacher = student_x0.float().new_zeros(())
            loss_dm = 0.5 * F.mse_loss(student_x0.float(), dm_target.float())
            loss_gan_gen = student_x0.float().new_zeros(())
            if self.gan_loss_weight > 0.0:
                assert gan_classifier_gen_fn is not None
                gan_t = self.sample_gan_t(batch_size, device)
                gan_x_t = self.noised_latents(student_x0, gan_t)
                gan_logits_fake_for_gen = gan_classifier_gen_fn(
                    gan_x_t,
                    source_latents,
                    prompt_embeds,
                    prompt_mask,
                    gan_t,
                    dtype=dtype,
                )
                loss_gan_gen = F.softplus(-gan_logits_fake_for_gen.float()).mean()
            loss_student = (
                self.teacher_match_loss_weight * loss_teacher
                + self.dm_loss_weight * loss_dm
                + self.gan_loss_weight * loss_gan_gen
            )
            self.require_finite(loss_student, "loss_student")
            (loss_student * backward_scale).backward()
            if self.gan_loss_weight > 0.0:
                assert gan_classifier_gen_fn is not None
                self.clear_param_grads(gan_classifier_gen_fn)
            generated_latents = student_x0.detach()
        else:
            with torch.no_grad():
                gen_noise = torch.randn_like(target_latents)
                generated_latents = self.student_train_sample(
                    initial_latents=gen_noise,
                    student_model_fn=student_model_fn,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    source_latents=source_latents,
                    dtype=dtype,
                )
                generated_latents = generated_latents.detach()
            loss_teacher = generated_latents.float().new_zeros(())
            loss_dm = generated_latents.float().new_zeros(())
            loss_student = generated_latents.float().new_zeros(())
            loss_gan_gen = generated_latents.float().new_zeros(())

        fake_t = self.sample_t(int(generated_latents.shape[0]), device)
        fake_noise = torch.randn_like(generated_latents)
        fake_x_t = self.expand_time(fake_t, generated_latents) * fake_noise + (
            1.0 - self.expand_time(fake_t, generated_latents)
        ) * generated_latents
        fake_target_v = fake_noise - generated_latents
        fake_pred_v = self.predict_cond_velocity(
            model_fn=fake_model_fn,
            x_t=fake_x_t,
            t=fake_t,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            source_latents=source_latents,
            dtype=dtype,
        )
        loss_fake = F.mse_loss(fake_pred_v.float(), fake_target_v.float())
        self.require_finite(loss_fake, "loss_fake")
        (self.fake_loss_weight * loss_fake * backward_scale).backward()

        loss_gan_classifier = generated_latents.float().new_zeros(())
        gan_logits_real_mean = generated_latents.float().new_zeros(())
        gan_logits_fake_mean = generated_latents.float().new_zeros(())
        if self.gan_classifier_loss_weight > 0.0:
            assert gan_classifier_train_fn is not None
            gan_classifier_backward_scale = self.gan_classifier_loss_weight * backward_scale
            real_t = self.sample_gan_t(batch_size, device)
            real_x_t = self.noised_latents(target_latents.detach(), real_t)
            gan_logits_real = gan_classifier_train_fn(
                real_x_t,
                source_latents.detach(),
                prompt_embeds.detach(),
                prompt_mask,
                real_t,
                dtype=dtype,
            )
            loss_gan_real = F.softplus(-gan_logits_real.float()).mean()
            self.require_finite(loss_gan_real, "loss_gan_classifier_real")
            gan_logits_real_mean = gan_logits_real.detach().float().mean()
            (gan_classifier_backward_scale * loss_gan_real).backward()
            loss_gan_classifier = loss_gan_classifier + loss_gan_real.detach()
            del gan_logits_real, loss_gan_real, real_x_t, real_t

            fake_t_for_gan = self.sample_gan_t(int(generated_latents.shape[0]), device)
            fake_x_t_for_gan = self.noised_latents(generated_latents.detach(), fake_t_for_gan)
            gan_logits_fake = gan_classifier_train_fn(
                fake_x_t_for_gan,
                source_latents.detach(),
                prompt_embeds.detach(),
                prompt_mask,
                fake_t_for_gan,
                dtype=dtype,
            )
            loss_gan_fake = F.softplus(gan_logits_fake.float()).mean()
            self.require_finite(loss_gan_fake, "loss_gan_classifier_fake")
            gan_logits_fake_mean = gan_logits_fake.detach().float().mean()
            (gan_classifier_backward_scale * loss_gan_fake).backward()
            loss_gan_classifier = loss_gan_classifier + loss_gan_fake.detach()
            del gan_logits_fake, loss_gan_fake, fake_x_t_for_gan, fake_t_for_gan

        loss = (
            loss_student.detach()
            + self.fake_loss_weight * loss_fake.detach()
            + self.gan_classifier_loss_weight * loss_gan_classifier.detach()
        )
        metrics = {
            "loss": loss.detach(),
            "loss_teacher": loss_teacher.detach(),
            "loss_dm": loss_dm.detach(),
            "loss_fake": loss_fake.detach(),
            "loss_student": loss_student.detach(),
            "loss_gan_gen": loss_gan_gen.detach(),
            "loss_gan_classifier": loss_gan_classifier.detach(),
            "gan_logits_real": gan_logits_real_mean.detach(),
            "gan_logits_fake": gan_logits_fake_mean.detach(),
            "generator_updated": torch.tensor(
                1.0 if compute_student_gradient else 0.0,
                device=device,
                dtype=torch.float32,
            ),
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
        if sampling_style == SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE:
            return source_flowmatch_euler_sampling_loop(
                noise,
                model_fn,
                sampling_steps=int(sampling_steps),
                c=c,
            )
        if sampling_style != DMD2_RENOISE_SAMPLING_STYLE:
            raise ValueError(
                "DMD2FullOfficial eval only supports official DMD2 re-noise sampling "
                f"({DMD2_RENOISE_SAMPLING_STYLE!r}) or source FlowMatchEuler "
                f"({SOURCE_FLOWMATCH_EULER_SAMPLING_STYLE!r}), got {sampling_style!r}"
            )
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
        latents = noise
        trajectory = []
        batch_size = int(latents.shape[0])
        generator = _.get("generator")
        for idx in range(int(sampling_steps)):
            t_curr = 1.0 - float(idx) / float(sampling_steps)
            t_next = 1.0 - float(idx + 1) / float(sampling_steps)
            t = torch.full((batch_size,), t_curr, device=latents.device, dtype=torch.float32)
            velocity = model_fn(latents, t, c)
            x0 = self.flow_x0(latents, t, velocity)
            trajectory.append(x0)
            latents = self.renoise_x0(x0, t_next, generator=generator)
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


def resolved_config_sha256(config: dict) -> str:
    """Return a stable fingerprint for an exact-resume configuration."""
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()


def save_full_checkpoint_model_only(
    *,
    output_dir: str,
    wrapped_model,
    global_step: int,
    rank: int,
    local_rank: int,
    train_state: dict,
    checkpoint_metadata: dict,
    checkpoints_total_limit: int = 0,
    checkpoint_expected_size_gb: float = 0.0,
    checkpoint_preclean_before_save: bool = False,
) -> str:
    """Save FSDP model shards without AdamW optimizer state.

    This keeps evaluation/loading compatibility through model_dcp, train_state,
    and RNG files, but it is not an exact optimizer-resumable checkpoint.
    """
    ckpt_root = Path(output_dir) / "checkpoints"
    paths = get_ckpt_paths(str(ckpt_root), global_step)
    metadata = dict(checkpoint_metadata)
    metadata["checkpoint_format"] = "torch_distributed_checkpoint_sharded_model_only"
    metadata["checkpoint_layout"] = {
        "model_dcp": "sharded FSDP transformer checkpoint",
        "optimizer_dcp": "omitted because train.save_optimizer_state=false",
        "ema_dcp": "optional sharded EMA transformer checkpoint",
        "ema": "optional HF EMA transformer checkpoint",
    }
    metadata["save_optimizer_state"] = False
    try:
        if checkpoint_preclean_before_save:
            maybe_preclean_checkpoints_for_save(
                output_dir=output_dir,
                global_step=global_step,
                checkpoints_total_limit=checkpoints_total_limit,
                expected_size_gb=checkpoint_expected_size_gb,
                rank=rank,
            )
        assert_checkpoint_free_space(
            output_dir=output_dir,
            expected_size_gb=checkpoint_expected_size_gb,
        )
        cuda_barrier(local_rank)
        if is_main_process(rank):
            reset_checkpoint_artifacts(paths)
            paths["step_dir"].mkdir(parents=True, exist_ok=True)
            write_json_atomic(paths["train_state"], train_state)
            write_json_atomic(paths["meta"], metadata)
        cuda_barrier(local_rank)
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(),
            },
            paths["step_dir"] / f"rng_state_rank{rank}.pt",
        )
        cuda_barrier(local_rank)

        dist_cp.save(
            state_dict=fsdp_model_state_dict(wrapped_model.transformer, offload_to_cpu=True),
            storage_writer=dist_cp.FileSystemWriter(str(paths["model_dcp"])),
        )

        if hasattr(wrapped_model, "ema_transformer"):
            ema_obj = wrapped_model.ema_transformer
            if isinstance(ema_obj, FSDP):
                dist_cp.save(
                    state_dict=fsdp_model_state_dict(ema_obj, offload_to_cpu=True),
                    storage_writer=dist_cp.FileSystemWriter(str(paths["ema_dcp"])),
                )
            elif is_main_process(rank):
                paths["ema_hf"].mkdir(parents=True, exist_ok=True)
                ema_obj.transformer.save_pretrained(str(paths["ema_hf"]))

        cuda_barrier(local_rank)
        if is_main_process(rank):
            with open(paths["complete"], "w", encoding="utf-8") as f:
                f.write(f"checkpoint save completed for global_step_{global_step}\n")
            cleanup_old_checkpoints(output_dir, int(checkpoints_total_limit or 0))
        cuda_barrier(local_rank)
        return str(paths["step_dir"])
    except Exception as exc:
        if is_main_process(rank):
            write_failure_marker(paths["failed"], exc)
        raise


def resolve_dmd2_resume_path(output_dir: str, load_checkpoint_path: str) -> str:
    """Resolve a resume path, accepting ``auto_latest`` for Slurm requeues.

    ``latest`` intentionally keeps the shared TwinFlow semantics and raises if
    no checkpoint exists.  ``auto_latest`` is for a fresh Slurm submission that
    may later be requeued: it starts fresh when no *complete* checkpoint exists
    and otherwise resumes only from the latest completed checkpoint.
    """
    value = str(load_checkpoint_path or "").strip()
    if value.lower() != "auto_latest":
        return resolve_resume_path(output_dir, value)

    ckpt_root = Path(output_dir) / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    for path in ckpt_root.glob("global_step_*"):
        if (
            not path.is_dir()
            or not (path / ".save_complete").is_file()
            or (path / ".save_failed").is_file()
        ):
            continue
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        candidates.append((step, path))
    return str(max(candidates, key=lambda item: item[0])[1]) if candidates else ""


def get_dmd2_full_checkpoint_paths(output_dir: str, global_step: int) -> dict:
    """Return the complete state layout for an exact DMD2 continuation."""
    paths = get_ckpt_paths(str(Path(output_dir) / "checkpoints"), global_step)
    step_dir = paths["step_dir"]
    paths.update(
        {
            "fake_model_dcp": step_dir / "fake_model_dcp",
            "fake_optimizer_dcp": step_dir / "fake_optimizer_dcp",
            "gan_state": step_dir / "gan_classifier.pt",
        }
    )
    return paths


def save_dmd2_full_training_checkpoint(
    *,
    output_dir: str,
    wrapped_model,
    optimizer,
    fake_wrapped_model,
    fake_optimizer,
    gan_classifier_ddp,
    gan_optimizer,
    global_step: int,
    rank: int,
    local_rank: int,
    train_state: dict,
    checkpoint_metadata: dict,
    checkpoints_total_limit: int = 0,
    checkpoint_expected_size_gb: float = 0.0,
    checkpoint_preclean_before_save: bool = False,
) -> str:
    """Save all mutable DMD2 state required for an exact FSDP continuation.

    The DMD2 student, fake critic, GAN head, their independent AdamW states,
    train cursor, and every rank's RNG state are all checkpointed before the
    completion marker is written.  A requeued Slurm job must only resume from a
    directory carrying that marker.
    """
    paths = get_dmd2_full_checkpoint_paths(output_dir, global_step)
    metadata = dict(checkpoint_metadata)
    metadata["checkpoint_format"] = "dmd2_full_training_state_fsdp"
    metadata["checkpoint_layout"] = {
        "model_dcp": "sharded FSDP student transformer",
        "optimizer_dcp": "sharded FSDP student AdamW",
        "fake_model_dcp": "sharded FSDP fake critic transformer",
        "fake_optimizer_dcp": "sharded FSDP fake critic AdamW",
        "gan_state": "rank-zero DDP GAN classifier and AdamW state" if gan_classifier_ddp is not None else "omitted: GAN disabled",
        "train_state": "global step and data cursor",
        "rng_state_rankN": "per-rank Python/NumPy/PyTorch RNG",
    }
    metadata["save_optimizer_state"] = True
    metadata["resume_capability"] = "exact_student_fake_gan_optimizer_rng"
    try:
        if checkpoint_preclean_before_save:
            maybe_preclean_checkpoints_for_save(
                output_dir=output_dir,
                global_step=global_step,
                checkpoints_total_limit=checkpoints_total_limit,
                expected_size_gb=checkpoint_expected_size_gb,
                rank=rank,
            )
        assert_checkpoint_free_space(
            output_dir=output_dir,
            expected_size_gb=checkpoint_expected_size_gb,
        )
        cuda_barrier(local_rank)
        if is_main_process(rank):
            reset_checkpoint_artifacts(paths)
            paths["step_dir"].mkdir(parents=True, exist_ok=True)
            write_json_atomic(paths["train_state"], train_state)
            write_json_atomic(paths["meta"], metadata)
        cuda_barrier(local_rank)
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(),
            },
            paths["step_dir"] / f"rng_state_rank{rank}.pt",
        )
        cuda_barrier(local_rank)

        dist_cp.save(
            state_dict=fsdp_model_state_dict(wrapped_model.transformer, offload_to_cpu=True),
            storage_writer=dist_cp.FileSystemWriter(str(paths["model_dcp"])),
        )
        dist_cp.save(
            state_dict=fsdp_optimizer_state_dict(wrapped_model.transformer, optimizer, offload_to_cpu=True),
            storage_writer=dist_cp.FileSystemWriter(str(paths["optimizer_dcp"])),
        )
        dist_cp.save(
            state_dict=fsdp_model_state_dict(fake_wrapped_model.transformer, offload_to_cpu=True),
            storage_writer=dist_cp.FileSystemWriter(str(paths["fake_model_dcp"])),
        )
        dist_cp.save(
            state_dict=fsdp_optimizer_state_dict(
                fake_wrapped_model.transformer,
                fake_optimizer,
                offload_to_cpu=True,
            ),
            storage_writer=dist_cp.FileSystemWriter(str(paths["fake_optimizer_dcp"])),
        )

        if gan_classifier_ddp is not None:
            if gan_optimizer is None:
                raise RuntimeError("GAN classifier exists but GAN optimizer is missing")
            if is_main_process(rank):
                torch.save(
                    {
                        "model": gan_classifier_ddp.module.state_dict(),
                        "optimizer": gan_optimizer.state_dict(),
                    },
                    paths["gan_state"],
                )

        cuda_barrier(local_rank)
        if is_main_process(rank):
            with open(paths["complete"], "w", encoding="utf-8") as f:
                f.write(f"complete DMD2 training checkpoint for global_step_{global_step}\n")
            cleanup_old_checkpoints(output_dir, int(checkpoints_total_limit or 0))
        cuda_barrier(local_rank)
        return str(paths["step_dir"])
    except Exception as exc:
        if is_main_process(rank):
            write_failure_marker(paths["failed"], exc)
        raise


def load_dmd2_full_training_checkpoint(
    *,
    checkpoint_dir: str,
    wrapped_model,
    optimizer,
    fake_wrapped_model,
    fake_optimizer,
    gan_classifier_ddp,
    gan_optimizer,
    rank: int,
    local_rank: int,
    expected_world_size: int,
    strict_world_size: bool,
    expected_config_sha256: str,
) -> dict:
    """Load an exact DMD2 continuation and reject incomplete/legacy state."""
    ckpt_dir = Path(checkpoint_dir)
    complete = ckpt_dir / ".save_complete"
    failed = ckpt_dir / ".save_failed"
    if failed.is_file():
        raise RuntimeError(f"Checkpoint has failure marker: {failed.read_text(encoding='utf-8').strip()}")
    if not complete.is_file():
        raise FileNotFoundError(f"Checkpoint is missing .save_complete: {complete}")

    metadata_path = ckpt_dir / "checkpoint_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing metadata required for exact resume: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_config_sha256 = metadata.get("resolved_config_sha256") if isinstance(metadata, dict) else None
    if not isinstance(saved_config_sha256, str) or not saved_config_sha256:
        raise RuntimeError(
            "Checkpoint lacks resolved_config_sha256 and cannot be resumed as an exact DMD2 state. "
            "Start a fresh run with the current harness."
        )
    if saved_config_sha256 != expected_config_sha256:
        raise RuntimeError(
            "Exact DMD2 resume rejects a changed resolved config: "
            f"checkpoint={saved_config_sha256} current={expected_config_sha256}. "
            "Start a fresh output directory for changed model, method, optimizer, sampler, or data settings."
        )

    train_state = load_train_state(str(ckpt_dir), require_state=True)
    saved_world_size = int(train_state.get("world_size", -1))
    if strict_world_size and saved_world_size != expected_world_size:
        raise RuntimeError(
            "Checkpoint world size must match before loading FSDP shards: "
            f"checkpoint={saved_world_size} current={expected_world_size}"
        )

    required_dirs = {
        "student model": ckpt_dir / "model_dcp",
        "student optimizer": ckpt_dir / "optimizer_dcp",
        "fake critic model": ckpt_dir / "fake_model_dcp",
        "fake critic optimizer": ckpt_dir / "fake_optimizer_dcp",
    }
    missing = [f"{name}={path}" for name, path in required_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint is not a complete DMD2 training state; missing " + ", ".join(missing)
        )

    load_fsdp_model_dcp(wrapped_model.transformer, ckpt_dir / "model_dcp")
    load_fsdp_optimizer_dcp(wrapped_model.transformer, optimizer, ckpt_dir / "optimizer_dcp")
    load_fsdp_model_dcp(fake_wrapped_model.transformer, ckpt_dir / "fake_model_dcp")
    load_fsdp_optimizer_dcp(fake_wrapped_model.transformer, fake_optimizer, ckpt_dir / "fake_optimizer_dcp")

    gan_state_path = ckpt_dir / "gan_classifier.pt"
    if gan_classifier_ddp is not None:
        if gan_optimizer is None:
            raise RuntimeError("GAN classifier exists but GAN optimizer is missing")
        if not gan_state_path.is_file():
            raise FileNotFoundError(f"Missing DMD2 GAN classifier state: {gan_state_path}")
        gan_state = torch.load(gan_state_path, map_location=f"cuda:{local_rank}", weights_only=False)
        if not isinstance(gan_state, dict) or "model" not in gan_state or "optimizer" not in gan_state:
            raise RuntimeError(f"Invalid DMD2 GAN classifier state: {gan_state_path}")
        gan_classifier_ddp.module.load_state_dict(gan_state["model"], strict=True)
        gan_optimizer.load_state_dict(gan_state["optimizer"])
    elif gan_state_path.exists():
        raise RuntimeError(
            "Checkpoint contains GAN state but the current config disables GAN; refusing incompatible resume"
        )

    load_rng_state(str(ckpt_dir), rank, require_rng_state=True)
    cuda_barrier(local_rank)
    return train_state


def validate_config(config: dict) -> None:
    if config["method"].get("method_type") != "DMD2FullOfficial":
        raise ValueError(f"method.method_type must be DMD2FullOfficial, got {config['method'].get('method_type')}")
    if config["model"].get("model_name") != "QwenImageEdit":
        raise ValueError(f"model.model_name must be QwenImageEdit, got {config['model'].get('model_name')}")
    data_cfg = config.get("data") or {}
    require_nonempty_str_list(data_cfg.get("train_jsonl"), "data.train_jsonl")
    if float((config.get("sample") or {}).get("cfg_scale", 0.0)) != 0.0:
        raise ValueError("sample.cfg_scale must be 0 for FireRed gray DMD2")
    if float((config.get("eval") or {}).get("cfg_scale", 0.0)) != 0.0:
        raise ValueError("eval.cfg_scale must be 0 for FireRed gray DMD2")
    if int(config["train"].get("checkpoints_total_limit", 0)) != 1:
        raise ValueError("train.checkpoints_total_limit must be 1 for this run")
    if int(config["train"].get("save_every", 0)) != 250:
        raise ValueError("train.save_every must be 250")
    if int(config["train"].get("grad_accumulation_steps", 1)) != 1:
        raise ValueError("DMD2FullOfficial currently requires train.grad_accumulation_steps=1")
    if "checkpoint_mode" not in config["train"]:
        raise ValueError(
            "train.checkpoint_mode must be explicit: use model_only_eval for inference-only "
            "checkpoints or full_training_state for exact resume"
        )
    checkpoint_mode = str(config["train"]["checkpoint_mode"])
    if checkpoint_mode not in {"model_only_eval", "full_training_state"}:
        raise ValueError(
            "train.checkpoint_mode must be model_only_eval or full_training_state, "
            f"got {checkpoint_mode!r}"
        )
    load_checkpoint_path = str(config["train"].get("load_checkpoint_path", "") or "").strip()
    save_optimizer_state = bool(config["train"].get("save_optimizer_state", False))
    if checkpoint_mode == "model_only_eval":
        if load_checkpoint_path:
            raise ValueError(
                "model_only_eval checkpoints cannot resume the DMD2 fake critic/GAN state. "
                "Use train.checkpoint_mode=full_training_state with a complete checkpoint."
            )
        if save_optimizer_state:
            raise ValueError(
                "model_only_eval must set train.save_optimizer_state=false; use full_training_state "
                "to save student, fake critic, GAN, optimizer, cursor, and RNG state."
            )
    else:
        if not save_optimizer_state:
            raise ValueError("full_training_state requires train.save_optimizer_state=true")
        if load_checkpoint_path.lower() != "auto_latest":
            raise ValueError(
                "full_training_state requires train.load_checkpoint_path=auto_latest for requeue-safe "
                "fresh-or-resume behavior."
            )
        if bool(config["train"].get("checkpoint_preclean_before_save", False)):
            raise ValueError(
                "full_training_state must set train.checkpoint_preclean_before_save=false. "
                "Keep the prior complete state until the replacement checkpoint has its .save_complete marker."
            )
    if float(config["train"].get("lr", 0.0)) <= 0.0:
        raise ValueError(f"train.lr must be positive, got {config['train'].get('lr')}")
    if float(config["train"].get("fake_lr", config["train"].get("lr", 0.0))) <= 0.0:
        raise ValueError(f"train.fake_lr must be positive, got {config['train'].get('fake_lr')}")
    method_cfg = config.get("method") or {}
    if int(method_cfg.get("student_train_sampling_steps", 1)) <= 0:
        raise ValueError("method.student_train_sampling_steps must be positive")
    if str(method_cfg.get("critic_mode", "")) != "separate_full":
        raise ValueError("method.critic_mode must be separate_full")
    if str(method_cfg.get("student_train_backprop_mode", "single_step")) not in {"single_step", "full_rollout"}:
        raise ValueError("method.student_train_backprop_mode must be single_step or full_rollout")
    if float(method_cfg.get("real_guidance_scale", method_cfg.get("train_cfg_scale", 0.0))) <= 1.0:
        raise ValueError("method.real_guidance_scale must be > 1.0")
    if float(method_cfg.get("fake_guidance_scale", 1.0)) != 1.0:
        raise ValueError("method.fake_guidance_scale must be 1.0")
    if int(method_cfg.get("dfake_gen_update_ratio", 1)) <= 0:
        raise ValueError("method.dfake_gen_update_ratio must be positive")
    teacher_match_loss_weight = float(method_cfg.get("teacher_match_loss_weight", 0.0))
    if teacher_match_loss_weight != 0.0:
        raise ValueError(
            "method.teacher_match_loss_weight is not implemented in DMD2FullOfficial; "
            "set it to 0.0 to avoid assuming a teacher MSE loss is active."
        )
    gan_loss_weight = float(method_cfg.get("gan_loss_weight", 0.0))
    gan_classifier_loss_weight = float(method_cfg.get("gan_classifier_loss_weight", 0.0))
    if gan_loss_weight < 0.0 or gan_classifier_loss_weight < 0.0:
        raise ValueError("method.gan_loss_weight and method.gan_classifier_loss_weight must be non-negative")
    if gan_loss_weight > 0.0 and gan_classifier_loss_weight <= 0.0:
        raise ValueError("method.gan_classifier_loss_weight must be > 0 when method.gan_loss_weight > 0")
    if (gan_loss_weight > 0.0 or gan_classifier_loss_weight > 0.0) and str(
        method_cfg.get("gan_classifier_type", "qwen_hidden_state")
    ) != "qwen_hidden_state":
        raise ValueError("method.gan_classifier_type must be qwen_hidden_state when GAN is enabled")
    if int(method_cfg.get("gan_classifier_hidden_channels", 128)) <= 0:
        raise ValueError("method.gan_classifier_hidden_channels must be positive")
    gan_feature_layer = method_cfg.get("gan_feature_layer", "middle")
    if isinstance(gan_feature_layer, str) and gan_feature_layer.lower() not in {"middle", "mid"}:
        try:
            int(gan_feature_layer)
        except ValueError as exc:
            raise ValueError(
                "method.gan_feature_layer must be an integer layer index or 'middle', "
                f"got {gan_feature_layer!r}"
            ) from exc
    gan_noise_t_min = float(method_cfg.get("gan_noise_t_min", 0.0))
    gan_noise_t_max = float(method_cfg.get("gan_noise_t_max", 0.98))
    if gan_noise_t_min < 0.0 or gan_noise_t_max > 1.0 or gan_noise_t_min > gan_noise_t_max:
        raise ValueError("method.gan_noise_t_min/max must satisfy 0 <= min <= max <= 1")
    if not config["data"].get("uncond_embedding_field"):
        raise ValueError("DMD2FullOfficial requires data.uncond_embedding_field")
    sample_cfg = config.get("sample") or {}
    if str(sample_cfg.get("sampling_style", "")) != DMD2_RENOISE_SAMPLING_STYLE:
        raise ValueError(f"sample.sampling_style must be {DMD2_RENOISE_SAMPLING_STYLE}")
    eval_cfg = config.get("eval") or {}
    if int(eval_cfg.get("every_steps", 0)) != 250:
        raise ValueError("eval.every_steps must be 250")
    if bool(eval_cfg.get("enabled", False)):
        require_nonempty_str_list(eval_cfg.get("eval_jsonl"), "eval.eval_jsonl")
    if str(eval_cfg.get("sampling_style", "")) != DMD2_RENOISE_SAMPLING_STYLE:
        raise ValueError(f"eval.sampling_style must be {DMD2_RENOISE_SAMPLING_STYLE}")
    for item in eval_cfg.get("variants", []):
        if str(item.get("sampling_style", "")) != DMD2_RENOISE_SAMPLING_STYLE:
            raise ValueError(f"DMD2 full eval variants must use sampling_style={DMD2_RENOISE_SAMPLING_STYLE}: {item}")


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


def get_qwen_core_transformer(model_fn):
    module = getattr(model_fn, "module", model_fn)
    transformer = getattr(module, "transformer", None)
    if transformer is None:
        raise RuntimeError(f"Could not resolve GenTransformer.transformer from {type(model_fn).__name__}")
    if not hasattr(transformer, "transformer_blocks"):
        raise RuntimeError(f"Resolved transformer has no transformer_blocks: {type(transformer).__name__}")
    return transformer


def infer_qwen_dims(transformer) -> tuple[int, int, int]:
    blocks = getattr(transformer, "transformer_blocks", None)
    num_layers = len(blocks) if blocks is not None else 0
    config = getattr(transformer, "config", None)
    hidden_dim = int(getattr(transformer, "inner_dim", 0) or 0)
    if hidden_dim <= 0 and config is not None:
        hidden_dim = int(getattr(config, "num_attention_heads", 0)) * int(getattr(config, "attention_head_dim", 0))
    prompt_dim = int(getattr(config, "joint_attention_dim", 0) or 0)
    if num_layers <= 0:
        raise RuntimeError("Qwen transformer has no blocks; cannot install GAN feature hook")
    if hidden_dim <= 0:
        raise RuntimeError("Could not infer Qwen transformer hidden dim for GAN classifier")
    if prompt_dim <= 0:
        raise RuntimeError("Could not infer Qwen prompt/joint attention dim for GAN classifier")
    return hidden_dim, prompt_dim, num_layers


def resolve_gan_feature_layer(value, num_layers: int) -> int:
    if value is None or (isinstance(value, str) and value.lower() in {"middle", "mid"}):
        return num_layers // 2
    idx = int(value)
    if idx < 0:
        idx += num_layers
    if idx < 0 or idx >= num_layers:
        raise ValueError(f"method.gan_feature_layer out of range: {value!r}, num_layers={num_layers}")
    return idx


def require_nonempty_str_list(value, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty YAML list")
    result = []
    for idx, item in enumerate(value):
        text = str(item).strip() if item is not None else ""
        if not text:
            raise ValueError(f"{label}[{idx}] must be a non-empty string")
        result.append(text)
    return result


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
    method = DMD2FullOfficialMethod(method_cfg)

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
    config_sha256 = resolved_config_sha256(config)

    dtype = torch.bfloat16
    checkpoint_metadata = {
        "checkpoint_format": "torch_distributed_checkpoint_sharded_model_only",
        "config_path": os.path.abspath(config_path),
        "fsdp_backend": fsdp_backend,
        "fsdp_use_orig_params": fsdp_use_orig_params,
        "model_name": config["model"]["model_name"],
        "model_path": config["model"]["model_path"],
        "method_type": "DMD2FullOfficial",
        "resolved_config_sha256": config_sha256,
        "critic_mode": method_cfg.get("critic_mode", "separate_full"),
        "student_train_sampling_steps": int(method_cfg.get("student_train_sampling_steps", 1)),
        "student_train_backprop_mode": str(method_cfg.get("student_train_backprop_mode", "single_step")),
        "real_guidance_scale": float(method_cfg.get("real_guidance_scale", method_cfg.get("train_cfg_scale", 0.0))),
        "fake_guidance_scale": float(method_cfg.get("fake_guidance_scale", 1.0)),
        "dfake_gen_update_ratio": int(method_cfg.get("dfake_gen_update_ratio", 1)),
        "teacher_match_loss_weight": float(method_cfg.get("teacher_match_loss_weight", 0.0)),
        "dm_loss_weight": float(method_cfg.get("dm_loss_weight", 1.0)),
        "fake_loss_weight": float(method_cfg.get("fake_loss_weight", 1.0)),
        "gan_loss_weight": float(method_cfg.get("gan_loss_weight", 0.0)),
        "gan_classifier_loss_weight": float(method_cfg.get("gan_classifier_loss_weight", 0.0)),
        "gan_classifier_type": str(method_cfg.get("gan_classifier_type", "qwen_hidden_state")),
        "gan_classifier_hidden_channels": int(method_cfg.get("gan_classifier_hidden_channels", 128)),
        "gan_feature_layer_requested": method_cfg.get("gan_feature_layer", "middle"),
        "gan_noise_t_min": float(method_cfg.get("gan_noise_t_min", 0.0)),
        "gan_noise_t_max": float(method_cfg.get("gan_noise_t_max", 0.98)),
    }

    wrapped_model = MODELS[config["model"]["model_name"]](
        model_id=config["model"]["model_path"],
        aux_time_embed=bool(config["model"].get("aux_time_embed", False)),
        text_dtype=dtype,
        imgs_dtype=dtype,
    )
    vae_z_dim = int(getattr(wrapped_model.model.vae.config, "z_dim", 0))
    if vae_z_dim <= 0:
        raise RuntimeError(f"Could not infer FireRed latent channels from VAE config: z_dim={vae_z_dim}")
    gan_feature_forwarder = None
    gan_classifier_ddp = None
    gan_classifier_module = None
    gan_classifier_params = []
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

    def drop_vae_for_transformer_only(model_obj, role: str) -> None:
        vae = getattr(getattr(model_obj, "model", None), "vae", None)
        if vae is not None:
            model_obj.model.vae = None
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            if is_main_process(rank):
                logger.info("Dropped VAE from %s transformer-only model", role)

    def build_transformer_only(role: str, *, trainable: bool):
        model_obj = MODELS[config["model"]["model_name"]](
            model_id=config["model"]["model_path"],
            aux_time_embed=bool(config["model"].get("aux_time_embed", False)),
            text_dtype=dtype,
            imgs_dtype=dtype,
        )
        maybe_drop_text_encoder(model_obj, True)
        drop_vae_for_transformer_only(model_obj, role)
        model_obj.transformer.requires_grad_(trainable)
        if trainable and bool(config["train"].get("gradient_checkpointing", True)):
            use_reentrant = bool(config["train"].get("gradient_checkpointing_use_reentrant", False))
            model_obj.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
        model_obj.transformer = wrap_fsdp1(model_obj.transformer)
        if trainable:
            model_obj.transformer.train()
        else:
            model_obj.transformer.eval()
        return model_obj

    wrapped_model.transformer = wrap_fsdp1(wrapped_model.transformer)
    transformer_ddp = wrapped_model.transformer
    trainable_params = [p for p in transformer_ddp.module.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable full transformer parameters found")
    transformer_ddp.train()

    fake_wrapped_model = build_transformer_only("fake_critic", trainable=True)
    fake_transformer_ddp = fake_wrapped_model.transformer
    fake_trainable_params = [p for p in fake_transformer_ddp.module.parameters() if p.requires_grad]
    if not fake_trainable_params:
        raise RuntimeError("No trainable fake critic transformer parameters found")

    if method.uses_gan:
        qwen_transformer = get_qwen_core_transformer(fake_transformer_ddp)
        qwen_hidden_dim, qwen_prompt_dim, qwen_num_layers = infer_qwen_dims(qwen_transformer)
        gan_feature_layer = resolve_gan_feature_layer(method_cfg.get("gan_feature_layer", "middle"), qwen_num_layers)
        gan_classifier_module = QwenHiddenStateGANClassifier(
            transformer_hidden_dim=qwen_hidden_dim,
            prompt_dim=qwen_prompt_dim,
            hidden_channels=int(method_cfg.get("gan_classifier_hidden_channels", 128)),
        ).to(device=device)
        gan_classifier_ddp = DDP(
            gan_classifier_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        gan_classifier_ddp.train()
        gan_classifier_params = [p for p in gan_classifier_ddp.module.parameters() if p.requires_grad]
        if not gan_classifier_params:
            raise RuntimeError("GAN loss is enabled but no trainable GAN classifier parameters were found")
        gan_feature_forwarder = QwenHiddenStateGANForwarder(
            model_fn=fake_transformer_ddp,
            classifier=gan_classifier_ddp,
            feature_layer_idx=gan_feature_layer,
        )
        checkpoint_metadata["gan_classifier_type"] = "qwen_hidden_state"
        checkpoint_metadata["gan_feature_layer"] = int(gan_feature_layer)
        checkpoint_metadata["gan_qwen_num_layers"] = int(qwen_num_layers)
        checkpoint_metadata["gan_qwen_hidden_dim"] = int(qwen_hidden_dim)
        checkpoint_metadata["gan_qwen_prompt_dim"] = int(qwen_prompt_dim)
        if is_main_process(rank):
            logger.info(
                "GAN classifier enabled: type=qwen_hidden_state layer=%d/%d hidden_dim=%d prompt_dim=%d head_channels=%d",
                gan_feature_layer,
                qwen_num_layers,
                qwen_hidden_dim,
                qwen_prompt_dim,
                int(method_cfg.get("gan_classifier_hidden_channels", 128)),
            )
    teacher_wrapped_model = build_transformer_only("real_teacher", trainable=False)
    teacher_transformer_ddp = teacher_wrapped_model.transformer
    if any(p.requires_grad for p in teacher_transformer_ddp.module.parameters()):
        raise RuntimeError("Frozen real teacher unexpectedly has trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["train"]["lr"]),
        betas=tuple(config["train"].get("betas", [0.9, 0.99])),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        foreach=True,
    )
    fake_optimizer = torch.optim.AdamW(
        fake_trainable_params,
        lr=float(config["train"].get("fake_lr", config["train"]["lr"])),
        betas=tuple(config["train"].get("betas", [0.9, 0.99])),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        foreach=True,
    )
    gan_optimizer = None
    if gan_classifier_params:
        gan_optimizer = torch.optim.AdamW(
            gan_classifier_params,
            lr=float(config["train"].get("fake_lr", config["train"]["lr"])),
            betas=tuple(config["train"].get("betas", [0.9, 0.99])),
            weight_decay=float(config["train"].get("weight_decay", 0.0)),
            foreach=True,
        )

    checkpoint_mode = str(config["train"].get("checkpoint_mode", "model_only_eval"))
    resume_path = resolve_dmd2_resume_path(
        config["train"]["output_dir"],
        str(config["train"].get("load_checkpoint_path", "") or ""),
    )
    initial_global_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    resume_micro_step = 0
    if resume_path:
        if checkpoint_mode != "full_training_state":
            raise RuntimeError(
                "DMD2FullOfficial refuses to resume a model-only checkpoint because fake critic/GAN "
                "training state would be missing."
            )
        resume_state = load_dmd2_full_training_checkpoint(
            checkpoint_dir=resume_path,
            wrapped_model=wrapped_model,
            optimizer=optimizer,
            fake_wrapped_model=fake_wrapped_model,
            fake_optimizer=fake_optimizer,
            gan_classifier_ddp=gan_classifier_ddp,
            gan_optimizer=gan_optimizer,
            rank=rank,
            local_rank=local_rank,
            expected_world_size=world_size,
            strict_world_size=bool(config["train"].get("resume_strict_world_size", True)),
            expected_config_sha256=config_sha256,
        )
        initial_global_step = int(resume_state["global_step"])
        start_epoch = int(resume_state.get("epoch", 0))
        resume_step_in_epoch = int(resume_state.get("next_step_in_epoch", 0))
        resume_micro_step = int(resume_state.get("micro_step", 0))
        if is_main_process(rank):
            logger.info(
                "Resumed exact DMD2 training state: checkpoint=%s global_step=%d epoch=%d "
                "next_step_in_epoch=%d micro_step=%d world_size=%d",
                resume_path,
                initial_global_step,
                start_epoch,
                resume_step_in_epoch,
                resume_micro_step,
                world_size,
            )

    micro_batch_size = int(config["train"]["micro_batch_size"])
    train_jsonl = require_nonempty_str_list(config["data"].get("train_jsonl"), "data.train_jsonl")
    dataset = FireRedEditJsonlDataset(
        jsonl_files=train_jsonl,
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

    def make_train_dataloader(skip_batches: int = 0, epoch: int = 0):
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
            # DataLoader iterator construction must not consume the restored
            # process RNG. Recreate this epoch/rank-specific generator on resume.
            generator=torch.Generator().manual_seed(int(config["train"].get("seed", 42)) + rank * 1000003 + epoch),
        )

    eval_config = dict(config.get("eval", {}) or {})
    eval_dataloader = None
    if bool(eval_config.get("enabled", False)):
        eval_jsonl = require_nonempty_str_list(eval_config.get("eval_jsonl"), "eval.eval_jsonl")
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
            generator=torch.Generator().manual_seed(int(config["train"].get("seed", 42)) + rank * 1000003 + 50000000),
        )

    max_train_steps = int(config["train"].get("max_train_steps", base_steps_per_epoch))
    grad_accum_steps = int(config["train"].get("grad_accumulation_steps", 1))
    save_every = int(config["train"].get("save_every", 500))
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    checkpoints_total_limit = int(config["train"].get("checkpoints_total_limit", 1))
    checkpoint_expected_size_gb = float(config["train"].get("checkpoint_expected_size_gb", 0.0) or 0.0)
    checkpoint_preclean_before_save = bool(config["train"].get("checkpoint_preclean_before_save", True))
    save_final_checkpoint = bool(config["train"].get("save_final_checkpoint", True))
    save_optimizer_state = bool(config["train"].get("save_optimizer_state", False))
    checkpoint_mode = str(config["train"].get("checkpoint_mode", "model_only_eval"))
    checkpoint_metadata["save_optimizer_state"] = save_optimizer_state
    checkpoint_metadata["checkpoint_mode"] = checkpoint_mode
    checkpoint_metadata["checkpoint_format"] = (
        "dmd2_full_training_state_fsdp"
        if checkpoint_mode == "full_training_state"
        else "torch_distributed_checkpoint_sharded_model_only"
    )
    num_train_epochs = int(config["train"].get("num_train_epochs", 1))
    condition_mode = str(config["data"].get("condition_mode", "offline"))
    if condition_mode != "offline":
        raise ValueError("DMD2 full requires data.condition_mode=offline")

    if is_main_process(rank):
        logger.info(
            "Running FireRed DMD2 full FSDP: records=%d world=%d micro_batch=%d steps_per_epoch=%d "
            "max_train_steps=%d output=%s method=%s student_nfe=%d backprop=%s real_cfg=%.3f fake_cfg=%.3f "
            "dfake_ratio=%d lr=%.2e fake_lr=%.2e gan_gen=%.3g gan_cls=%.3g",
            len(dataset),
            world_size,
            micro_batch_size,
            base_steps_per_epoch,
            max_train_steps,
            config["train"]["output_dir"],
            checkpoint_metadata["critic_mode"],
            checkpoint_metadata["student_train_sampling_steps"],
            checkpoint_metadata["student_train_backprop_mode"],
            checkpoint_metadata["real_guidance_scale"],
            checkpoint_metadata["fake_guidance_scale"],
            checkpoint_metadata["dfake_gen_update_ratio"],
            float(config["train"]["lr"]),
            float(config["train"].get("fake_lr", config["train"]["lr"])),
            checkpoint_metadata["gan_loss_weight"],
            checkpoint_metadata["gan_classifier_loss_weight"],
        )
        if checkpoint_mode == "full_training_state":
            logger.info(
                "Checkpoint saving mode: exact full training state (student/fake/GAN/optimizers/cursor/RNG)"
            )
        else:
            logger.info("Checkpoint saving mode: model-only; optimizer state will not be saved")

    global_step = initial_global_step
    micro_step = resume_micro_step
    current_epoch = start_epoch
    next_step_in_epoch = resume_step_in_epoch
    last_checkpoint_step = initial_global_step if global_step >= max_train_steps else -1
    optimizer.zero_grad(set_to_none=True)
    fake_optimizer.zero_grad(set_to_none=True)
    if gan_optimizer is not None:
        gan_optimizer.zero_grad(set_to_none=True)

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
            train_dataloader = make_train_dataloader(skip_batches=skip_batches, epoch=epoch)
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
                if is_sync_step:
                    sync_context = nullcontext()
                else:
                    sync_context = transformer_ddp.no_sync()

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
                        compute_student_gradient = global_step % method.dfake_gen_update_ratio == 0
                        loss, metrics = method.training_step_backward(
                            student_model_fn=transformer_ddp,
                            fake_model_fn=fake_transformer_ddp,
                            teacher_model_fn=teacher_transformer_ddp,
                            gan_classifier_gen_fn=gan_feature_forwarder,
                            gan_classifier_train_fn=gan_feature_forwarder,
                            target_latents=target_latents,
                            source_latents=source_latents,
                            prompt_embeds=prompt_embeds,
                            prompt_mask=prompt_mask,
                            uncond_embeds=uncond_embeds,
                            uncond_mask=uncond_mask,
                            dtype=dtype,
                            backward_scale=1.0 / float(grad_accum_steps),
                            compute_student_gradient=compute_student_gradient,
                            debug_prefix=(
                                f"rank={rank:02d} local={local_rank} step={global_step + 1:08d}"
                                if method.debug_timing
                                else ""
                            ),
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
                if is_sync_step:
                    generator_updated = bool(float(metrics["generator_updated"].detach().item()) > 0.5)
                    if generator_updated:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                        optimizer.step()
                    else:
                        grad_norm = torch.zeros((), device=device)
                    fake_grad_norm = torch.nn.utils.clip_grad_norm_(fake_trainable_params, max_grad_norm)
                    if gan_optimizer is not None:
                        torch.nn.utils.clip_grad_norm_(gan_classifier_params, max_grad_norm)
                    fake_optimizer.step()
                    if gan_optimizer is not None:
                        gan_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    fake_optimizer.zero_grad(set_to_none=True)
                    if gan_optimizer is not None:
                        gan_optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    next_step_in_epoch = step_in_epoch + 1

                    avg = {name: all_reduce_mean(value, world_size) for name, value in metrics.items()}
                    if is_main_process(rank):
                        logger.info(
                            "step=%08d loss=%.6f student=%.6f teacher=%.6f dm=%.6f fake=%.6f "
                            "gan_gen=%.6f gan_cls=%.6f gan_real=%.4f gan_fake=%.4f gen_update=%.0f "
                            "grad_norm=%.4f fake_grad_norm=%.4f step_time=%.2fs uid=%s",
                            global_step,
                            avg["loss"],
                            avg["loss_student"],
                            avg["loss_teacher"],
                            avg["loss_dm"],
                            avg["loss_fake"],
                            avg["loss_gan_gen"],
                            avg["loss_gan_classifier"],
                            avg["gan_logits_real"],
                            avg["gan_logits_fake"],
                            avg["generator_updated"],
                            float(grad_norm),
                            float(fake_grad_norm),
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
                        if checkpoint_mode == "full_training_state":
                            ckpt_path = save_dmd2_full_training_checkpoint(
                                output_dir=config["train"]["output_dir"],
                                wrapped_model=wrapped_model,
                                optimizer=optimizer,
                                fake_wrapped_model=fake_wrapped_model,
                                fake_optimizer=fake_optimizer,
                                gan_classifier_ddp=gan_classifier_ddp,
                                gan_optimizer=gan_optimizer,
                                global_step=global_step,
                                rank=rank,
                                local_rank=local_rank,
                                train_state=current_train_state(),
                                checkpoint_metadata=checkpoint_metadata,
                                checkpoints_total_limit=checkpoints_total_limit,
                                checkpoint_expected_size_gb=checkpoint_expected_size_gb,
                                checkpoint_preclean_before_save=checkpoint_preclean_before_save,
                            )
                        else:
                            ckpt_path = save_full_checkpoint_model_only(
                                output_dir=config["train"]["output_dir"],
                                wrapped_model=wrapped_model,
                                global_step=global_step,
                                rank=rank,
                                local_rank=local_rank,
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
            if checkpoint_mode == "full_training_state":
                ckpt_path = save_dmd2_full_training_checkpoint(
                    output_dir=config["train"]["output_dir"],
                    wrapped_model=wrapped_model,
                    optimizer=optimizer,
                    fake_wrapped_model=fake_wrapped_model,
                    fake_optimizer=fake_optimizer,
                    gan_classifier_ddp=gan_classifier_ddp,
                    gan_optimizer=gan_optimizer,
                    global_step=global_step,
                    rank=rank,
                    local_rank=local_rank,
                    train_state=current_train_state(),
                    checkpoint_metadata=checkpoint_metadata,
                    checkpoints_total_limit=checkpoints_total_limit,
                    checkpoint_expected_size_gb=checkpoint_expected_size_gb,
                    checkpoint_preclean_before_save=checkpoint_preclean_before_save,
                )
            else:
                ckpt_path = save_full_checkpoint_model_only(
                    output_dir=config["train"]["output_dir"],
                    wrapped_model=wrapped_model,
                    global_step=global_step,
                    rank=rank,
                    local_rank=local_rank,
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
