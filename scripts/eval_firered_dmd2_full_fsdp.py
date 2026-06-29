#!/usr/bin/env python
import argparse
import gc
import json
import os
import sys
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
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader

HARNESS_ROOT = Path(__file__).resolve().parents[1]
TWINFLOW_SRC = Path("/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")
for path in (HARNESS_ROOT, TWINFLOW_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_firered_dmd2_full_fsdp import DMD2FullSharedMethod  # noqa: E402
from data.firered_edit_jsonl_dataset import FireRedEditJsonlDataset, collate_firered_edit  # noqa: E402
from networks import MODELS  # noqa: E402
from steerers.qwenimage.sft_ddp_lora_firered_edit import (  # noqa: E402
    maybe_drop_text_encoder,
    optional_field,
)
from steerers.qwenimage.sft_fsdp_firered_edit import (  # noqa: E402
    cleanup_distributed,
    get_fsdp_use_orig_params,
    load_full_checkpoint,
    run_offline_eval,
    set_seed,
    setup_distributed,
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
    return parser.parse_args()


def exp_step(checkpoint: str) -> int:
    name = Path(checkpoint).name
    if not name.startswith("global_step_"):
        raise ValueError(f"Checkpoint path must end with global_step_*, got {checkpoint}")
    return int(name.split("_")[-1])


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


def main() -> None:
    args = parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if config["method"].get("method_type") != "DMD2FullShared":
        raise ValueError("This eval script only supports method.method_type=DMD2FullShared")

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
    method = DMD2FullSharedMethod(method_cfg)

    wrapped_model = build_fsdp_model(config, local_rank, dtype)
    load_full_checkpoint(
        checkpoint_dir=args.checkpoint,
        wrapped_model=wrapped_model,
        optimizer=None,
        rank=rank,
        require_optimizer=False,
    )
    wrapped_model.transformer.eval()

    eval_config = dict(config.get("eval", {}) or {})
    eval_config.update(
        {
            "enabled": True,
            "height": int(args.height),
            "width": int(args.width),
            "max_samples": int(args.max_samples),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "cfg_scale": 0.0,
            "sampling_style": "few",
            "variants": [
                {"label": "dmd2_full_1nfe_1024", "sampling_style": "few", "sampling_steps": 1},
                {"label": "dmd2_full_few_2nfe_1024", "sampling_style": "few", "sampling_steps": 2},
                {"label": "dmd2_full_few_4nfe_1024", "sampling_style": "few", "sampling_steps": 4},
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
        contact_sheet = run_offline_eval(
            wrapped_model,
            method,
            eval_dataloader,
            config.get("sample", {}),
            eval_config,
            args.output_dir,
            exp_step(args.checkpoint),
            rank,
            device,
            dtype,
        )
        if rank == 0:
            manifest_path = Path(args.output_dir) / "offline_eval" / f"step_{exp_step(args.checkpoint):08d}" / "eval_1024_invocation.json"
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
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_distributed()


if __name__ == "__main__":
    main()
