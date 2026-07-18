import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def fail(message: str) -> None:
    raise SystemExit(f"[ERR] {message}")


def require_path(path: str, label: str, is_file: bool | None = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        fail(f"{label} missing: {resolved}")
    if is_file is True and not resolved.is_file():
        fail(f"{label} must be a file: {resolved}")
    if is_file is False and not resolved.is_dir():
        fail(f"{label} must be a directory: {resolved}")
    return resolved


def load_jsonl_head(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON at {path}:{line_no}: {exc}")
            if len(rows) >= limit:
                break
    if not rows:
        fail(f"JSONL has no records: {path}")
    return rows


def resolve_data_path(raw_path: str, root: Path, local_data_root: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if local_data_root is not None:
        candidate = local_data_root / path
        if candidate.exists():
            return candidate
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def validate_legacy_local_protocol(cfg: dict) -> None:
    dmd2_cfg = cfg.get("dmd2") or {}
    if (
        bool(dmd2_cfg.get("use_gan_classifier", False))
        or float(dmd2_cfg.get("gen_cls_loss_weight", 0.0)) != 0.0
        or float(dmd2_cfg.get("guidance_cls_loss_weight", 0.0)) != 0.0
    ):
        fail(
            "The local LoRA preflight/trainer uses a legacy latent classifier, not the Qwen hidden-state GAN "
            "classifier required by the official DMD2 protocol. Use scripts/train_firered_dmd2_full_fsdp.py "
            "for GAN feature experiments, or disable classifier losses for legacy debugging."
        )
    if bool(dmd2_cfg.get("use_backward_simulation", False)):
        fail(
            "The local LoRA trainer does not implement the official 4NFE DMD2 re-noise rollout. "
            "Use scripts/train_firered_dmd2_full_fsdp.py for the current official protocol."
        )


def is_full_official_config(cfg: dict) -> bool:
    return ((cfg.get("method") or {}).get("method_type") == "DMD2FullOfficial")


def _require_path_list(value, label: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        fail(f"{label} must be a non-empty YAML list")
    paths = []
    for raw in value:
        paths.append(require_path(str(raw), label, is_file=True))
    return paths


def _as_str_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    fail(f"{label} must be a string or list of strings, got {type(value).__name__}")


def _looks_local(raw_path: str) -> bool:
    path = Path(raw_path)
    return path.is_absolute() and path.exists()


def _load_firered_image_utils(project_root: Path):
    utils_path = require_path(str(project_root / "train" / "src" / "utils" / "image_utils.py"), "FireRed image_utils.py", is_file=True)
    spec = importlib.util.spec_from_file_location("_firered_image_utils_preflight", str(utils_path))
    if spec is None or spec.loader is None:
        fail(f"Could not import FireRed image utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_tensor_like(value: Any, label: str) -> None:
    shape = getattr(value, "shape", None)
    dim = value.dim() if hasattr(value, "dim") else None
    if dim != 2 and not (dim == 3 and int(shape[0]) == 1):
        fail(f"{label} must load to rank-2 tensor or [1, seq, dim], got type={type(value).__name__} shape={shape}")
    if any(int(part) <= 0 for part in shape):
        fail(f"{label} loaded an empty tensor: shape={shape}")


def validate_full_official_protocol(cfg: dict, cfg_path: Path, sample_check_count: int) -> None:
    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}
    method_cfg = cfg.get("method") or {}
    train_cfg = cfg.get("train") or {}
    sample_cfg = cfg.get("sample") or {}
    eval_cfg = cfg.get("eval") or {}

    if model_cfg.get("model_name") != "QwenImageEdit":
        fail(f"model.model_name must be QwenImageEdit, got {model_cfg.get('model_name')}")
    model_path = require_path(str(model_cfg.get("model_path", "")), "model.model_path", is_file=False)
    for rel_path in ("model_index.json", "transformer/config.json", "vae/config.json"):
        require_path(str(model_path / rel_path), f"model.model_path/{rel_path}", is_file=True)

    if float(sample_cfg.get("cfg_scale", 0.0)) != 0.0:
        fail(f"sample.cfg_scale must be 0 for FireRed gray DMD2, got {sample_cfg.get('cfg_scale')}")
    if float(eval_cfg.get("cfg_scale", 0.0)) != 0.0:
        fail(f"eval.cfg_scale must be 0 for FireRed gray DMD2, got {eval_cfg.get('cfg_scale')}")
    if str(sample_cfg.get("sampling_style", "")) != "dmd2_renoise":
        fail("sample.sampling_style must be dmd2_renoise")
    if str(eval_cfg.get("sampling_style", "")) != "dmd2_renoise":
        fail("eval.sampling_style must be dmd2_renoise")
    for item in eval_cfg.get("variants", []):
        if str((item or {}).get("sampling_style", "")) != "dmd2_renoise":
            fail(f"eval.variants must use dmd2_renoise: {item}")

    if int(train_cfg.get("save_every", 0)) != 250:
        fail("train.save_every must be 250")
    if int(eval_cfg.get("every_steps", 0)) != 250:
        fail("eval.every_steps must be 250")
    if int(train_cfg.get("checkpoints_total_limit", 0)) != 1:
        fail("train.checkpoints_total_limit must be 1")
    if int(train_cfg.get("grad_accumulation_steps", 1)) != 1:
        fail("train.grad_accumulation_steps must be 1")
    if "checkpoint_mode" not in train_cfg:
        fail(
            "train.checkpoint_mode must be explicit: use model_only_eval for inference-only "
            "checkpoints or full_training_state for exact resume"
        )
    checkpoint_mode = str(train_cfg["checkpoint_mode"])
    if checkpoint_mode not in {"model_only_eval", "full_training_state"}:
        fail(
            "train.checkpoint_mode must be model_only_eval or full_training_state, "
            f"got {checkpoint_mode!r}"
        )
    load_checkpoint_path = str(train_cfg.get("load_checkpoint_path", "") or "").strip()
    save_optimizer_state = bool(train_cfg.get("save_optimizer_state", False))
    if checkpoint_mode == "model_only_eval":
        if load_checkpoint_path:
            fail(
                "model_only_eval checkpoints cannot resume DMD2 fake critic/GAN state; "
                "use checkpoint_mode=full_training_state"
            )
        if save_optimizer_state:
            fail("model_only_eval requires train.save_optimizer_state=false")
    else:
        if not save_optimizer_state:
            fail("full_training_state requires train.save_optimizer_state=true")
        if load_checkpoint_path.lower() != "auto_latest":
            fail("full_training_state requires train.load_checkpoint_path=auto_latest for requeue-safe fresh-or-resume behavior")
        if bool(train_cfg.get("checkpoint_preclean_before_save", False)):
            fail(
                "full_training_state requires train.checkpoint_preclean_before_save=false so the prior "
                "complete state survives until the replacement checkpoint is complete"
            )
    if float(train_cfg.get("lr", 0.0)) <= 0.0:
        fail(f"train.lr must be positive, got {train_cfg.get('lr')}")
    if float(train_cfg.get("fake_lr", train_cfg.get("lr", 0.0))) <= 0.0:
        fail(f"train.fake_lr must be positive, got {train_cfg.get('fake_lr')}")

    if method_cfg.get("critic_mode") != "separate_full":
        fail("method.critic_mode must be separate_full")
    real_guidance_scale = float(method_cfg.get("real_guidance_scale", method_cfg.get("train_cfg_scale", 0.0)))
    if real_guidance_scale <= 1.0:
        fail("method.real_guidance_scale must be > 1.0")
    if float(method_cfg.get("fake_guidance_scale", 1.0)) != 1.0:
        fail("method.fake_guidance_scale must be 1.0")
    if int(method_cfg.get("student_train_sampling_steps", 1)) <= 0:
        fail("method.student_train_sampling_steps must be positive")
    if str(method_cfg.get("student_train_backprop_mode", "single_step")) not in {"single_step", "full_rollout"}:
        fail("method.student_train_backprop_mode must be single_step or full_rollout")
    if int(method_cfg.get("dfake_gen_update_ratio", 1)) <= 0:
        fail("method.dfake_gen_update_ratio must be positive")
    if float(method_cfg.get("teacher_match_loss_weight", 0.0)) != 0.0:
        fail("method.teacher_match_loss_weight is not implemented; set it to 0.0")
    gan_loss_weight = float(method_cfg.get("gan_loss_weight", 0.0))
    gan_classifier_loss_weight = float(method_cfg.get("gan_classifier_loss_weight", 0.0))
    if gan_loss_weight < 0.0 or gan_classifier_loss_weight < 0.0:
        fail("method.gan_loss_weight and method.gan_classifier_loss_weight must be non-negative")
    if gan_loss_weight > 0.0 and gan_classifier_loss_weight <= 0.0:
        fail("method.gan_classifier_loss_weight must be > 0 when GAN loss is enabled")
    if (gan_loss_weight > 0.0 or gan_classifier_loss_weight > 0.0) and method_cfg.get(
        "gan_classifier_type", "qwen_hidden_state"
    ) != "qwen_hidden_state":
        fail("method.gan_classifier_type must be qwen_hidden_state when GAN is enabled")
    if int(method_cfg.get("gan_classifier_hidden_channels", 128)) <= 0:
        fail("method.gan_classifier_hidden_channels must be positive")
    gan_feature_layer = method_cfg.get("gan_feature_layer", "middle")
    if isinstance(gan_feature_layer, str) and gan_feature_layer.lower() not in {"middle", "mid"}:
        try:
            int(gan_feature_layer)
        except ValueError as exc:
            fail(f"method.gan_feature_layer must be an integer layer index or 'middle', got {gan_feature_layer!r}")
    gan_noise_t_min = float(method_cfg.get("gan_noise_t_min", 0.0))
    gan_noise_t_max = float(method_cfg.get("gan_noise_t_max", 0.98))
    if gan_noise_t_min < 0.0 or gan_noise_t_max > 1.0 or gan_noise_t_min > gan_noise_t_max:
        fail("method.gan_noise_t_min/max must satisfy 0 <= min <= max <= 1")

    embedding_field = data_cfg.get("embedding_field")
    uncond_embedding_field = data_cfg.get("uncond_embedding_field")
    if not embedding_field or not uncond_embedding_field:
        fail("data.embedding_field and data.uncond_embedding_field are required")
    for key in ("height", "width"):
        if int(data_cfg.get(key, 0)) <= 0:
            fail(f"data.{key} must be positive")
    if int(data_cfg.get("repeat", 1)) <= 0:
        fail("data.repeat must be positive")
    project_root_raw = str(data_cfg.get("firered_project_root", "") or "").strip()
    if not project_root_raw:
        fail("data.firered_project_root is required")
    firered_project_root = require_path(project_root_raw, "data.firered_project_root", is_file=False)
    image_utils = _load_firered_image_utils(firered_project_root)

    jsonl_paths = _require_path_list(data_cfg.get("train_jsonl"), "data.train_jsonl")
    eval_jsonl_paths = []
    if bool(eval_cfg.get("enabled", False)):
        eval_jsonl_paths = _require_path_list(eval_cfg.get("eval_jsonl"), "eval.eval_jsonl")
        reference_manifest = str(eval_cfg.get("reference_manifest", "") or "").strip()
        if not reference_manifest:
            fail("eval.reference_manifest is required when eval.enabled=true")
        require_path(reference_manifest, "eval.reference_manifest", is_file=True)

    required_fields = ["source_image", "edit_image", str(embedding_field), str(uncond_embedding_field)]
    remote_fields = 0
    rows_checked = 0
    first_row = None
    for jsonl_path in jsonl_paths + eval_jsonl_paths:
        rows = load_jsonl_head(jsonl_path, sample_check_count)
        rows_checked += len(rows)
        if first_row is None:
            first_row = rows[0]
        for idx, row in enumerate(rows):
            for field in required_fields:
                value = row.get(field)
                if value in (None, ""):
                    fail(f"{jsonl_path} row={idx} missing required field={field}")
                if not _looks_local(str(value)):
                    remote_fields += 1
    if remote_fields and os.environ.get("FIRERED_USE_COS", "0") != "1":
        fail(
            "Full-official FireRed data references COS keys. Set FIRERED_USE_COS=1 and export COS credentials, "
            "or run the Slurm wrapper which exports COS credentials before its embedded preflight."
        )

    if first_row is None:
        fail("No records found in train/eval JSONLs")
    source_paths = _as_str_list(first_row.get("source_image"), "source_image")
    if len(source_paths) != 1:
        fail(f"Only one source image is supported, got {len(source_paths)}")
    edit_image = first_row.get("edit_image")
    if not isinstance(edit_image, str):
        fail(f"edit_image must be a string, got {type(edit_image).__name__}")
    source_image = image_utils.load_image(source_paths[0])
    edit_loaded = image_utils.load_image(edit_image)
    if getattr(source_image, "size", None) is None or getattr(edit_loaded, "size", None) is None:
        fail("FireRed image_utils.load_image did not return PIL-like images")
    prompt_tensor = image_utils.load_tensor(str(first_row[str(embedding_field)]))
    uncond_tensor = image_utils.load_tensor(str(first_row[str(uncond_embedding_field)]))
    _assert_tensor_like(prompt_tensor, str(embedding_field))
    _assert_tensor_like(uncond_tensor, str(uncond_embedding_field))

    output_dir_raw = str(train_cfg.get("output_dir", "") or "").strip()
    if not output_dir_raw:
        fail("train.output_dir is required")
    output_dir = Path(output_dir_raw).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024**3)
    min_free_gb = float(train_cfg.get("checkpoint_expected_size_gb", train_cfg.get("min_free_space_gb", 0)) or 0)
    preclean_before_save = bool(train_cfg.get("checkpoint_preclean_before_save", False))
    checkpoints_total_limit = int(train_cfg.get("checkpoints_total_limit", 0) or 0)
    if min_free_gb > 0 and free_gb < min_free_gb:
        if not (preclean_before_save and checkpoints_total_limit > 0):
            fail(f"Insufficient free space at {output_dir}: free={free_gb:.1f}GiB required={min_free_gb:.1f}GiB")
        print(
            "Preflight note: launch-time space is below one full checkpoint; "
            "the training checkpoint path will preclean retained states before its authoritative guard."
        )

    print("Preflight OK")
    print("mode=full_official")
    print(f"config={cfg_path}")
    print(f"model_path={model_path}")
    print(f"jsonl_count={len(jsonl_paths)}")
    print(f"eval_jsonl_count={len(eval_jsonl_paths)}")
    print(f"records_checked={rows_checked}")
    print(f"remote_fields={remote_fields}")
    print(f"free_space_gib={free_gb:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-check-count", type=int, default=8)
    args = parser.parse_args()

    cfg_path = require_path(args.config, "config", is_file=True)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if is_full_official_config(cfg):
        validate_full_official_protocol(cfg, cfg_path, args.sample_check_count)
        return
    validate_legacy_local_protocol(cfg)

    base_model = require_path(cfg["model"]["base_model_path"], "base_model_path", is_file=False)
    require_path(cfg["model"]["merged_gray_model_path"], "merged_gray_model_path", is_file=False)
    teacher_adapter_path = str(cfg["model"].get("teacher_adapter_path", "") or "").strip()
    if teacher_adapter_path:
        adapter_dir = require_path(teacher_adapter_path, "teacher_adapter_path", is_file=False)
        if not ((adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()):
            fail(f"teacher_adapter_path has no adapter_model.safetensors/bin: {adapter_dir}")

    jsonl_path = require_path(cfg["data"]["jsonl"], "data.jsonl", is_file=True)
    data_root = jsonl_path.parents[2] if len(jsonl_path.parents) >= 3 else base_model.parent
    local_data_root = None
    if cfg["data"].get("local_data_root"):
        local_data_root = require_path(cfg["data"]["local_data_root"], "data.local_data_root", is_file=False)
    rows = load_jsonl_head(jsonl_path, args.sample_check_count)
    require_local_files = bool(cfg["data"].get("require_local_files", True))
    allow_cos_fallback = bool(cfg["data"].get("allow_cos_fallback", False))
    if allow_cos_fallback:
        if os.environ.get("FIRERED_USE_COS", "0") != "1":
            fail("data.allow_cos_fallback=true requires FIRERED_USE_COS=1")
        firered_project_root = cfg["data"].get("firered_project_root")
        if not firered_project_root:
            fail("data.firered_project_root is required when allow_cos_fallback=true")
        root = require_path(firered_project_root, "data.firered_project_root", is_file=False)
        require_path(str(root / "train" / "src" / "utils" / "image_utils.py"), "FireRed image_utils.py", is_file=True)

    required_fields = [
        cfg["data"]["source_image_field"],
        cfg["data"]["target_image_field"],
        cfg["data"]["embedding_field"],
        cfg["data"]["uncond_embedding_field"],
    ]
    missing = []
    checked_files = 0
    remote_fallback_files = 0
    for idx, row in enumerate(rows):
        for field in required_fields:
            if field not in row or row[field] in (None, ""):
                missing.append(f"row={idx} field={field}")
                continue
            candidate = resolve_data_path(str(row[field]), data_root, local_data_root=local_data_root)
            if not candidate.exists():
                if require_local_files or not allow_cos_fallback:
                    missing.append(f"row={idx} field={field} path={candidate}")
                else:
                    remote_fallback_files += 1
            else:
                checked_files += 1
    if missing:
        fail("Missing required data files/fields:\n" + "\n".join(missing[:30]))

    if float(cfg["dmd2"].get("cfg_scale", -1)) != 0:
        fail(f"dmd2.cfg_scale must be 0 for FireRed gray, got {cfg['dmd2'].get('cfg_scale')}")
    if float(cfg["eval"].get("cfg_scale", -1)) != 0:
        fail(f"eval.cfg_scale must be 0 for FireRed gray, got {cfg['eval'].get('cfg_scale')}")
    real_guidance_scale = float(cfg["dmd2"].get("real_guidance_scale", -1))
    fake_guidance_scale = float(cfg["dmd2"].get("fake_guidance_scale", -1))
    if real_guidance_scale <= 1.0:
        fail(
            "dmd2.real_guidance_scale must be > 1.0 for CFG distillation; "
            f"got {real_guidance_scale}"
        )
    if fake_guidance_scale != 1.0:
        fail(f"dmd2.fake_guidance_scale must be 1.0 for official DMD2 fake critic; got {fake_guidance_scale}")
    if int(cfg["dmd2"].get("dfake_gen_update_ratio", 0)) <= 0:
        fail("dmd2.dfake_gen_update_ratio must be positive")

    output_dir = Path(cfg["project"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024**3)
    min_free_gb = float(cfg["train"].get("min_free_space_gb", 0))
    if free_gb < min_free_gb:
        fail(f"Insufficient free space at {output_dir}: free={free_gb:.1f}GiB required={min_free_gb:.1f}GiB")

    print("Preflight OK")
    print(f"config={cfg_path}")
    print("teacher_adapter_mode=adapter" if teacher_adapter_path else "teacher_adapter_mode=base_model")
    print(f"jsonl={jsonl_path}")
    print(f"records_checked={len(rows)}")
    print(f"files_checked={checked_files}")
    print(f"remote_fallback_files={remote_fallback_files}")
    print(f"free_space_gib={free_gb:.1f}")


if __name__ == "__main__":
    main()
