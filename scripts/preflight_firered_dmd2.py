import argparse
import json
import shutil
from pathlib import Path

import yaml


def fail(message: str) -> None:
    raise RuntimeError(message)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-check-count", type=int, default=8)
    args = parser.parse_args()

    cfg_path = require_path(args.config, "config", is_file=True)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    base_model = require_path(cfg["model"]["base_model_path"], "base_model_path", is_file=False)
    require_path(cfg["model"]["merged_gray_model_path"], "merged_gray_model_path", is_file=False)
    adapter_dir = require_path(cfg["model"]["teacher_adapter_path"], "teacher_adapter_path", is_file=False)
    if not ((adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()):
        fail(f"teacher_adapter_path has no adapter_model.safetensors/bin: {adapter_dir}")

    jsonl_path = require_path(cfg["data"]["jsonl"], "data.jsonl", is_file=True)
    data_root = jsonl_path.parents[2] if len(jsonl_path.parents) >= 3 else base_model.parent
    local_data_root = None
    if cfg["data"].get("local_data_root"):
        local_data_root = require_path(cfg["data"]["local_data_root"], "data.local_data_root", is_file=False)
    rows = load_jsonl_head(jsonl_path, args.sample_check_count)

    required_fields = [
        cfg["data"]["source_image_field"],
        cfg["data"]["target_image_field"],
        cfg["data"]["embedding_field"],
        cfg["data"]["uncond_embedding_field"],
    ]
    missing = []
    checked_files = 0
    for idx, row in enumerate(rows):
        for field in required_fields:
            if field not in row or row[field] in (None, ""):
                missing.append(f"row={idx} field={field}")
                continue
            candidate = resolve_data_path(str(row[field]), data_root, local_data_root=local_data_root)
            if not candidate.exists():
                missing.append(f"row={idx} field={field} path={candidate}")
            else:
                checked_files += 1
    if missing:
        fail("Missing required data files/fields:\n" + "\n".join(missing[:30]))

    if float(cfg["dmd2"].get("cfg_scale", -1)) != 0:
        fail(f"dmd2.cfg_scale must be 0 for FireRed gray, got {cfg['dmd2'].get('cfg_scale')}")
    if float(cfg["eval"].get("cfg_scale", -1)) != 0:
        fail(f"eval.cfg_scale must be 0 for FireRed gray, got {cfg['eval'].get('cfg_scale')}")

    output_dir = Path(cfg["project"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024**3)
    min_free_gb = float(cfg["train"].get("min_free_space_gb", 0))
    if free_gb < min_free_gb:
        fail(f"Insufficient free space at {output_dir}: free={free_gb:.1f}GiB required={min_free_gb:.1f}GiB")

    print("Preflight OK")
    print(f"config={cfg_path}")
    print(f"jsonl={jsonl_path}")
    print(f"records_checked={len(rows)}")
    print(f"files_checked={checked_files}")
    print(f"free_space_gib={free_gb:.1f}")


if __name__ == "__main__":
    main()
