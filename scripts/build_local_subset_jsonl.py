import argparse
import json
from pathlib import Path


def resolve_path(raw_path: str, local_data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return local_data_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--local-data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--required-fields",
        nargs="+",
        default=["source_image", "edit_image", "embeddings_tensor_en", "embeddings_tensor_droptext"],
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).expanduser()
    local_data_root = Path(args.local_data_root).expanduser()
    out_path = Path(args.out).expanduser()

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"jsonl missing: {jsonl_path}")
    if not local_data_root.is_dir():
        raise FileNotFoundError(f"local data root missing: {local_data_root}")

    kept = []
    scanned = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            ok = True
            for field in args.required_fields:
                if field not in row:
                    ok = False
                    break
                if not resolve_path(str(row[field]), local_data_root).exists():
                    ok = False
                    break
            if ok:
                kept.append(row)
            if args.max_records > 0 and len(kept) >= args.max_records:
                break

    if not kept:
        raise RuntimeError(f"No fully local records found in {jsonl_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote={out_path}")
    print(f"records={len(kept)}")
    print(f"scanned={scanned}")


if __name__ == "__main__":
    main()

