"""Create a small MakeSense review batch with matching images and YOLO labels."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a MakeSense review batch")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--status",
        choices=["pending", "needs_fix", "accepted", "rejected"],
        default="needs_fix",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("limit must be positive and offset cannot be negative")

    records = [
        record for record in read_manifest(workspace)
        if record["review_status"] == args.status
        and (args.source is None or record["source"] == args.source)
    ]
    records.sort(key=lambda record: (
        record["source"],
        record.get("frame_number") if record.get("frame_number") is not None else -1,
        record["sample_id"],
    ))
    selected = records[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("No samples match this batch filter")

    image_dir = output / "01_images"
    annotation_dir = output / "02_yolo_import"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        source_image = workspace / record["image"]
        source_label = workspace / record["detect_label"]
        _link_or_copy(source_image, image_dir / source_image.name)
        shutil.copy2(source_label, annotation_dir / source_label.name)
    shutil.copy2(workspace / "classes.txt", annotation_dir / "labels.txt")

    report = {
        "workspace": str(workspace),
        "source_filter": args.source,
        "status_filter": args.status,
        "offset": args.offset,
        "images": len(selected),
        "sample_ids": [record["sample_id"] for record in selected],
        "instructions": [
            "Load every image in 01_images into MakeSense.",
            "Import YOLO annotations by pressing Ctrl+A in 02_yolo_import.",
            "Export the corrected annotations as a YOLO ZIP and keep the ZIP as an audit artifact.",
        ],
    }
    (output / "batch_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "sample_ids"},
                     ensure_ascii=False, indent=2))
    return 0


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
