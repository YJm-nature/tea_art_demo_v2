"""Prepare CVAT image batches for kettle pose annotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shutil


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_KEY_RE = re.compile(r"__(?P<video_hash>[0-9a-f]{10})__f(?P<frame>\d+)$")


def source_key(stem: str) -> str:
    match = SOURCE_KEY_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot recover source key from: {stem}")
    return f"{match.group('video_hash')}:{int(match.group('frame'))}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare kettle pose CVAT batches")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--output-name", default="kettle_pose_warm_clean_injection_v1"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    root = args.root.resolve()
    action_root = root / "derived" / "action_pool" / "new_front_full_202608"
    output = root / "annotation_batches" / args.output_name
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"Output is not empty; use --force: {output}")
        shutil.rmtree(output)

    selected: dict[str, tuple[Path, str]] = {}
    for action in ("warm_clean", "water_injection"):
        for image in sorted((action_root / action).rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            selected.setdefault(source_key(image.stem), (image, action))
    if not selected:
        raise ValueError("No warm_clean or water_injection images found")

    records: list[dict[str, str]] = []
    for index, (key, (image, action)) in enumerate(sorted(selected.items()), 1):
        batch_number = (index - 1) // args.batch_size + 1
        batch_name = f"batch_{batch_number:03d}"
        image_dir = output / batch_name / "01_images_for_cvat"
        export_dir = output / batch_name / "02_cvat_pose_export"
        image_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dir / image.name)
        records.append({
            "batch": batch_name,
            "image_name": image.name,
            "source_key": key,
            "action": action,
            "annotation_priority": "kettle skeleton: outlet_tip, body_center, handle_center",
        })

    for batch_name in sorted({record["batch"] for record in records}):
        batch_records = [record for record in records if record["batch"] == batch_name]
        manifest = output / batch_name / "batch_manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(batch_records[0]))
            writer.writeheader()
            writer.writerows(batch_records)

    with (output / "all_batches_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print({
        "images": len(records),
        "batches": len({record["batch"] for record in records}),
        "output": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
