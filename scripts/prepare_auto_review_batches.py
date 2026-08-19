"""Regroup auto-labeled images into larger MakeSense review batches."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
DEFAULT_SOURCE = DEFAULT_ROOT / "annotation_batches" / "detection_v1"
DEFAULT_OUTPUT = DEFAULT_ROOT / "annotation_batches" / "detection_v2_200"
CLASSES = PROJECT / "dataset" / "tea_dataset_v1_reviewed" / "classes.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="将预标注图片整理为每批200张的检查批次")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {output}")
    if args.batch_size <= 0:
        raise ValueError("batch-size必须大于0")

    records: list[dict[str, str]] = []
    for batch in sorted(source.glob("batch_*")):
        images_dir = batch / "01_images"
        labels_dir = batch / "02_auto_labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        batch_manifest = batch / "batch_manifest.csv"
        metadata: dict[str, dict[str, str]] = {}
        if batch_manifest.exists():
            with batch_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    metadata[row.get("image_name", "")] = row
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"缺少预标注: {label}")
            row = dict(metadata.get(image.name, {}))
            row.update({"source_batch": batch.name, "source_image": str(image), "source_label": str(label)})
            records.append(row)
    if not records:
        raise ValueError("没有发现可检查的预标注图片")

    class_text = CLASSES.read_text(encoding="utf-8-sig")
    summaries: list[dict[str, object]] = []
    for offset in range(0, len(records), args.batch_size):
        selected = records[offset:offset + args.batch_size]
        number = offset // args.batch_size + 1
        batch = output / f"batch_{number:03d}"
        images_dir = batch / "01_images"
        labels_dir = batch / "02_auto_labels"
        corrected_dir = batch / "03_corrected_export"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        corrected_dir.mkdir(parents=True, exist_ok=True)
        (batch / "labels.txt").write_text(class_text, encoding="utf-8")
        rows: list[dict[str, str]] = []
        for record in selected:
            image = Path(record["source_image"])
            label = Path(record["source_label"])
            image_target = images_dir / image.name
            label_target = labels_dir / label.name
            link_or_copy(image, image_target)
            link_or_copy(label, label_target)
            rows.append({
                "image_name": image.name,
                "auto_label_name": label.name,
                "source_batch": record.get("source_batch", ""),
                "candidate_path": record.get("candidate_path", ""),
                "source_relative_path": record.get("source_relative_path", ""),
                "session_id": record.get("session_id", ""),
                "frame_index": record.get("frame_index", ""),
                "candidate_kind": record.get("candidate_kind", ""),
                "group": record.get("group", ""),
                "review_status": "pending_manual_check",
            })
        fields = list(rows[0].keys())
        with (batch / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summaries.append({"batch": batch.name, "images": len(rows), "status": "pending_manual_check"})

    with (output / "batches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "images", "status"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"整理完成: {len(records)}张，{len(summaries)}批 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
