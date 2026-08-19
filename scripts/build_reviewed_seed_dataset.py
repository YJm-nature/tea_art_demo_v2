"""Build a training snapshot from corrected MakeSense exports.

The source annotation batches are never modified. Corrected ZIP entries take
priority; missing entries from an empty-negative group become empty labels,
otherwise the batch's original auto label is used as a conservative fallback.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = PROJECT / "dataset" / "tea_sop_front_v1" / "annotation_batches" / "detection_v2_100"
DEFAULT_OUTPUT = PROJECT / "dataset" / "tea_sop_front_v1" / "training" / "reviewed600_v1"
DEFAULT_CLASSES = PROJECT / "dataset" / "tea_dataset_v1_reviewed" / "classes.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_zip_labels(zip_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            labels[Path(info.filename).name] = archive.read(info).decode("utf-8-sig")
    return labels


def validate_label_text(text: str, class_count: int, source: str) -> str:
    valid: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{source}:{line_number}: YOLO标签字段数不是5")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if not 0 <= class_id < class_count:
            raise ValueError(f"{source}:{line_number}: 类别ID越界 {class_id}")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{source}:{line_number}: 归一化框坐标越界")
        valid.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
    return "\n".join(valid) + ("\n" if valid else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="整理前六批修订标签为600张训练快照")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--first-batch", type=int, default=1)
    parser.add_argument("--last-batch", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="允许删除并重建已有训练快照")
    args = parser.parse_args()

    batch_root = args.batch_root.resolve()
    output = args.output.resolve()
    classes = args.classes.resolve().read_text(encoding="utf-8-sig").splitlines()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"输出目录非空，使用--force才允许重建: {output}")
        shutil.rmtree(output)

    images_out = output / "images"
    labels_out = output / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    source_counts = {"corrected": 0, "auto_fallback": 0, "empty_negative": 0}
    for number in range(args.first_batch, args.last_batch + 1):
        batch = batch_root / f"batch_{number:03d}"
        image_dir = batch / "01_images"
        auto_dir = batch / "02_auto_labels"
        manifest_path = batch / "batch_manifest.csv"
        zip_files = sorted((batch / "03_corrected_export").glob("*.zip"))
        if not image_dir.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"批次缺少图片或manifest: {batch}")
        corrected = read_zip_labels(zip_files[-1]) if zip_files else {}
        metadata: dict[str, dict[str, str]] = {}
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metadata[row["image_name"]] = row

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_index, image in enumerate(images):
            row = metadata.get(image.name, {})
            output_stem = f"{batch.name}__{image.stem}"
            image_target = images_out / f"{output_stem}{image.suffix.lower()}"
            label_target = labels_out / f"{output_stem}.txt"
            shutil.copy2(image, image_target)

            corrected_name = f"{image.stem}.txt"
            if corrected_name in corrected:
                label_text = corrected[corrected_name]
                label_source = "corrected"
            elif row.get("group") == "empty_negative":
                label_text = ""
                label_source = "empty_negative"
            else:
                auto_path = auto_dir / corrected_name
                label_text = auto_path.read_text(encoding="utf-8-sig") if auto_path.exists() else ""
                label_source = "auto_fallback"
            label_target.write_text(
                validate_label_text(label_text, len(classes), str(label_target)),
                encoding="utf-8",
            )
            source_counts[label_source] += 1
            records.append({
                "image": str(image_target),
                "label": str(label_target),
                "batch": batch.name,
                "image_index": str(image_index),
                "label_source": label_source,
                "original_image": str(image),
            })

    train_lines: list[str] = []
    val_lines: list[str] = []
    for record in records:
        # One deterministic validation image per ten images in each source batch.
        target = val_lines if int(record["image_index"]) % 10 == 0 else train_lines
        target.append(record["image"])

    (output / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        f"nc: {len(classes)}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(classes)],
    ]
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print({"images": len(records), "train": len(train_lines), "val": len(val_lines), **source_counts})
    print(f"训练快照: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
