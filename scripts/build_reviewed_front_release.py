"""Build a leak-free YOLO18 release from completed MakeSense batch exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = PROJECT / "dataset" / "tea_sop_front_v1" / "annotation_batches" / "detection_v2_100"
DEFAULT_OUTPUT = PROJECT / "dataset" / "tea_sop_front_v1" / "releases" / "detection" / "front_detect_reviewed_v1"
DEFAULT_CLASSES = PROJECT / "dataset" / "tea_dataset_v1_reviewed" / "classes.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_zip_labels(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(item.filename).name: archive.read(item).decode("utf-8-sig")
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".txt")
        }


def normalize_labels(text: str, class_count: int, source: str) -> str:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{source}:{number}: YOLO标签必须有5列")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not 0 <= class_id < class_count or any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{source}:{number}: 类别或坐标越界")
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def split_for_session(session: str) -> str:
    # Keep every continuous source together. The six new full-front videos are train;
    # the two old independent recordings provide val/test without frame leakage.
    if session == "new_front_full_202608":
        return "train"
    if session == "legacy_utensils_202606":
        return "val"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="整理全部已审核批次为正式YOLO18数据集")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, output = args.batch_root.resolve(), args.output.resolve()
    classes = args.classes.resolve().read_text(encoding="utf-8-sig").splitlines()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"发布目录非空，使用--force重建: {output}")
        shutil.rmtree(output)

    records = []
    for batch in sorted(root.glob("batch_*")):
        if not batch.is_dir():
            continue
        image_dir, manifest_path = batch / "01_images", batch / "batch_manifest.csv"
        zip_files = sorted((batch / "03_corrected_export").glob("*.zip"))
        if not image_dir.is_dir() or not manifest_path.is_file() or not zip_files:
            raise FileNotFoundError(f"{batch}缺少图片、manifest或修订ZIP")
        corrected = read_zip_labels(zip_files[-1])
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            metadata = {row["image_name"]: row for row in csv.DictReader(handle)}
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            row = metadata.get(image.name, {})
            corrected_name = image.stem + ".txt"
            if corrected_name in corrected:
                label_text = corrected[corrected_name]
            elif row.get("group") == "empty_negative":
                # MakeSense omits empty YOLO files from some exports. These are
                # deliberate background frames and must remain empty labels.
                label_text = ""
            else:
                # A non-negative image missing from the export is only safe to
                # recover from its original auto label when that label exists.
                fallback = batch / "02_auto_labels" / corrected_name
                if not fallback.is_file():
                    raise ValueError(f"{batch.name}: 修订ZIP缺少 {image.name}")
                label_text = fallback.read_text(encoding="utf-8-sig")
            records.append({
                "image": image,
                "label": label_text,
                "session": row.get("session_id", ""),
                "batch": batch.name,
                "source": row.get("source_relative_path", ""),
            })

    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split = split_for_session(record["session"])
        if split not in counts:
            raise ValueError(f"未知session: {record['session']}")
        image_target = output / split / "images" / record["image"].name
        label_target = output / split / "labels" / f"{record['image'].stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], image_target)
        label_target.write_text(normalize_labels(record["label"], len(classes), str(label_target)), encoding="utf-8")
        counts[split] += 1

    if not all(counts.values()):
        raise ValueError(f"划分不完整: {counts}")
    data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
        "session_split": {
            "new_front_full_202608": "train",
            "legacy_utensils_202606": "val",
            "legacy_root_25d00c9abd": "test",
            "legacy_root_a0ae6a2e54": "test",
        },
    }
    import yaml
    (output / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "session", "batch", "image", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow({"split": split_for_session(record["session"]), "session": record["session"], "batch": record["batch"], "image": record["image"].name, "source": record["source"]})
    (output / "release_summary.yaml").write_text(yaml.safe_dump({"images": len(records), "counts": counts}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print({"images": len(records), **counts, "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
