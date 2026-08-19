"""Publish reviewed kettle detection and vessel-pose training releases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import zipfile

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
BOX_BATCHES = ROOT / "annotation_batches" / "kettle_box_warm_clean_injection_v1"
POSE_BATCHES = ROOT / "annotation_batches" / "kettle_pose_warm_clean_injection_v1"
BASE_DETECTION = ROOT / "releases" / "detection" / "front_detect_merged_dedup_v2"
DETECTION_OUTPUT = ROOT / "releases" / "detection" / "front_detect_merged_kettle_v3"
POSE_OUTPUT = ROOT / "releases" / "pose" / "front_vessel_pose_prototype_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_KEY_RE = re.compile(r"__(?P<video_hash>[0-9a-f]{10})__f(?P<frame>\d+)$")
POSE_TO_DETECT = {0: 9, 1: 2, 2: 0, 3: 4}
POSE_NAMES = {0: "kettle", 1: "pitcher", 2: "gaiwan_body", 3: "tea_lotus"}
# One complete source video is a temporary validation holdout. A newly recorded
# independent validation session must replace this split before final acceptance.
POSE_VAL_HASH = "fafbee2735"


def source_key(stem: str) -> str:
    match = SOURCE_KEY_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot parse source frame key: {stem}")
    return f"{match.group('video_hash')}:{int(match.group('frame'))}"


def read_latest_zip(directory: Path) -> zipfile.ZipFile:
    archives = sorted(directory.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not archives:
        raise FileNotFoundError(f"No ZIP export found: {directory}")
    return zipfile.ZipFile(archives[-1])


def normalized_rows(text: str, fields: int, source: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != fields:
            raise ValueError(f"{source}:{line_number}: expected {fields} fields")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if any(value < 0 or value > 2 for value in values):
            raise ValueError(f"{source}:{line_number}: numeric value outside valid range")
        rows.append([str(class_id), *[f"{value:.6f}" for value in values]])
    return rows


def load_box_exports() -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    for batch in sorted(path for path in BOX_BATCHES.glob("batch_*") if path.is_dir()):
        with read_latest_zip(batch / "03_corrected_export") as archive:
            for item in archive.infolist():
                name = Path(item.filename).name
                if item.is_dir() or not name.lower().endswith(".txt"):
                    continue
                if name.lower() in {"labels.txt", "classes.txt", "train.txt"}:
                    continue
                result[Path(name).stem] = normalized_rows(
                    archive.read(item).decode("utf-8-sig"), 5,
                    f"{archive.filename}:{item.filename}",
                )
    return result


def load_pose_exports() -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    expected_names = POSE_NAMES
    for batch in sorted(path for path in POSE_BATCHES.glob("batch_*") if path.is_dir()):
        with read_latest_zip(batch / "02_cvat_pose_export") as archive:
            yaml_items = [item for item in archive.infolist() if Path(item.filename).name == "data.yaml"]
            if len(yaml_items) != 1:
                raise ValueError(f"{archive.filename}: missing or duplicate data.yaml")
            data = yaml.safe_load(archive.read(yaml_items[0]).decode("utf-8-sig"))
            names = {int(key): value for key, value in data.get("names", {}).items()}
            if data.get("kpt_shape") != [3, 3] or names != expected_names:
                raise ValueError(f"{archive.filename}: incompatible pose schema: {data}")
            for item in archive.infolist():
                name = Path(item.filename).name
                if item.is_dir() or not item.filename.startswith("labels/") or not name.endswith(".txt"):
                    continue
                result[Path(name).stem] = normalized_rows(
                    archive.read(item).decode("utf-8-sig"), 14,
                    f"{archive.filename}:{item.filename}",
                )
    return result


def image_index() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for batch in sorted(path for path in POSE_BATCHES.glob("batch_*") if path.is_dir()):
        for image in sorted((batch / "01_images_for_cvat").iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                result[image.stem] = image
    return result


def copy_tree_files(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def publish_detection(
    boxes: dict[str, list[list[str]]], images: dict[str, Path], output: Path
) -> dict:
    copy_tree_files(BASE_DETECTION, output)
    train_images = output / "train" / "images"
    train_labels = output / "train" / "labels"
    kettle_instances = 0
    for stem, image in sorted(images.items()):
        rows = boxes.get(stem)
        if rows is None:
            raise ValueError(f"Missing reviewed detection labels: {stem}")
        kettle_instances += sum(int(row[0]) == 9 for row in rows)
        shutil.copy2(image, train_images / image.name)
        (train_labels / f"{stem}.txt").write_text(
            "\n".join(" ".join(row) for row in rows) + "\n", encoding="utf-8"
        )

    data_path = output / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["path"] = output.as_posix()
    data["kettle_supplement"] = {
        "images": len(images),
        "instances": kettle_instances,
        "split": "train_only",
        "reason": "six source videos are designated training material; independent kettle validation is pending",
    }
    data_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = {
        "base_images": sum(
            1 for split in ("train", "val", "test")
            for path in (BASE_DETECTION / split / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ),
        "added_images": len(images),
        "total_images": sum(
            1 for split in ("train", "val", "test")
            for path in (output / split / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ),
        "kettle_instances_added": kettle_instances,
    }
    (output / "kettle_supplement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def center_distance(pose_row: list[str], detect_row: list[str]) -> float:
    return (float(pose_row[1]) - float(detect_row[1])) ** 2 + (
        float(pose_row[2]) - float(detect_row[2])
    ) ** 2


def merge_pose_boxes(
    pose_rows: list[list[str]], detect_rows: list[list[str]]
) -> tuple[list[list[str]], list[dict]]:
    merged: list[list[str]] = []
    dropped: list[dict] = []
    available = list(enumerate(detect_rows))
    used: set[int] = set()
    for pose_row in pose_rows:
        pose_class = int(pose_row[0])
        detect_class = POSE_TO_DETECT[pose_class]
        candidates = [
            (index, row) for index, row in available
            if index not in used and int(row[0]) == detect_class
        ]
        if not candidates:
            dropped.append({"pose_class": pose_class, "reason": "no_matching_reviewed_box"})
            continue
        index, detect_row = min(candidates, key=lambda item: center_distance(pose_row, item[1]))
        used.add(index)
        merged.append([
            pose_row[0],
            *detect_row[1:5],
            *pose_row[5:14],
        ])
    return merged, dropped


def publish_pose(
    boxes: dict[str, list[list[str]]], poses: dict[str, list[list[str]]],
    images: dict[str, Path], output: Path,
) -> dict:
    counts = {"train": 0, "val": 0}
    class_counts = {"train": {key: 0 for key in POSE_NAMES}, "val": {key: 0 for key in POSE_NAMES}}
    dropped_records: list[dict] = []
    for stem, image in sorted(images.items()):
        if stem not in boxes or stem not in poses:
            raise ValueError(f"Missing detection or pose label: {stem}")
        merged, dropped = merge_pose_boxes(poses[stem], boxes[stem])
        if not any(int(row[0]) == 0 for row in merged):
            raise ValueError(f"Kettle pose could not be matched to reviewed box: {stem}")
        match = SOURCE_KEY_RE.search(stem)
        assert match
        split = "val" if match.group("video_hash") == POSE_VAL_HASH else "train"
        image_target = output / split / "images" / image.name
        label_target = output / split / "labels" / f"{stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_target)
        label_target.write_text(
            "\n".join(" ".join(row) for row in merged) + "\n", encoding="utf-8"
        )
        counts[split] += 1
        for row in merged:
            class_counts[split][int(row[0])] += 1
        for item in dropped:
            dropped_records.append({"image": image.name, **item})

    data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "nc": 4,
        "names": POSE_NAMES,
        "kpt_shape": [3, 3],
        "flip_idx": [0, 1, 2],
        "prototype_same_session_holdout": True,
        "split_policy": {
            "unit": "complete_source_video_hash",
            "validation_hash": POSE_VAL_HASH,
            "warning": "temporary internal validation; record an independent session before acceptance testing",
        },
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "images": counts,
        "instances": class_counts,
        "dropped_unmatched_non_kettle_skeletons": len(dropped_records),
        "all_kettle_skeletons_retained": True,
    }
    (output / "release_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "dropped_pose_objects.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "pose_class", "reason"])
        writer.writeheader()
        writer.writerows(dropped_records)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish kettle detection and pose releases")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for output in (DETECTION_OUTPUT, POSE_OUTPUT):
        if output.exists() and any(output.iterdir()):
            if not args.force:
                raise FileExistsError(f"Output is not empty; use --force: {output}")
            shutil.rmtree(output)

    boxes = load_box_exports()
    poses = load_pose_exports()
    images = image_index()
    if len(boxes) != 126 or len(poses) != 126 or len(images) != 126:
        raise ValueError(
            f"Expected 126 samples, got boxes={len(boxes)}, poses={len(poses)}, images={len(images)}"
        )
    detection = publish_detection(boxes, images, DETECTION_OUTPUT)
    pose = publish_pose(boxes, poses, images, POSE_OUTPUT)
    print(json.dumps({
        "detection": detection,
        "pose": pose,
        "detection_yaml": str(DETECTION_OUTPUT / "data.yaml"),
        "pose_yaml": str(POSE_OUTPUT / "data.yaml"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
