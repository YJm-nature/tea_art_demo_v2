"""Prepare reviewed seeds and conservatively merge pseudo detections."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from statistics import median
from typing import Iterable, Sequence
from zipfile import ZipFile

import yaml

from .dataset_rebuild import parse_yolo_labels, read_manifest, write_manifest, write_yolo_labels


YoloRow = tuple[int, float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def row(self) -> YoloRow:
        return (self.class_id, self.x, self.y, self.width, self.height)


def read_seed_archive(archive: Path, workspace: Path, class_count: int) -> dict[str, list[YoloRow]]:
    image_stems = {path.stem for path in (workspace / "pool" / "images").iterdir() if path.is_file()}
    seeds: dict[str, list[YoloRow]] = {}
    with ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            entry_path = Path(entry.filename)
            if entry_path.name != entry.filename or entry_path.suffix.lower() != ".txt":
                raise ValueError(f"种子压缩包含非法路径或非TXT文件: {entry.filename}")
            stem = entry_path.stem
            if stem in seeds:
                raise ValueError(f"种子压缩包重复文件: {entry.filename}")
            if stem not in image_stems:
                raise ValueError(f"种子标签没有同名图片: {entry.filename}")
            text = bundle.read(entry).decode("utf-8-sig")
            seeds[stem] = parse_yolo_text(text, entry.filename, class_count)
    if not seeds:
        raise ValueError("种子压缩包为空")
    return seeds


def parse_yolo_text(text: str, source: str, class_count: int) -> list[YoloRow]:
    rows: list[YoloRow] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number} 应为5列")
        class_id = int(fields[0])
        coords = tuple(float(value) for value in fields[1:])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{source}:{line_number} 类别ID越界: {class_id}")
        x, y, width, height = coords
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{source}:{line_number} 坐标越界")
        rows.append((class_id, x, y, width, height))
    return rows


def prepare_seed_dataset(
    workspace: Path,
    archive: Path,
    output: Path,
    validation_stride: int = 5,
) -> dict:
    workspace = workspace.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")
    class_names = workspace.joinpath("classes.txt").read_text(encoding="utf-8-sig").splitlines()
    seeds = read_seed_archive(archive.resolve(), workspace, len(class_names))
    records = {record["sample_id"]: record for record in read_manifest(workspace)}
    class_counts = Counter(row[0] for rows in seeds.values() for row in rows)
    seed_manifest = []

    for index, stem in enumerate(sorted(seeds)):
        split = "val" if (index + 1) % validation_stride == 0 else "train"
        record = records[stem]
        source_image = workspace / record["image"]
        target_image = output / split / "images" / source_image.name
        target_label = output / split / "labels" / f"{stem}.txt"
        _link_or_copy(source_image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_labels(target_label, seeds[stem])
        seed_manifest.append({
            "sample_id": stem,
            "split": split,
            "session_id": record["session_id"],
            "image": target_image.relative_to(output).as_posix(),
            "label": target_label.relative_to(output).as_posix(),
            "boxes": len(seeds[stem]),
        })

    data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "seed_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seed_manifest),
        encoding="utf-8",
    )
    summary = {
        "purpose": "pseudo_annotation_only",
        "formal_validation": False,
        "seed_images": len(seeds),
        "train_images": sum(row["split"] == "train" for row in seed_manifest),
        "val_images": sum(row["split"] == "val" for row in seed_manifest),
        "session_ids": sorted({row["session_id"] for row in seed_manifest}),
        "class_instances": {
            str(class_id): class_counts[class_id] for class_id in range(len(class_names))
        },
        "active_class_ids": sorted(class_counts),
        "inactive_class_ids": [class_id for class_id in range(len(class_names)) if not class_counts[class_id]],
    }
    (output / "seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def merge_baseline_with_predictions(
    baseline: Sequence[YoloRow],
    predictions: Sequence[Prediction],
    active_classes: set[int],
    split_classes: set[int] = frozenset({0, 1}),
    match_iou: float = 0.35,
    refine_confidence: float = 0.35,
    add_confidence: float = 0.55,
    split_add_confidence: float = 0.25,
) -> tuple[list[YoloRow], dict]:
    """Keep existing boxes, refine matched boxes, and add only high-confidence new boxes."""
    output = [row for row in baseline if row[0] not in active_classes]
    baseline_by_class: dict[int, list[YoloRow]] = defaultdict(list)
    predictions_by_class: dict[int, list[Prediction]] = defaultdict(list)
    for row in baseline:
        if row[0] in active_classes:
            baseline_by_class[row[0]].append(row)
    for prediction in predictions:
        if prediction.class_id in active_classes:
            predictions_by_class[prediction.class_id].append(prediction)

    added = 0
    refined = 0
    retained = 0
    accepted_confidences = []
    for class_id in sorted(active_classes):
        available = sorted(
            predictions_by_class[class_id], key=lambda prediction: prediction.confidence, reverse=True
        )
        used: set[int] = set()
        for baseline_row in baseline_by_class[class_id]:
            candidates = [
                (index, box_iou(baseline_row, prediction.row), prediction)
                for index, prediction in enumerate(available)
                if index not in used
            ]
            best = max(candidates, key=lambda item: item[1], default=None)
            if best and best[1] >= match_iou and best[2].confidence >= refine_confidence:
                used.add(best[0])
                output.append(best[2].row)
                refined += 1
                accepted_confidences.append(best[2].confidence)
            else:
                output.append(baseline_row)
                retained += 1

        threshold = split_add_confidence if class_id in split_classes else add_confidence
        for index, prediction in enumerate(available):
            if index in used or prediction.confidence < threshold:
                continue
            if any(
                row[0] == class_id and box_iou(row, prediction.row) >= match_iou
                for row in output
            ):
                continue
            output.append(prediction.row)
            added += 1
            accepted_confidences.append(prediction.confidence)

    output.sort(key=lambda row: (row[0], row[1], row[2]))
    report = {
        "baseline_boxes": len(baseline),
        "prediction_boxes": len(predictions),
        "output_boxes": len(output),
        "added": added,
        "refined": refined,
        "retained": retained,
        "mean_accepted_confidence": (
            round(sum(accepted_confidences) / len(accepted_confidences), 4)
            if accepted_confidences else None
        ),
    }
    return output, report


def box_iou(first: YoloRow, second: YoloRow) -> float:
    first_box = _corners(first)
    second_box = _corners(second)
    intersection_width = max(0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]))
    intersection_height = max(0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_box[2] - first_box[0]) * max(0.0, first_box[3] - first_box[1])
    second_area = max(0.0, second_box[2] - second_box[0]) * max(0.0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def learn_legacy_split_transforms(
    workspace: Path,
    seeds: dict[str, list[YoloRow]],
    split_classes: set[int] = frozenset({0, 1}),
) -> dict[int, tuple[float, float, float, float]]:
    """Learn robust relative boxes from a legacy whole-gaiwan box and reviewed split boxes."""
    records = {record["sample_id"]: record for record in read_manifest(workspace)}
    values: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for stem, reviewed_rows in seeds.items():
        record = records[stem]
        legacy_rows = parse_yolo_labels(workspace / record["legacy_label"], 9)
        legacy_gaiwans = [row for row in legacy_rows if row[0] == 0]
        if len(legacy_gaiwans) != 1:
            continue
        _, old_x, old_y, old_width, old_height = legacy_gaiwans[0]
        for class_id, x, y, width, height in reviewed_rows:
            if class_id not in split_classes:
                continue
            relative = (
                (x - old_x) / old_width,
                (y - old_y) / old_height,
                width / old_width,
                height / old_height,
            )
            dx, dy, width_ratio, height_ratio = relative
            if (
                abs(dx) <= 0.5
                and abs(dy) <= 0.75
                and 0.2 <= width_ratio <= 1.5
                and 0.2 <= height_ratio <= 1.5
            ):
                values[class_id].append(relative)

    missing = split_classes - set(values)
    if missing:
        raise ValueError(f"No reviewed legacy transform examples for classes: {sorted(missing)}")
    return {
        class_id: tuple(median(row[index] for row in rows) for index in range(4))
        for class_id, rows in values.items()
    }


def add_missing_legacy_split_boxes(
    rows: Sequence[YoloRow],
    legacy_rows: Sequence[YoloRow],
    transforms: dict[int, tuple[float, float, float, float]],
) -> tuple[list[YoloRow], int]:
    """Add review candidates only for split classes absent from model/baseline output."""
    output = list(rows)
    present = {row[0] for row in output}
    added = 0
    for _, old_x, old_y, old_width, old_height in (row for row in legacy_rows if row[0] == 0):
        for class_id, (dx, dy, width_ratio, height_ratio) in sorted(transforms.items()):
            if class_id in present:
                continue
            candidate = _clip_box((
                class_id,
                old_x + dx * old_width,
                old_y + dy * old_height,
                width_ratio * old_width,
                height_ratio * old_height,
            ))
            if candidate is not None:
                output.append(candidate)
                added += 1
                present.add(class_id)
    output.sort(key=lambda row: (row[0], row[1], row[2]))
    return output, added


def apply_candidate_labels(workspace: Path, candidates: Path, run_name: str) -> dict:
    workspace = workspace.resolve()
    candidates = candidates.resolve()
    records = read_manifest(workspace)
    expected = {record["sample_id"] for record in records}
    provided = {path.stem for path in candidates.glob("*.txt")}
    if provided != expected:
        raise ValueError(
            f"候选标签不完整: missing={len(expected - provided)} extra={len(provided - expected)}"
        )
    class_count = len(workspace.joinpath("classes.txt").read_text(encoding="utf-8-sig").splitlines())
    for stem in sorted(provided):
        parse_yolo_labels(candidates / f"{stem}.txt", class_count)

    detect_dir = workspace / "pool" / "labels" / "detect"
    backup_dir = workspace / "pool" / "labels" / f"detect_before_{run_name}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)
    shutil.copytree(detect_dir, backup_dir)
    for stem in sorted(provided):
        shutil.copy2(candidates / f"{stem}.txt", detect_dir / f"{stem}.txt")

    preserved_rejected = 0
    for record in records:
        if record["review_status"] == "rejected":
            preserved_rejected += 1
            continue
        record["review_status"] = "needs_fix"
        record["reviewer"] = None
        record["reviewed_at"] = None
        record["review_note"] = f"{run_name}伪标注，必须人工复核"
        if record.get("second_review_required"):
            record["second_review_status"] = "pending"
            record["second_reviewer"] = None
            record["second_reviewed_at"] = None
            record["second_review_note"] = ""
    write_manifest(workspace, records)
    return {
        "applied_labels": len(provided),
        "backup": str(backup_dir),
        "review_status": "needs_fix",
        "preserved_rejected": preserved_rejected,
    }


def _corners(row: YoloRow) -> tuple[float, float, float, float]:
    _, x, y, width, height = row
    return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def _clip_box(row: YoloRow) -> YoloRow | None:
    class_id, _, _, _, _ = row
    x1, y1, x2, y2 = _corners(row)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
