"""Learn a robust box correction from a reviewed batch and apply it to pending labels."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import median
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_annotation import box_iou
from src.dataset_rebuild import parse_yolo_labels, read_manifest, write_manifest, write_yolo_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a reviewed scene-specific box correction")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("reference_before_labels", type=Path)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--status", default="needs_fix")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-iou", type=float, default=0.15)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    current_labels = workspace / "pool" / "labels" / "detect"
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    class_count = len((workspace / "classes.txt").read_text(encoding="utf-8-sig").splitlines())
    if not 0 <= args.class_id < class_count:
        raise ValueError(f"class ID越界: {args.class_id}")
    correction = learn_correction(
        args.reference_before_labels.resolve(),
        current_labels,
        reference["sample_ids"],
        class_count,
        args.class_id,
        args.min_iou,
    )
    if correction["matched_pairs"] < args.min_pairs:
        raise ValueError(
            f"有效人工修正样本不足: {correction['matched_pairs']} < {args.min_pairs}"
        )

    records = read_manifest(workspace)
    targets = [
        record for record in records
        if record["source"] == args.source and record["review_status"] == args.status
    ]
    if not targets:
        raise ValueError("没有符合条件的待校正样本")
    backup_dir = workspace / "pool" / "labels" / f"detect_before_{args.run_name}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)

    changed_images = 0
    changed_boxes = 0
    with tempfile.TemporaryDirectory(prefix="box_correction_", dir=workspace) as temp_dir:
        staged = Path(temp_dir)
        for record in targets:
            source_label = workspace / record["detect_label"]
            rows = parse_yolo_labels(source_label, class_count)
            corrected = []
            image_changed = False
            for row in rows:
                if row[0] == args.class_id:
                    row = apply_correction(row, correction)
                    changed_boxes += 1
                    image_changed = True
                corrected.append(row)
            write_yolo_labels(staged / f"{record['sample_id']}.txt", corrected)
            changed_images += image_changed

        backup_dir.mkdir(parents=True)
        for record in targets:
            source_label = workspace / record["detect_label"]
            shutil.copy2(source_label, backup_dir / source_label.name)
            shutil.copy2(staged / source_label.name, source_label)

    corrected_at = datetime.now().astimezone().isoformat()
    for record in targets:
        record["last_box_correction"] = args.run_name
        record["box_corrected_at"] = corrected_at
        record["review_note"] = f"{args.run_name} 自动校正class {args.class_id}，仍需人工复核"
    write_manifest(workspace, records)
    report = {
        "run_name": args.run_name,
        "class_id": args.class_id,
        "source": args.source,
        "target_status": args.status,
        "target_images": len(targets),
        "changed_images": changed_images,
        "changed_boxes": changed_boxes,
        "backup": str(backup_dir),
        "correction": correction,
        "review_status_unchanged": True,
    }
    report_dir = workspace / "output" / "box_corrections"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{args.run_name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def learn_correction(
    before_dir: Path,
    after_dir: Path,
    sample_ids: list[str],
    class_count: int,
    class_id: int,
    min_iou: float,
) -> dict:
    values = []
    for stem in sample_ids:
        before = [
            row for row in parse_yolo_labels(before_dir / f"{stem}.txt", class_count)
            if row[0] == class_id
        ]
        after = [
            row for row in parse_yolo_labels(after_dir / f"{stem}.txt", class_count)
            if row[0] == class_id
        ]
        if len(before) != 1 or len(after) != 1 or box_iou(before[0], after[0]) < min_iou:
            continue
        old, new = before[0], after[0]
        values.append((
            new[1] - old[1],
            new[2] - old[2],
            new[3] / old[3],
            new[4] / old[4],
            box_iou(old, new),
        ))
    if not values:
        raise ValueError("没有可匹配的人工修正框")
    return {
        "matched_pairs": len(values),
        "dx": round(median(value[0] for value in values), 6),
        "dy": round(median(value[1] for value in values), 6),
        "width_ratio": round(median(value[2] for value in values), 6),
        "height_ratio": round(median(value[3] for value in values), 6),
        "median_reference_iou": round(median(value[4] for value in values), 6),
    }


def apply_correction(row: tuple, correction: dict) -> tuple:
    class_id, x, y, width, height = row
    width *= correction["width_ratio"]
    height *= correction["height_ratio"]
    x += correction["dx"]
    y += correction["dy"]
    x1, y1 = max(0.0, x - width / 2), max(0.0, y - height / 2)
    x2, y2 = min(1.0, x + width / 2), min(1.0, y + height / 2)
    return (class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


if __name__ == "__main__":
    raise SystemExit(main())
