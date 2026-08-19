"""Remove a wrong-class box when it strongly duplicates a trusted-class box."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_annotation import box_iou
from src.dataset_rebuild import parse_yolo_labels, read_manifest, write_manifest, write_yolo_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove strongly overlapping wrong-class boxes")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--wrong-class", type=int, required=True)
    parser.add_argument("--reference-class", type=int, required=True)
    parser.add_argument("--min-iou", type=float, default=0.75)
    parser.add_argument("--source", required=True)
    parser.add_argument("--status", default="needs_fix")
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    class_count = len((workspace / "classes.txt").read_text(encoding="utf-8-sig").splitlines())
    for class_id in (args.wrong_class, args.reference_class):
        if not 0 <= class_id < class_count:
            raise ValueError(f"class ID越界: {class_id}")
    if not 0 < args.min_iou <= 1:
        raise ValueError("min-iou必须在(0, 1]范围内")

    excluded = set()
    if args.exclude_manifest:
        excluded = set(json.loads(
            args.exclude_manifest.read_text(encoding="utf-8")
        )["sample_ids"])
    records = read_manifest(workspace)
    targets = [
        record for record in records
        if record["source"] == args.source
        and record["review_status"] == args.status
        and record["sample_id"] not in excluded
    ]

    changes = {}
    removed_scores = []
    for record in targets:
        label_path = workspace / record["detect_label"]
        rows = parse_yolo_labels(label_path, class_count)
        cleaned, removed = remove_overlaps(
            rows, args.wrong_class, args.reference_class, args.min_iou
        )
        if removed:
            changes[record["sample_id"]] = (label_path, cleaned)
            removed_scores.extend(removed)
    if not changes:
        raise ValueError("没有命中重叠错类规则的标签")

    backup_dir = workspace / "pool" / "labels" / f"detect_before_{args.run_name}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)
    backup_dir.mkdir(parents=True)
    for stem, (label_path, cleaned) in sorted(changes.items()):
        shutil.copy2(label_path, backup_dir / label_path.name)
        write_yolo_labels(label_path, cleaned)

    corrected_at = datetime.now().astimezone().isoformat()
    for record in records:
        if record["sample_id"] in changes:
            record["last_overlap_cleanup"] = args.run_name
            record["overlap_cleaned_at"] = corrected_at
            record["review_note"] = (
                f"{args.run_name} 删除与class {args.reference_class}高度重叠的"
                f"错误class {args.wrong_class}框，仍需人工复核"
            )
    write_manifest(workspace, records)
    report = {
        "run_name": args.run_name,
        "source": args.source,
        "target_status": args.status,
        "excluded_images": len(excluded),
        "scanned_images": len(targets),
        "affected_images": len(changes),
        "removed_boxes": len(removed_scores),
        "wrong_class": args.wrong_class,
        "reference_class": args.reference_class,
        "min_iou": args.min_iou,
        "removed_iou_range": [round(min(removed_scores), 4), round(max(removed_scores), 4)],
        "backup": str(backup_dir),
        "review_status_unchanged": True,
    }
    report_dir = workspace / "output" / "overlap_cleanups"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{args.run_name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def remove_overlaps(
    rows: list[tuple], wrong_class: int, reference_class: int, min_iou: float
) -> tuple[list[tuple], list[float]]:
    references = [row for row in rows if row[0] == reference_class]
    cleaned = []
    removed_scores = []
    for row in rows:
        score = max((box_iou(row, reference) for reference in references), default=0.0)
        if row[0] == wrong_class and score >= min_iou:
            removed_scores.append(score)
        else:
            cleaned.append(row)
    return cleaned, removed_scores


if __name__ == "__main__":
    raise SystemExit(main())
