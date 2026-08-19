"""Create MakeSense annotation batches from reviewed frame candidates.

The script deduplicates regular/action candidates by source video and frame
index, ignores images deleted during manual review, and hard-links images when
possible so a 1,000+ image review does not duplicate disk usage.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
DEFAULT_OUTPUT = DEFAULT_ROOT / "annotation_batches" / "detection_v1"
CLASSES = PROJECT / "dataset" / "tea_dataset_v1_reviewed" / "classes.txt"


def read_candidates(root: Path, filename: str, kind: str) -> list[dict[str, str]]:
    path = root / "manifests" / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            candidate = root / row["candidate_path"]
            if not candidate.is_file():
                continue
            row["candidate_kind"] = kind
            row["group"] = row.get("module") or row.get("action_id") or kind
            rows.append(row)
        return rows


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="将审核后的候选帧整理为MakeSense标注批次")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch-size必须大于0")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {output}")

    regular = read_candidates(root, "frame_candidates.csv", "detection")
    action = read_candidates(root, "action_frame_candidates.csv", "action")
    # Prefer regular detection paths for frames that also occur in an action
    # pool; the source/frame key keeps only one copy for annotation.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in regular + action:
        key = (row["source_relative_path"], row["frame_index"])
        unique.setdefault(key, row)
    rows = sorted(unique.values(), key=lambda row: (
        row["session_id"], row["source_relative_path"], int(row["frame_index"]), row["candidate_path"],
    ))
    if not rows:
        raise ValueError("没有可用候选图片")

    class_text = CLASSES.read_text(encoding="utf-8-sig")
    summary: list[dict[str, object]] = []
    for offset in range(0, len(rows), args.batch_size):
        selected = rows[offset:offset + args.batch_size]
        batch_number = offset // args.batch_size + 1
        batch = output / f"batch_{batch_number:03d}"
        images = batch / "01_images"
        exports = batch / "02_makesense_export"
        images.mkdir(parents=True, exist_ok=True)
        exports.mkdir(parents=True, exist_ok=True)
        (batch / "labels.txt").write_text(class_text, encoding="utf-8")
        manifest_rows: list[dict[str, str]] = []
        for row in selected:
            source = root / row["candidate_path"]
            target = images / source.name
            if target.exists():
                raise FileExistsError(f"批次内文件名冲突: {target.name}")
            link_or_copy(source, target)
            manifest_rows.append({
                "image_name": target.name,
                "candidate_path": row["candidate_path"],
                "source_relative_path": row["source_relative_path"],
                "session_id": row["session_id"],
                "frame_index": row["frame_index"],
                "candidate_kind": row["candidate_kind"],
                "group": row["group"],
                "review_status": "pending_annotation",
            })
        fields = ["image_name", "candidate_path", "source_relative_path", "session_id", "frame_index", "candidate_kind", "group", "review_status"]
        with (batch / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest_rows)
        summary.append({"batch": batch.name, "images": len(selected), "status": "pending_annotation"})

    with (output / "batches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "images", "status"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"标注图片: {len(rows)}，批次: {len(summary)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
