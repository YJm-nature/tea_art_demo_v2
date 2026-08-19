"""Build a traceable multi-task catalogue without copying source media.

The tea project deliberately keeps one physical image in its original dataset.
This script creates CSV manifests that let the same source frame participate in
separate tasks: utensil detection, action-segment validation, and vessel pose.
It never creates a train/val/test split and never modifies annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset"
DEFAULT_OUTPUT = DATASET / "tea_multitask_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def add_asset(
    assets: dict[str, dict[str, Any]],
    path: Path,
    *,
    source_dataset: str,
    session_id: str,
    camera_role: str,
    label_path: Path | None,
    tasks: set[str],
    detector_use: str,
    split_note: str,
) -> str:
    digest = sha1(path)
    asset = assets.setdefault(
        digest,
        {
            "asset_id": f"sha1:{digest}",
            "image_paths": set(),
            "label_paths": set(),
            "source_datasets": set(),
            "session_ids": set(),
            "camera_roles": set(),
            "tasks": set(),
            "detector_uses": set(),
            "split_notes": set(),
        },
    )
    asset["image_paths"].add(relative(path))
    if label_path and label_path.is_file():
        asset["label_paths"].add(relative(label_path))
    asset["source_datasets"].add(source_dataset)
    asset["session_ids"].add(session_id)
    asset["camera_roles"].add(camera_role)
    asset["tasks"].update(tasks)
    asset["detector_uses"].add(detector_use)
    asset["split_notes"].add(split_note)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create multi-task tea-data manifests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace only the generated workspace")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"workspace exists: {output}; use --force to regenerate")
        shutil.rmtree(output)
    manifests = output / "manifests"
    sources = output / "source_pointers"
    manifests.mkdir(parents=True)
    sources.mkdir()

    assets: dict[str, dict[str, Any]] = {}
    action_refs: list[dict[str, str]] = []

    # 1. The reviewed YOLOv8n front release is the current, usable detector base.
    front_release = DATASET / "tea_sop_front_v1" / "releases" / "detection" / "front_detect_reviewed_v1"
    with (front_release / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image = front_release / row["split"] / "images" / row["image"]
            label = front_release / row["split"] / "labels" / f"{Path(row['image']).stem}.txt"
            add_asset(
                assets, image,
                source_dataset="front_detect_reviewed_v1",
                session_id=row["session"], camera_role="front", label_path=label,
                tasks={"utensil_detection"}, detector_use="front_detector_current_release",
                split_note=f"existing_{row['split']}_session_split",
            )

    # 2. The fully reviewed legacy 18-class labels are valuable detector candidates,
    # but their sessions must be assigned before they are merged into a final release.
    reviewed = DATASET / "tea_dataset_v1_reviewed"
    for row in read_jsonl(reviewed / "manifest.jsonl"):
        if row.get("review_status") != "accepted":
            continue
        image = reviewed / row["image"]
        label = reviewed / row["detect_label"]
        add_asset(
            assets, image,
            source_dataset="reviewed_legacy_18class",
            session_id=str(row.get("session_id", "unknown")), camera_role="front_or_unknown",
            label_path=label, tasks={"utensil_detection"},
            detector_use="front_detector_merge_candidate",
            split_note="assign_whole_source_session_before_final_release",
        )

    # 3. Side material is retained for action rules and later pose data. It must not
    # influence front-camera validation or test results.
    side = DATASET / "tea_sop_side_transition_v1"
    side_metadata = {
        Path(row["image"]).name: row
        for row in read_jsonl(side / "manifests" / "frames.jsonl")
    }
    for image in image_files(side / "pool" / "images"):
        row = side_metadata.get(image.name, {})
        label = side / "pool" / "labels" / "detect" / f"{image.stem}.txt"
        add_asset(
            assets, image,
            source_dataset="side_transition_18class",
            session_id=str(row.get("session_id", "office_side_unknown")), camera_role="side",
            label_path=label, tasks={"utensil_detection", "action_rule_validation", "pose_candidate"},
            detector_use="side_only_optional_training_not_front_eval",
            split_note="side_sessions_are_never_front_val_or_front_test",
        )

    # 4. Action frames reference the same physical content as many detection frames.
    # They carry action labels through the segment manifest, never through YOLO class IDs.
    candidates_path = DATASET / "tea_sop_front_v1" / "manifests" / "action_frame_candidates.csv"
    with candidates_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image = DATASET / "tea_sop_front_v1" / row["candidate_path"]
            if not image.is_file():
                continue
            digest = add_asset(
                assets, image,
                source_dataset="front_action_pool",
                session_id=row["session_id"], camera_role="front", label_path=None,
                tasks={"action_segment", "pose_candidate"},
                detector_use="reuse_detector_box_only_if_matching_reviewed_label_exists",
                split_note="all_new_front_full_202608_samples_are_train_until_new_sessions_exist",
            )
            action_refs.append({
                "asset_id": f"sha1:{digest}", "image_path": relative(image),
                "source_video": row["source_relative_path"], "session_id": row["session_id"],
                "action_id": row["action_id"], "variant": row["variant"],
                "frame_index": row["frame_index"], "time_s": row["time_s"],
                "review_status": row["review_status"], "split": row["split"],
            })

    asset_rows = []
    for digest, item in sorted(assets.items()):
        asset_rows.append({
            "asset_id": item["asset_id"],
            "image_paths": " | ".join(sorted(item["image_paths"])),
            "label_paths": " | ".join(sorted(item["label_paths"])),
            "source_datasets": " | ".join(sorted(item["source_datasets"])),
            "session_ids": " | ".join(sorted(item["session_ids"])),
            "camera_roles": " | ".join(sorted(item["camera_roles"])),
            "tasks": " | ".join(sorted(item["tasks"])),
            "detector_uses": " | ".join(sorted(item["detector_uses"])),
            "split_notes": " | ".join(sorted(item["split_notes"])),
        })
    write_csv(manifests / "image_assets.csv", asset_rows)
    write_csv(manifests / "action_frame_references.csv", action_refs)

    # Copy only metadata for direct review. Raw videos remain where they were captured.
    shutil.copy2(DATASET / "tea_sop_front_v1" / "manifests" / "action_segments_review.csv", manifests / "action_segments_review.csv")
    summary = {
        "schema_version": "1.0",
        "physical_images": len(asset_rows),
        "action_frame_references": len(action_refs),
        "assets_by_source": dict(Counter(
            source for item in assets.values() for source in item["source_datasets"]
        )),
        "assets_by_task": dict(Counter(
            task for item in assets.values() for task in item["tasks"]
        )),
        "duplicate_action_frame_references": len(action_refs) - len({row["asset_id"] for row in action_refs}),
        "rules": [
            "One physical image may have multiple task annotations; do not copy it into multiple datasets.",
            "YOLO detection labels contain only the fixed 18 utensil classes, never action names.",
            "Action labels live in action segments and frame-reference manifests.",
            "Train, validation, and test are assigned by complete recording session, never by random frames.",
            "Side-camera material must not be used to claim front-camera validation or test metrics.",
        ],
    }
    (manifests / "catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_pointers(sources)
    (output / "README.md").write_text(README, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"workspace: {output}")
    return 0


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pointers(directory: Path) -> None:
    pointers = {
        "front_detector_current_release.txt": "dataset/tea_sop_front_v1/releases/detection/front_detect_reviewed_v1\n",
        "legacy_reviewed_detector_candidates.txt": "dataset/tea_dataset_v1_reviewed/pool/images\n",
        "side_action_and_pose_candidates.txt": "dataset/tea_sop_side_transition_v1\n",
        "front_action_frames.txt": "dataset/tea_sop_front_v1/derived/action_pool\n",
        "modular_raw_videos.txt": "dataset/tea_sop_modular_v1/raw_videos\n",
    }
    for name, target in pointers.items():
        (directory / name).write_text(target, encoding="utf-8")


README = """# Tea Multitask Workspace v1

This is a manifest workspace. It does not duplicate images or labels; the source
paths remain authoritative and are listed in `source_pointers/`.

## Tasks

- `utensil_detection`: YOLO 18-class boxes. Use front-camera data for the formal
  front model. Action frames may also be used if their utensil boxes are reviewed.
- `action_segment`: source video, action, positive/error/hard-negative variant,
  start/end time, and selected evidence frames. Action names never enter YOLO box
  labels.
- `pose_candidate`: selected frames that need 3 vessel keypoints for YOLOv8-pose.
- `action_rule_validation`: positive and negative video segments used to tune and
  evaluate the temporal rules plus the SOP state machine.

## Generated files

- `manifests/image_assets.csv`: one row per physical image, including all task uses.
- `manifests/action_frame_references.csv`: one row per action-frame use. A frame may
  occur in more than one action at a boundary; this is expected.
- `manifests/action_segments_review.csv`: action time ranges to complete and review.

## Safety rules

1. Assign one complete recording session to exactly one of train, val, or test.
2. Do not combine side-camera samples with front-camera val/test results.
3. Do not copy the same picture under both `warm_clean` and `tea_lotus_to_gaiwan`.
   Keep one image asset, with two task references only when it is genuinely a
   boundary frame for both segments.
4. Do not train an action class named `warm_clean` or `water_injection` in the
   utensil YOLO model. Those are temporal observations assembled from detection,
   hand/pose, keypoint, tracking, and timing evidence.
"""


if __name__ == "__main__":
    raise SystemExit(main())
