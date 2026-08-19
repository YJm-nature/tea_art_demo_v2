"""Create the isolated front-camera SOP dataset workspace."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
SESSIONS = [f"front_s{index:02d}" for index in range(1, 9)]
SPLITS = {**{name: "train" for name in SESSIONS[:6]}, "front_s07": "val", "front_s08": "test"}
ACTIONS = (
    "warm_clean", "hold_lotus", "open_lid_smell", "tea_lotus_to_gaiwan",
    "water_injection", "brew_timing", "gaiwan_to_pitcher",
    "tea_distribution", "cup_layout", "two_hand_serve_tray",
)


def initialize(root: Path) -> None:
    directories = [
        root / "derived" / "detection" / "images",
        root / "derived" / "pose" / "images",
        root / "derived" / "ocr" / "temperature" / "images",
        root / "derived" / "ocr" / "weight" / "images",
        root / "annotations" / "detection_yolo18" / "labels",
        root / "annotations" / "pose_yolo" / "labels",
        root / "annotations" / "actions",
        root / "annotations" / "ocr",
        root / "manifests",
        root / "splits",
        root / "releases" / "detection",
        root / "releases" / "pose",
        root / "releases" / "ocr",
        root / "reports",
    ]
    for session in SESSIONS:
        base = root / "raw_videos" / session
        directories.extend([
            base / "01_utensils" / "single",
            base / "01_utensils" / "grouped",
            base / "01_utensils" / "occlusion_handheld",
            base / "02_actions",
            base / "03_ocr" / "temperature",
            base / "03_ocr" / "weight",
            base / "04_pose" / "kettle",
            base / "04_pose" / "pitcher",
            base / "04_pose" / "gaiwan",
            base / "04_pose" / "lotus",
            base / "05_negatives",
        ])
        for action in ACTIONS:
            for variant in ("positive", "error", "hard_negative"):
                directories.append(base / "02_actions" / action / variant)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    session_manifest = root / "manifests" / "front_sessions.csv"
    if not session_manifest.exists():
        with session_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "session_id", "split", "operator_id", "capture_date", "lighting",
                "camera_position_locked", "review_status", "notes",
            ])
            for session in SESSIONS:
                writer.writerow([session, SPLITS[session], "", "", "", "yes", "pending", ""])

    inventory = root / "manifests" / "video_inventory.csv"
    if not inventory.exists():
        with inventory.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "relative_path", "session_id", "module", "variant", "repetitions",
                "front_view_ok", "display_readable", "review_status", "notes",
            ])

    action_annotations = root / "annotations" / "actions" / "events.jsonl"
    action_annotations.touch(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="建立正面机位SOP数据集目录")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    initialize(root)
    print(f"正面数据集目录已就绪: {root}")
    print("train: front_s01-front_s06 | val: front_s07 | test: front_s08")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

