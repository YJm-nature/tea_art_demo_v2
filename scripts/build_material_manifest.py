"""Inventory all source material without moving or modifying original files.

The manifest is deliberately conservative: source groups are treated as sessions,
and all material from a group must stay in one split.  This is the first step
before frame extraction, annotation review, and release publishing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
SOURCE_ROOT = PROJECT / "dataset"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ACTIONS = (
    "warm_clean", "hold_lotus", "open_lid_smell", "tea_lotus_to_gaiwan",
    "water_injection", "brew_timing", "gaiwan_to_pitcher", "tea_distribution",
    "cup_layout", "two_hand_serve_tray",
)


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def sha1_file(path: Path, chunk_size: int = 256 * 1024) -> str:
    """Fast content fingerprint using file size plus head/tail blocks.

    A full hash is unnecessarily expensive for large source videos.  The
    fingerprint is used for duplicate detection, while the original path and
    session remain the authoritative provenance fields.
    """
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(0, size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def classify(relative: str) -> tuple[str, str, str]:
    """Return module, variant, and camera hint from a dataset-relative path."""
    text = relative.replace("\\", "/")
    name = Path(text).stem.lower()
    if "正面完整流程视频" in text:
        return "full_sop", "positive", "front"
    if "01_single_object" in text:
        return "utensil_static", "positive", "front"
    if "02_grouped_objects" in text:
        return "utensil_grouped", "positive", "front"
    if "03_occlusion_handheld" in text:
        return ("smell" if "闻香" in text else "occlusion_handheld"), (
            "hard_negative" if "负例" in text else "positive"), "front"
    if "04_empty_negatives" in text:
        return "empty_negative", "negative", "front"
    if "step03_tea_preparation" in text:
        return "hold_lotus", "error" if "error" in text else "positive", "front"
    if "step06_serve" in text or "杯位布局" in text:
        return "cup_layout", "positive", "front"
    if "00_inbox" in text:
        return "legacy_full_sop", "unknown", "front_or_side"
    if "茶拨茶夹专项" in text or "补充茶拨茶夹公道杯" in text:
        return "utensil_occlusion", "positive", "front_or_side"
    if "演示视频" in text or "WIN_20260722" in text:
        return "legacy_full_sop", "unknown", "front_or_side"
    return "unknown_review", "unknown", "unknown"


def session_for(relative: str, is_image: bool = False) -> str:
    text = relative.replace("\\", "/")
    if "tea_sop_side_transition_v1/正面完整流程视频" in text:
        return "new_front_full_202608"
    if "tea_sop_modular_v1/raw_videos/00_inbox" in text:
        return "legacy_inbox_202606"
    if "tea_sop_modular_v1/raw_videos/01_utensils" in text:
        return "legacy_utensils_202606"
    if "tea_sop_modular_v1/raw_videos/02_sop_steps" in text:
        return "legacy_actions_202606"
    if "tea_dataset_v1_reviewed/修框后图片" in text or "tea_dataset_v1_reviewed/pool/images" in text:
        # Images with the same capture prefix stay together.  This prevents
        # repaired frames from one continuous video leaking across splits.
        stem = Path(text).stem
        prefix = re.split(r"__|_f\d|\d{4,}", stem, maxsplit=1)[0]
        prefix = re.sub(r"[^A-Za-z0-9一-龥]+", "_", prefix).strip("_") or "reviewed"
        return f"legacy_reviewed_{prefix}"
    if "tea_sop_side_transition_v1" in text:
        return "legacy_side_transition"
    fallback = re.sub(r"[^A-Za-z0-9]+", "_", Path(text).stem).strip("_").lower()
    if not fallback:
        fallback = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return "legacy_root_" + fallback


def video_metadata(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
    duration = frames / fps if fps > 0 else 0.0
    sharpness: list[float] = []
    if opened and frames > 0:
        for index in (0, frames // 2, max(0, frames - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()
    return {
        "readable": opened and frames > 0,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frames,
        "duration_s": round(duration, 3),
        "sample_laplacian_mean": round(sum(sharpness) / len(sharpness), 2) if sharpness else 0.0,
        "sample_laplacian_min": round(min(sharpness), 2) if sharpness else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(source_root: Path, output_root: Path) -> None:
    output = output_root / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    video_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    groups: dict[str, dict[str, object]] = {}

    for path in iter_files(source_root, VIDEO_SUFFIXES):
        relative = path.relative_to(source_root).as_posix()
        module, variant, camera = classify(relative)
        session = session_for(relative)
        metadata = video_metadata(path)
        row = {
            "kind": "video", "relative_path": relative, "session_id": session,
            "module": module, "variant": variant, "camera_hint": camera,
            "size_bytes": path.stat().st_size, "sha1": sha1_file(path),
            "review_status": "pending", "notes": "",
            **metadata,
        }
        video_rows.append(row)
        group = groups.setdefault(session, {"session_id": session, "source_count": 0, "kinds": set(), "modules": set()})
        group["source_count"] = int(group["source_count"]) + 1
        group["kinds"].add("video")
        group["modules"].add(module)

    # The reviewed folder contains MakeSense zip exports; the extracted,
    # image/label-aligned pool is the usable image source.  Prefer it while
    # retaining the zip folder as provenance in the notes.
    reviewed = source_root / "tea_dataset_v1_reviewed" / "pool" / "images"
    if not reviewed.is_dir():
        reviewed = source_root / "tea_dataset_v1_reviewed" / "修框后图片"
    if reviewed.is_dir():
        for path in iter_files(reviewed, IMAGE_SUFFIXES):
            relative = path.relative_to(source_root).as_posix()
            session = session_for(relative, is_image=True)
            row = {
                "kind": "image", "relative_path": relative, "session_id": session,
                "module": "legacy_reviewed_detection", "variant": "reviewed",
                "camera_hint": "front_or_unknown", "size_bytes": path.stat().st_size,
                "sha1": sha1_file(path), "review_status": "reviewed_pending_release", "notes": "",
            }
            image_rows.append(row)
            group = groups.setdefault(session, {"session_id": session, "source_count": 0, "kinds": set(), "modules": set()})
            group["source_count"] = int(group["source_count"]) + 1
            group["kinds"].add("image")
            group["modules"].add("legacy_reviewed_detection")

    fields = ["kind", "relative_path", "session_id", "module", "variant", "camera_hint", "size_bytes", "sha1", "readable", "width", "height", "fps", "frame_count", "duration_s", "sample_laplacian_mean", "sample_laplacian_min", "review_status", "notes"]
    write_csv(output / "all_materials.csv", video_rows + image_rows, fields)

    split_rows: list[dict[str, object]] = []
    for session in sorted(groups):
        # Existing and legacy material is train-only until manually approved.
        split = "train_candidate" if session.startswith("legacy_") else "pending"
        split_rows.append({"session_id": session, "split": split, "source_count": groups[session]["source_count"], "modules": ";".join(sorted(groups[session]["modules"])), "review_status": "pending", "notes": "同一session不得拆分"})
    write_csv(output / "source_groups.csv", split_rows, ["session_id", "split", "source_count", "modules", "review_status", "notes"])

    module_counts = Counter(row["module"] for row in video_rows)
    readable = sum(bool(row.get("readable")) for row in video_rows)
    image_fingerprints = defaultdict(list)
    for row in image_rows:
        image_fingerprints[row["sha1"]].append(row["relative_path"])
    duplicate_groups = [paths for paths in image_fingerprints.values() if len(paths) > 1]
    label_root = source_root / "tea_dataset_v1_reviewed" / "pool" / "labels" / "detect"
    class_file = source_root / "tea_dataset_v1_reviewed" / "classes.txt"
    label_instances: Counter[str] = Counter()
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    if reviewed.is_dir() and label_root.is_dir() and class_file.exists():
        class_names = class_file.read_text(encoding="utf-8-sig").splitlines()
        image_stems = {path.stem for path in iter_files(reviewed, IMAGE_SUFFIXES)}
        label_stems = set()
        for label in label_root.glob("*.txt"):
            if label.name in {"classes.txt", "labels.txt"}:
                continue
            label_stems.add(label.stem)
            for line in label.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                if line.strip():
                    class_id = line.split()[0]
                    label_instances[class_id] += 1
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        label_instances = Counter({
            class_names[int(class_id)] if class_id.isdigit() and int(class_id) < len(class_names) else class_id: count
            for class_id, count in label_instances.items()
        })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "video_count": len(video_rows), "reviewed_image_count": len(image_rows),
        "session_count": len(groups), "readable_video_count": readable,
        "video_modules": dict(sorted(module_counts.items())),
        "reviewed_label_instances": dict(sorted(label_instances.items())),
        "reviewed_images_missing_labels": missing_labels,
        "reviewed_orphan_labels": orphan_labels,
        "reviewed_exact_duplicate_groups": len(duplicate_groups),
        "reviewed_exact_duplicate_files": sum(len(paths) for paths in duplicate_groups),
        "train_candidate_sessions": sorted(s for s in groups if s.startswith("legacy_")),
        "pending_sessions": sorted(s for s in groups if not s.startswith("legacy_")),
        "warnings": [
            "所有视频/图片仍为候选池，未自动发布到train/val/test。",
            "new_front_full_202608的六个视频属于同一连续session。",
            "正式test需要全新、未参与训练的session；当前没有自动宣称正式指标。",
        ],
    }
    (output / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    template_path = output / "action_segments_template.jsonl"
    with template_path.open("w", encoding="utf-8") as handle:
        for row in video_rows:
            for action in ACTIONS:
                handle.write(json.dumps({
                    "source_relative_path": row["relative_path"],
                    "session_id": row["session_id"], "action_id": action,
                    "variant": "pending", "start_s": None, "end_s": None,
                    "review_status": "pending", "notes": "填写时间段后再进入动作数据集",
                }, ensure_ascii=False) + "\n")
    csv_template = output / "action_segments_review.csv"
    if not csv_template.exists():
        csv_template.write_text(
            "source_relative_path,session_id,action_id,variant,start_s,end_s,notes\n",
            encoding="utf-8-sig",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="建立全量茶艺素材清单，不移动或删除原始文件")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    build(args.source_root.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
