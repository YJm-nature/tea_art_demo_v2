"""Extract traceable, deduplicated detection frames from modular raw videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class FrameRecord:
    image_name: str
    video_id: str
    session_id: str
    source_path: str
    capture_group: str
    camera_role: str
    frame_index: int
    timestamp_ms: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    interval_sec: float
    review_status: str = "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/tea_sop_modular_v1"),
    )
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--dedup-distance",
        type=int,
        default=3,
        help="Reject a candidate when its dHash distance to the last kept frame is at most this value.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Read videos and report candidates without writing images."
    )
    parser.add_argument(
        "--scope",
        choices=("all", "inbox", "utensils"),
        default="all",
        help="Process all supported sources or only one source group.",
    )
    parser.add_argument(
        "--match",
        default="",
        help="Optional case-insensitive substring filter on the source filename.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Optional stable video id filter. Repeat to process multiple videos.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Override the inferred session id for all selected videos.",
    )
    return parser.parse_args()


def stable_video_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]


def extraction_policy(relative_path: str) -> tuple[float, str, str, str]:
    normalized = relative_path.replace("\\", "/")
    if "/01_utensils/01_single_object/" in normalized:
        return 2.5, "utensil_single", "single_oblique", "office_modular_s01"
    if "/01_utensils/02_grouped_objects/" in normalized:
        return 2.0, "utensil_grouped", "single_oblique", "office_modular_s01"
    if "/01_utensils/03_occlusion_handheld/" in normalized:
        return 1.0, "utensil_occlusion", "single_oblique", "office_modular_s01"
    if "/01_utensils/04_empty_negatives/" in normalized:
        return 3.0, "empty_negative", "single_oblique", "office_modular_s01"
    if "/00_inbox/" in normalized:
        return 3.0, "legacy_inbox", "mixed_review_required", "legacy_inbox_s01"
    raise ValueError(f"No extraction policy for {relative_path}")


def dhash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def resize_for_output(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_existing_manifest(path: Path) -> dict[str, FrameRecord]:
    records: dict[str, FrameRecord] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records[data["image_name"]] = FrameRecord(**data)
    return records


def write_manifest(path: Path, records: dict[str, FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name in sorted(records):
            handle.write(json.dumps(asdict(records[name]), ensure_ascii=False) + "\n")


def extract_video(
    video_path: Path,
    dataset_root: Path,
    output_dir: Path,
    records: dict[str, FrameRecord],
    max_width: int,
    jpeg_quality: int,
    dedup_distance: int,
    dry_run: bool,
    session_override: str = "",
) -> dict[str, object]:
    relative_path = video_path.relative_to(dataset_root).as_posix()
    interval_sec, capture_group, camera_role, session_id = extraction_policy(relative_path)
    if session_override:
        session_id = session_override
    video_id = stable_video_id(relative_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "relative_path": relative_path, "error": "open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return {"video_id": video_id, "relative_path": relative_path, "error": "invalid_metadata"}

    previous_image_names = {
        name for name, record in records.items() if record.video_id == video_id
    }
    for name in previous_image_names:
        records.pop(name)
    current_image_names: set[str] = set()

    duration_sec = frame_count / fps
    timestamps = np.arange(0.0, duration_sec, interval_sec)
    last_hash: int | None = None
    candidates = kept = duplicates = read_failures = 0

    # Decode sequentially. Random seeking is prohibitively slow for large 4K phone files.
    target_indices = [min(frame_count - 1, round(timestamp_sec * fps)) for timestamp_sec in timestamps]
    target_cursor = 0
    frame_index = -1
    while target_cursor < len(target_indices):
        ok = cap.grab()
        frame_index += 1
        if not ok:
            read_failures += 1
            break
        if frame_index < target_indices[target_cursor]:
            continue

        ok, frame = cap.retrieve()
        if not ok:
            read_failures += 1
            target_cursor += 1
            continue

        timestamp_sec = frame_index / fps
        candidates += 1

        current_hash = dhash(frame)
        if last_hash is not None and hamming(last_hash, current_hash) <= dedup_distance:
            duplicates += 1
            target_cursor += 1
            continue
        last_hash = current_hash

        image_name = f"{video_id}__f{frame_index:08d}_ms{round(timestamp_sec * 1000):09d}.jpg"
        output = resize_for_output(frame, max_width)
        output_height, output_width = output.shape[:2]
        record = FrameRecord(
            image_name=image_name,
            video_id=video_id,
            session_id=session_id,
            source_path=relative_path,
            capture_group=capture_group,
            camera_role=camera_role,
            frame_index=frame_index,
            timestamp_ms=round(timestamp_sec * 1000),
            source_width=source_width,
            source_height=source_height,
            output_width=output_width,
            output_height=output_height,
            interval_sec=interval_sec,
        )
        records[image_name] = record
        current_image_names.add(image_name)
        if not dry_run:
            target = output_dir / image_name
            if not target.exists():
                cv2.imwrite(
                    str(target), output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
        kept += 1
        target_cursor += 1

    cap.release()
    if not dry_run:
        for obsolete_name in previous_image_names - current_image_names:
            obsolete_path = output_dir / obsolete_name
            if obsolete_path.is_file():
                obsolete_path.unlink()
    return {
        "video_id": video_id,
        "relative_path": relative_path,
        "session_id": session_id,
        "capture_group": capture_group,
        "interval_sec": interval_sec,
        "duration_sec": round(duration_sec, 2),
        "candidates": candidates,
        "kept": kept,
        "near_duplicates_rejected": duplicates,
        "read_failures": read_failures,
    }


def update_video_manifest(dataset_root: Path, summaries: list[dict[str, object]]) -> None:
    path = dataset_root / "manifests" / "videos.csv"
    fieldnames = [
        "video_id",
        "session_id",
        "relative_path",
        "capture_group",
        "sop_step",
        "observation_ids",
        "camera_role",
        "sample_kind",
        "operator",
        "recorded_at",
        "resolution",
        "fps",
        "items_present",
        "review_status",
        "notes",
    ]
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("video_id"):
                    existing[row["video_id"]] = row

    for summary in summaries:
        video_id = str(summary.get("video_id", ""))
        if not video_id or summary.get("error"):
            continue
        row = existing.get(video_id, {name: "" for name in fieldnames})
        row.update(
            {
                "video_id": video_id,
                "session_id": str(summary["session_id"]),
                "relative_path": str(summary["relative_path"]),
                "capture_group": str(summary["capture_group"]),
                "camera_role": "mixed_review_required"
                if summary["capture_group"] == "legacy_inbox"
                else "single_oblique",
                "sample_kind": "unlabeled_detection_source",
                "review_status": "preview_reviewed",
                "notes": f"extracted={summary['kept']}; interval={summary['interval_sec']}s",
            }
        )
        existing[video_id] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing[key] for key in sorted(existing))


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    raw_root = dataset_root / "raw_videos"
    output_dir = dataset_root / "derived" / "detection_pool" / "images"
    manifest_path = dataset_root / "manifests" / "frame_manifest.jsonl"
    report_path = dataset_root / "reports" / "detection_extraction_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and ("00_inbox" in path.parts or "01_utensils" in path.parts)
    )
    if args.scope == "inbox":
        videos = [path for path in videos if "00_inbox" in path.parts]
    elif args.scope == "utensils":
        videos = [path for path in videos if "01_utensils" in path.parts]
    if args.match:
        needle = args.match.casefold()
        videos = [path for path in videos if needle in path.name.casefold()]
    if args.video_id:
        selected_ids = set(args.video_id)
        videos = [
            path
            for path in videos
            if stable_video_id(path.relative_to(dataset_root).as_posix()) in selected_ids
        ]
    records = load_existing_manifest(manifest_path)
    summaries = []
    for index, video_path in enumerate(videos, start=1):
        summary = extract_video(
            video_path,
            dataset_root,
            output_dir,
            records,
            args.max_width,
            args.jpeg_quality,
            args.dedup_distance,
            args.dry_run,
            args.session_id,
        )
        summaries.append(summary)
        print(f"[{index:02d}/{len(videos):02d}] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    if not args.dry_run:
        write_manifest(manifest_path, records)
        update_video_manifest(dataset_root, summaries)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        merged_summaries: dict[str, dict[str, object]] = {}
        if report_path.exists():
            try:
                previous = json.loads(report_path.read_text(encoding="utf-8"))
                merged_summaries.update(
                    {
                        str(item["video_id"]): item
                        for item in previous
                        if item.get("video_id")
                    }
                )
            except (json.JSONDecodeError, TypeError):
                pass
        merged_summaries.update(
            {
                str(item["video_id"]): item
                for item in summaries
                if item.get("video_id")
            }
        )
        report_path.write_text(
            json.dumps(
                [merged_summaries[key] for key in sorted(merged_summaries)],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    total_kept = sum(int(item.get("kept", 0)) for item in summaries)
    total_rejected = sum(int(item.get("near_duplicates_rejected", 0)) for item in summaries)
    print(f"videos={len(videos)} kept={total_kept} near_duplicates_rejected={total_rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
