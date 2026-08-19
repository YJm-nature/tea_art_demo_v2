"""Extract dense, quality-filtered frames from manually timed action segments.

The input CSV intentionally stores time ranges instead of treating an action as
a YOLO class.  This is suitable for short actions such as opening the gaiwan and
smelling, where uniform sampling over a several-minute video misses the event.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
DEFAULT_SEGMENTS = DEFAULT_ROOT / "manifests" / "action_segments_review.csv"
VIDEO_ROOT = PROJECT / "dataset"


def parse_time(value: str) -> float:
    """Accept seconds, MM:SS, or HH:MM:SS as used by video players."""
    parts = value.strip().replace("：", ":").split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) not in {2, 3}:
        raise ValueError(value)
    numbers = [float(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def dhash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def read_segments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if not row.get("source_relative_path") or not row.get("action_id"):
                continue
            if not row.get("start_s") or not row.get("end_s"):
                continue
            try:
                start = parse_time(row["start_s"])
                end = parse_time(row["end_s"])
            except ValueError:
                continue
            if start < 0 or end <= start:
                continue
            row["start_s"], row["end_s"] = str(start), str(end)
            rows.append(row)
        return rows


def extract_segment(
    video: Path,
    start_s: float,
    end_s: float,
    fps_sample: float,
    min_sharpness: float,
    min_hash_distance: int,
    max_frames: int,
) -> list[tuple[int, np.ndarray, float]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total <= 0:
        capture.release()
        return []
    start = max(0, int(round(start_s * fps)))
    end = min(total - 1, int(round(end_s * fps)))
    step = max(1, int(round(fps / fps_sample)))
    possible = list(range(start, end + 1, step))
    # Use extra quality candidates spread over the whole segment.  This keeps
    # long warm-clean/distribution segments from filling the quota only from
    # their first few seconds.
    candidate_limit = max_frames * 3
    if len(possible) > candidate_limit:
        target_indexes = np.linspace(0, len(possible) - 1, candidate_limit, dtype=np.int64)
        targets = {possible[int(index)] for index in target_indexes.tolist()}
    else:
        targets = set(possible)
    selected: list[tuple[int, np.ndarray, float]] = []
    hashes: list[np.ndarray] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    while index <= end:
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets:
            score = sharpness(frame)
            if score >= min_sharpness:
                current = dhash(frame)
                if not hashes or min(hamming(current, old) for old in hashes) >= min_hash_distance:
                    selected.append((index, frame, score))
                    hashes.append(current)
                    if len(selected) >= max_frames:
                        break
        index += 1
    capture.release()
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="按动作时间段密集抽取闻香等短动作帧")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--fps-sample", type=float, default=5.0)
    parser.add_argument("--min-sharpness", type=float, default=25.0)
    parser.add_argument("--min-hash-distance", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--action", action="append", default=[], help="只抽取指定action_id，可重复指定")
    args = parser.parse_args()
    root = args.root.resolve()
    segments = read_segments(args.segments.resolve())
    if args.action:
        enabled_actions = set(args.action)
        segments = [row for row in segments if row["action_id"] in enabled_actions]
    if not segments:
        raise SystemExit("没有有效动作片段。请在 action_segments_review.csv 填写 start_s 和 end_s。")

    output_root = root / "derived" / "action_pool"
    manifest_path = root / "manifests" / "action_frame_candidates.csv"
    fields = ["candidate_path", "source_relative_path", "session_id", "action_id", "variant", "frame_index", "time_s", "sharpness", "review_status", "split"]
    existing: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("candidate_path") and (root / row["candidate_path"]).exists():
                    existing[row["candidate_path"]] = row

    total = 0
    for segment in segments:
        source = VIDEO_ROOT / Path(segment["source_relative_path"])
        selected = extract_segment(
            source, float(segment["start_s"]), float(segment["end_s"]),
            args.fps_sample, args.min_sharpness, args.min_hash_distance, args.max_frames,
        )
        video_id = hashlib.sha1(segment["source_relative_path"].encode("utf-8")).hexdigest()[:10]
        metadata_capture = cv2.VideoCapture(str(source))
        source_fps = float(metadata_capture.get(cv2.CAP_PROP_FPS) or 1.0)
        metadata_capture.release()
        action = segment["action_id"]
        variant = segment.get("variant") or "positive"
        target_dir = output_root / segment["session_id"] / action / variant
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame, score in selected:
            name = f"{segment['session_id']}__{action}__{video_id}__f{frame_index:07d}.jpg"
            target = target_dir / name
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                continue
            relative = target.relative_to(root).as_posix()
            existing[relative] = {
                "candidate_path": relative,
                "source_relative_path": segment["source_relative_path"],
                "session_id": segment["session_id"], "action_id": action,
                "variant": variant, "frame_index": str(frame_index),
                "time_s": f"{frame_index / max(1.0, source_fps):.3f}",
                "sharpness": f"{score:.2f}", "review_status": "pending", "split": "pending",
            }
            total += 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing.values())
    print(f"本次抽取: {total}，累计动作候选帧: {len(existing)} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
