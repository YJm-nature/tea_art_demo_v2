"""Extract the first detection/pose annotation batches from one front session."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import cv2


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def videos_under(path: Path) -> list[Path]:
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in VIDEO_SUFFIXES]


def allocate(total: int, videos: Iterable[Path]) -> list[tuple[Path, int]]:
    rows = list(videos)
    if not rows or total <= 0:
        return []
    base, remainder = divmod(total, len(rows))
    return [(path, base + (1 if index < remainder else 0)) for index, path in enumerate(rows)]


def extract(rows: list[tuple[Path, int]], output: Path, session_id: str, prefix: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for video, requested in rows:
        if requested <= 0:
            continue
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or frame_count <= 0:
            capture.release()
            print(f"跳过无法读取的视频: {video}")
            continue
        margin = min(int(capture.get(cv2.CAP_PROP_FPS) * 2), max(0, frame_count // 10))
        start, end = margin, max(margin, frame_count - margin - 1)
        positions = [
            int(round(start + (end - start) * index / max(1, requested - 1)))
            for index in range(requested)
        ]
        video_id = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:8]
        for frame_index in sorted(set(positions)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            target = output / f"{session_id}__{prefix}__{video_id}__f{frame_index:07d}.jpg"
            if cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                written += 1
        capture.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="抽取正面数据首轮标注图片")
    parser.add_argument("session", help="例如 front_s01")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--detection-limit", type=int, default=100)
    parser.add_argument("--pose-limit", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()
    session_root = root / "raw_videos" / args.session
    if not session_root.is_dir():
        raise FileNotFoundError(session_root)

    all_videos = videos_under(session_root)
    pose_videos = videos_under(session_root / "04_pose")
    pose_videos.extend(videos_under(session_root / "02_actions" / "water_injection"))
    pose_videos.extend(videos_under(session_root / "02_actions" / "gaiwan_to_pitcher"))
    pose_videos.extend(videos_under(session_root / "02_actions" / "tea_distribution"))
    pose_videos.extend(videos_under(session_root / "02_actions" / "tea_lotus_to_gaiwan"))
    if not all_videos:
        raise ValueError(f"session内没有视频: {session_root}")

    detection_count = extract(
        allocate(args.detection_limit, all_videos),
        root / "derived" / "detection" / "images",
        args.session,
        "detect",
    )
    pose_count = extract(
        allocate(args.pose_limit, sorted(set(pose_videos))),
        root / "derived" / "pose" / "images",
        args.session,
        "pose",
    )
    print(f"检测待标注图片: {detection_count}")
    print(f"关键点待标注图片: {pose_count}")
    if pose_count == 0:
        print("提示: 04_pose和四类倾倒动作目录中尚无视频")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

