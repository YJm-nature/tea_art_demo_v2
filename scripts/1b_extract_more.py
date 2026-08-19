"""从多个项目内视频补充提取清晰、去重的候选帧。"""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"
DEFAULT_OUTPUT = PROJECT / "dataset" / "images"
DEFAULT_VIDEOS = (
    (INBOX / "VID_20260612_093545.mp4", 100),
    (INBOX / "已加速- IMG_4901.MOV", 60),
    (INBOX / "VID_20260612_095428.mp4", 25),
    (INBOX / "VID_20260612_094722.mp4", 12),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从多个视频补充提取清晰、去重帧")
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=None,
        help="输入视频，可重复传入；指定后覆盖项目内默认视频列表",
    )
    parser.add_argument("--step", type=int, default=30, help="自定义视频的采样帧间隔")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图片输出目录")
    parser.add_argument("--count", type=int, default=150, help="本次最多新增图片数")
    parser.add_argument("--width", type=int, default=1280, help="输出图片宽度")
    parser.add_argument("--similarity", type=float, default=0.92, help="相似帧跳过阈值")
    parser.add_argument("--blur", type=float, default=100.0, help="清晰度最低阈值")
    return parser.parse_args()


def _next_frame_id(output_dir: Path) -> int:
    ids = []
    for image in output_dir.glob("frame_*.jpg"):
        try:
            ids.append(int(image.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(ids) + 1 if ids else 0


def extract_from_videos(
    videos: list[tuple[Path, int]],
    output_dir: Path,
    target_count: int,
    target_width: int,
    similarity_threshold: float,
    blur_threshold: float,
) -> int:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_id = _next_frame_id(output_dir)
    print(f"已有图片: {len(list(output_dir.glob('frame_*.jpg')))}，起始编号: {start_id}")

    extracted = 0
    last_gray = None
    for raw_video_path, step in videos:
        if extracted >= target_count:
            break
        video_path = raw_video_path.expanduser().resolve()
        if not video_path.is_file():
            print(f"跳过不存在的视频: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"跳过无法打开的视频: {video_path}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total / fps if fps > 0 else 0
        print(f"\n[{video_path.name}] {total} 帧, {duration:.0f}s, step={step}")

        frame_idx = 0
        while extracted < target_count and frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            scale = target_width / width
            frame_small = cv2.resize(frame, (target_width, int(height * scale)))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:
                frame_idx += step
                continue

            gray_tiny = cv2.resize(gray, (128, 128))
            if last_gray is not None:
                correlation = np.corrcoef(gray_tiny.ravel(), last_gray.ravel())[0, 1]
                if correlation > similarity_threshold:
                    frame_idx += step
                    continue

            last_gray = gray_tiny
            output_path = output_dir / f"frame_{start_id + extracted:04d}.jpg"
            cv2.imwrite(str(output_path), frame_small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
            if extracted % 30 == 0:
                print(f"  {extracted}/{target_count}")
            frame_idx += step
        cap.release()

    print(f"\n完成，本次新增 {extracted} 张图片")
    print(f"输出目录: {output_dir}")
    return extracted


def main() -> int:
    args = parse_args()
    if args.video:
        videos = [(path, max(1, args.step)) for path in args.video]
    else:
        videos = list(DEFAULT_VIDEOS)
    extract_from_videos(
        videos=videos,
        output_dir=args.output,
        target_count=max(1, args.count),
        target_width=max(1, args.width),
        similarity_threshold=args.similarity,
        blur_threshold=args.blur,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
