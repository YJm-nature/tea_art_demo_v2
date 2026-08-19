"""从单个视频中提取清晰、去重的代表帧。"""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    PROJECT
    / "dataset"
    / "tea_sop_modular_v1"
    / "raw_videos"
    / "00_inbox"
    / "茶具检测.MP4"
)
DEFAULT_OUTPUT = PROJECT / "dataset" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从视频中提取清晰、去重的标注候选帧")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="输入视频")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图片输出目录")
    parser.add_argument("--count", type=int, default=150, help="目标图片数量")
    parser.add_argument("--width", type=int, default=1280, help="输出图片宽度")
    parser.add_argument("--similarity", type=float, default=0.92, help="相似帧跳过阈值")
    parser.add_argument("--blur", type=float, default=100.0, help="清晰度最低阈值")
    return parser.parse_args()


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_count: int,
    target_width: int,
    similarity_threshold: float,
    blur_threshold: float,
) -> int:
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    step = max(1, total_frames // max(1, target_count * 3))

    print(f"视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps:.1f}, 时长: {duration:.0f}s")
    print(f"目标: 提取 {target_count} 张去重帧")

    extracted = 0
    last_gray = None
    frame_idx = 0
    while extracted < target_count and frame_idx < total_frames:
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
        output_path = output_dir / f"frame_{extracted:04d}.jpg"
        cv2.imwrite(str(output_path), frame_small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted += 1
        if extracted % 25 == 0:
            print(f"  已提取 {extracted}/{target_count} 张")
        frame_idx += step

    cap.release()
    print(f"完成，共提取 {extracted} 张图片")
    print(f"输出目录: {output_dir}")
    return extracted


def main() -> int:
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        target_count=max(1, args.count),
        target_width=max(1, args.width),
        similarity_threshold=args.similarity,
        blur_threshold=args.blur,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
