"""为离线视频生成带检测框和评分面板的结果视频。"""

import argparse
from pathlib import Path
import sys
import time

import cv2


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.detection_memory import DetectionMemory
from src.draw_utils import draw_detections, draw_info_panel
from src.item_matcher import ItemMatcher
from src.tea_detector import TeaDetector


DEFAULT_VIDEO = (
    PROJECT
    / "dataset"
    / "tea_sop_modular_v1"
    / "raw_videos"
    / "00_inbox"
    / "VID_20260612_094722.mp4"
)
DEFAULT_OUTPUT = PROJECT / "output" / "result_annotated.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成带茶具检测结果的离线视频")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO, help="输入视频")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 MP4")
    parser.add_argument("--width", type=int, default=1280, help="输出宽度")
    parser.add_argument("--height", type=int, default=720, help="输出高度")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not video_path.is_file():
        print(f"ERROR: 输入视频不存在: {video_path}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = TeaDetector(use_yolo=True)
    matcher = ItemMatcher()
    memory = DetectionMemory()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频: {video_path}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (max(1, args.width), max(1, args.height))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        out_size,
    )
    if not writer.isOpened():
        cap.release()
        print(f"ERROR: 无法创建输出视频: {output_path}")
        return 1

    print(f"Input: {video_path.name} ({orig_w}x{orig_h}, {total} frames, {fps_in:.1f} FPS)")
    print(f"Output: {output_path}")
    print("Processing...")

    frame_idx = 0
    started_at = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_inf = cv2.resize(frame, out_size)

        items = detector.detect(frame_inf)
        matched = matcher.match(items, frame_inf.shape[:2])
        memory.accumulate(matched, frame_idx)
        checklist = memory.get_checklist(matcher.items_config)
        detected_count, total_count, score = matcher.compute_score(checklist)
        grade, color = matcher.get_verdict(detected_count, total_count)

        annotated = draw_detections(frame_inf, matched)
        annotated = draw_info_panel(
            annotated,
            {
                "detected_count": detected_count,
                "total_count": total_count,
                "score": score,
                "grade": grade,
                "grade_color": color,
                "fps": fps_in,
                "frame_idx": frame_idx,
            },
        )
        writer.write(annotated)

        if frame_idx % 100 == 0:
            elapsed = time.time() - started_at
            eta = elapsed / frame_idx * max(0, total - frame_idx)
            print(f"  {frame_idx}/{total} ({100 * frame_idx / max(1, total):.0f}%) ETA: {eta:.0f}s")

    writer.release()
    cap.release()
    elapsed = time.time() - started_at
    speed = frame_idx / elapsed if elapsed > 0 else 0
    print(f"Done! {frame_idx} frames in {elapsed:.0f}s ({speed:.1f} FPS)")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
