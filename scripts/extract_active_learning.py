"""从补采视频提取去重、清晰且具有状态变化/模型不确定性的候选帧。"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="主动学习候选帧抽取")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="可选YOLO权重，用于不确定性排序")
    parser.add_argument("--interval", type=float, default=1.0, help="候选采样间隔秒")
    parser.add_argument("--max-per-video", type=int, default=500)
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.985)
    parser.add_argument("--phash-gap", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖: {output}")
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(str(args.model.resolve()))

    manifest = []
    for video in args.videos:
        manifest.extend(extract_video(video.resolve(), image_dir, model, args))
    (output / "candidate_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    summary = {
        "videos": len(args.videos),
        "candidates": len(manifest),
        "with_model_uncertainty": model is not None,
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def extract_video(video: Path, image_dir: Path, model, args) -> list:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise ValueError(f"视频FPS无效: {video}")
    step = max(1, int(round(fps * args.interval)))
    previous_tiny = None
    hashes = []
    records = []
    session = _safe_name(video.stem)

    for frame_index in range(0, total_frames, step):
        if len(records) >= args.max_per_video:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < args.blur_threshold:
            continue
        tiny = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        similarity = _correlation(previous_tiny, tiny)
        previous_tiny = tiny

        uncertainty, detections = _model_uncertainty(model, frame)
        state_changed = similarity < args.similarity_threshold
        if not state_changed and uncertainty < 0.45:
            continue
        phash = _phash(gray)
        if any((phash ^ previous).bit_count() <= args.phash_gap for previous in hashes):
            continue
        hashes.append(phash)

        timestamp_ms = int(frame_index / fps * 1000)
        stem = f"{session}__f{frame_index:08d}_ms{timestamp_ms:09d}"
        output = image_dir / f"{stem}.jpg"
        cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({
            "sample_id": stem,
            "session_id": session,
            "source_video": str(video),
            "frame_number": frame_index,
            "timestamp_ms": timestamp_ms,
            "image": output.name,
            "blur_score": round(blur, 2),
            "previous_similarity": round(similarity, 5),
            "model_uncertainty": round(uncertainty, 4),
            "detections": detections,
            "label_status": "unlabeled",
        })
    capture.release()
    return records


def _model_uncertainty(model, frame):
    if model is None:
        return 0.0, 0
    result = model.predict(frame, imgsz=640, conf=0.15, device=0, verbose=False)[0]
    confidences = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.array([])
    if len(confidences) == 0:
        return 0.8, 0
    uncertainty = float(np.mean(1.0 - np.abs(confidences - 0.5) * 2.0))
    return max(0.0, min(1.0, uncertainty)), int(len(confidences))


def _correlation(previous, current):
    if previous is None:
        return 0.0
    value = np.corrcoef(previous.flatten(), current.flatten())[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _phash(gray):
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    values = cv2.dct(np.float32(resized))[:8, :8].flatten()[1:]
    bits = values > np.median(values)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
