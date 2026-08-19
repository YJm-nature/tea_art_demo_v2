"""Build a small, review-first side-view detector supplement."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos"
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_side_transition_v1"
DEFAULT_MODEL = PROJECT / "models" / "low_vram" / "current10_640_20260731" / "weights" / "best.pt"
ACTIVE_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14]
DEFERRED_CLASS_IDS = [8, 9, 11, 12, 13, 15, 16, 17]

# This is intentionally a small review batch. All sources remain train-only
# supplements when the final front-camera dataset is assembled.
SOURCE_PLAN = (
    ("01_utensils/03_occlusion_handheld/茶具展示并遮挡.mp4", 50, "general_occlusion", "office_side_s01"),
    ("01_utensils/03_occlusion_handheld/盖碗开合补充.mp4", 40, "gaiwan_open_close", "office_side_s02"),
    ("01_utensils/03_occlusion_handheld/闻香补充.mp4", 30, "smell_positive", "office_side_s02"),
    ("01_utensils/03_occlusion_handheld/闻香负例.mp4", 30, "smell_negative", "office_side_s02"),
    ("02_sop_steps/step03_tea_preparation/positive/双手托举茶荷.mp4", 30, "hold_positive", "office_side_s02"),
    ("02_sop_steps/step03_tea_preparation/error/单手托举等茶荷负例.mp4", 30, "hold_negative", "office_side_s02"),
    ("02_sop_steps/step06_serve/杯位布局.mp4", 30, "cup_layout", "office_side_s02"),
    ("01_utensils/01_single_object/茶夹.mp4", 25, "tea_tongs", "office_side_s01"),
    ("01_utensils/01_single_object/茶拨补充.mp4", 35, "tea_pick", "office_side_s02"),
    ("01_utensils/01_single_object/公道杯近景.mp4", 20, "fairness_pitcher", "office_side_s01"),
    ("01_utensils/01_single_object/茶荷近景.mp4", 15, "tea_lotus", "office_side_s01"),
    ("01_utensils/02_grouped_objects/相似的茶夹茶拨茶荷.mp4", 20, "small_tools_group", "office_side_s01"),
    ("01_utensils/01_single_object/茶巾近景.mp4", 10, "tea_towel", "office_side_s01"),
    ("01_utensils/01_single_object/茶叶罐.mp4", 10, "tea_canister", "office_side_s01"),
    ("01_utensils/01_single_object/建水.mp4", 10, "waste_bowl", "office_side_s01"),
    ("01_utensils/04_empty_negatives/empty_table.mp4", 5, "empty_negative", "office_side_s01"),
    ("01_utensils/04_empty_negatives/empty_table_2.mp4", 5, "empty_negative", "office_side_s01"),
)

CLASS_THRESHOLDS = {
    0: 0.20, 1: 0.20, 2: 0.18, 3: 0.20, 4: 0.20,
    5: 0.20, 6: 0.08, 7: 0.10, 10: 0.20, 14: 0.20,
}

CANONICAL_NAMES = {
    0: "\u76d6\u7897\u7897\u8eab", 1: "\u76d6\u7897\u7897\u76d6", 2: "\u516c\u9053\u676f", 3: "\u54c1\u8317\u676f",
    4: "\u8336\u8377", 5: "\u8336\u5dfe", 6: "\u8336\u5939", 7: "\u8336\u62e8", 8: "\u8336\u76d8",
    9: "\u70e7\u6c34\u58f6", 10: "\u5efa\u6c34", 11: "\u7535\u5b50\u79e4", 12: "\u6e29\u5ea6\u8ba1",
    13: "\u8ba1\u65f6\u5668", 14: "\u8336\u53f6\u7f50", 15: "\u8336\u5219", 16: "\u6c34\u58f6\u663e\u793a\u5c4f",
    17: "\u7535\u5b50\u79e4\u663e\u793a\u5c4f",
}


@dataclass
class FrameRecord:
    sample_id: str
    image: str
    detect_label: str
    source_video: str
    source_group: str
    session_id: str
    frame_index: int
    timestamp_seconds: float
    split_policy: str = "train_only_supplement"
    review_status: str = "pending"


def ontology_names() -> dict[int, str]:
    # Keep the fixed numeric ontology, even if an older YAML was saved with
    # a damaged console encoding. Labels are numeric; names are only for UI.
    return dict(CANONICAL_NAMES)


def dhash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def resolve_source(relative: str, source_group: str) -> Path:
    """Resolve source videos by semantic group so mojibake in old manifests cannot break extraction."""
    direct = SOURCE_ROOT / Path(relative)
    if direct.is_file():
        return direct
    patterns = {
        "general_occlusion": "01_utensils/03_occlusion_handheld/\u8336\u5177\u5c55\u793a\u5e76\u906e\u6321.mp4",
        "gaiwan_open_close": "01_utensils/03_occlusion_handheld/\u76d6\u7897\u5f00\u5408\u8865\u5145.mp4",
        "smell_positive": "01_utensils/03_occlusion_handheld/\u95fb\u9999\u8865\u5145.mp4",
        "smell_negative": "01_utensils/03_occlusion_handheld/\u95fb\u9999\u8d1f\u4f8b.mp4",
        "hold_positive": "02_sop_steps/step03_tea_preparation/positive/\u53cc\u624b\u6258\u4e3e\u8336\u8377.mp4",
        "hold_negative": "02_sop_steps/step03_tea_preparation/error/\u5355\u624b\u6258\u4e3e\u7b49\u8336\u8377\u8d1f\u4f8b.mp4",
        "cup_layout": "02_sop_steps/step06_serve/\u676f\u4f4d\u5e03\u5c40.mp4",
        "tea_tongs": "01_utensils/01_single_object/\u8336\u5939.mp4",
        "tea_pick": "01_utensils/01_single_object/\u8336\u62e8\u8865\u5145.mp4",
        "fairness_pitcher": "01_utensils/01_single_object/\u516c\u9053\u676f\u8fd1\u666f.mp4",
        "tea_lotus": "01_utensils/01_single_object/\u8336\u8377\u8fd1\u666f.mp4",
        "small_tools_group": "01_utensils/02_grouped_objects/\u76f8\u4f3c\u7684\u8336\u5939\u8336\u62e8\u8336\u8377.mp4",
        "tea_towel": "01_utensils/01_single_object/\u8336\u5dfe\u8fd1\u666f.mp4",
        "tea_canister": "01_utensils/01_single_object/\u8336\u53f6\u7f50.mp4",
        "waste_bowl": "01_utensils/01_single_object/\u5efa\u6c34.mp4",
    }
    candidate = SOURCE_ROOT / patterns.get(source_group, relative)
    if candidate.is_file():
        return candidate
    return direct


def extract(root: Path, max_width: int = 1920) -> dict[str, Any]:
    image_dir = root / "pool" / "images"
    label_dir = root / "pool" / "labels" / "detect"
    manifest_path = root / "manifests" / "frames.jsonl"
    if image_dir.exists() and any(image_dir.iterdir()):
        raise FileExistsError(f"抽帧目录非空，拒绝覆盖: {image_dir}")
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    names = ontology_names()
    (root / "classes.txt").write_text(
        "\n".join(names[index] for index in range(len(names))) + "\n", encoding="utf-8"
    )

    records: list[FrameRecord] = []
    source_summaries = []
    for relative, quota, group, session in SOURCE_PLAN:
        video = resolve_source(relative, group)
        if not video.is_file():
            source_summaries.append({"source": relative, "error": "missing"})
            continue
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or frame_count <= 0:
            capture.release()
            source_summaries.append({"source": relative, "error": "invalid_video"})
            continue
        margin = min(round(fps * 2), max(0, frame_count // 10))
        candidates = np.linspace(margin, max(margin, frame_count - margin - 1), quota * 2, dtype=int)
        target_indices = sorted(set(int(value) for value in candidates))
        target_cursor = 0
        frame_index = -1
        previous_hash = None
        kept = duplicates = 0
        video_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        while target_cursor < len(target_indices) and kept < quota:
            ok = capture.grab()
            frame_index += 1
            if not ok:
                break
            if frame_index < target_indices[target_cursor]:
                continue
            ok, frame = capture.retrieve()
            target_cursor += 1
            if not ok:
                continue
            current_hash = dhash(frame)
            if previous_hash is not None and (previous_hash ^ current_hash).bit_count() <= 2:
                duplicates += 1
                continue
            previous_hash = current_hash
            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
            sample_id = f"{session}__{group}__{video_id}__f{frame_index:08d}"
            image_path = image_dir / f"{sample_id}.jpg"
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                continue
            label_path = label_dir / f"{sample_id}.txt"
            label_path.touch()
            records.append(FrameRecord(
                sample_id=sample_id,
                image=image_path.relative_to(root).as_posix(),
                detect_label=label_path.relative_to(root).as_posix(),
                source_video=(Path("raw_videos") / video.relative_to(SOURCE_ROOT)).as_posix(),
                source_group=group,
                session_id=session,
                frame_index=frame_index,
                timestamp_seconds=round(frame_index / fps, 3),
            ))
            kept += 1
        capture.release()
        source_summaries.append({
            "source": video.relative_to(SOURCE_ROOT).as_posix(), "quota": quota, "kept": kept,
            "duplicates": duplicates, "session_id": session, "source_group": group,
        })
        print(f"{group}: {kept}/{quota}", flush=True)

    manifest_path.write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "purpose": "train_only_side_supplement",
        "images": len(records),
        "planned_images": sum(row[1] for row in SOURCE_PLAN),
        "sessions": sorted({record.session_id for record in records}),
        "active_class_ids": ACTIVE_CLASS_IDS,
        "deferred_class_ids": DEFERRED_CLASS_IDS,
        "sources": source_summaries,
    }
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def pseudo_label(root: Path, model_path: Path, imgsz: int = 640) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    manifest_path = root / "manifests" / "frames.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("抽帧清单为空")
    from ultralytics import YOLO
    model = YOLO(str(model_path.resolve()))
    counts: Counter[int] = Counter()
    empty_images = 0
    for index, record in enumerate(records, 1):
        image = root / record["image"]
        prediction = model.predict(
            str(image), imgsz=imgsz, conf=min(CLASS_THRESHOLDS.values()), iou=0.50,
            classes=ACTIVE_CLASS_IDS, device=0, verbose=False,
        )[0]
        height, width = prediction.orig_shape
        rows = []
        if prediction.boxes is not None:
            for values in prediction.boxes.data.detach().cpu().tolist():
                x1, y1, x2, y2, confidence, class_id_value = values
                class_id = int(class_id_value)
                if confidence < CLASS_THRESHOLDS[class_id]:
                    continue
                rows.append((
                    class_id,
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ))
                counts[class_id] += 1
        label = root / record["detect_label"]
        label.write_text(
            "".join(
                f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                for class_id, x, y, w, h in rows
            ),
            encoding="utf-8",
        )
        record["review_status"] = "needs_fix"
        if not rows:
            empty_images += 1
        if index % 25 == 0 or index == len(records):
            print(f"pseudo-label: {index}/{len(records)}", flush=True)
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    names = ontology_names()
    summary = {
        "model": str(model_path.resolve()),
        "images": len(records),
        "empty_or_missed_images": empty_images,
        "review_status": "needs_fix",
        "class_instances": {
            str(class_id): {"name": names[class_id], "instances": counts[class_id]}
            for class_id in ACTIVE_CLASS_IDS
        },
        "warning": "Pseudo-labels only reduce drawing work; every missing or incorrect box must be reviewed.",
    }
    (root / "reports" / "pseudo_label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def review_batch(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifests" / "frames.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("抽帧清单为空")
    output = root / "review_batches" / "side_transition_all"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    images = output / "01_images"
    labels = output / "02_yolo_import"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for record in records:
        source_image = root / record["image"]
        source_label = root / record["detect_label"]
        try:
            os.link(source_image, images / source_image.name)
        except OSError:
            shutil.copy2(source_image, images / source_image.name)
        shutil.copy2(source_label, labels / source_label.name)
    shutil.copy2(root / "classes.txt", labels / "labels.txt")
    archive = output / "yolo_labels_import.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(labels.glob("*.txt")):
            bundle.write(path, path.name)
    batch_manifest = output / "batch_manifest.json"
    batch_manifest.write_text(
        json.dumps(
            {"sample_ids": [record["sample_id"] for record in records], "images": len(records)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "images": len(records),
        "image_dir": str(images),
        "label_dir": str(labels),
        "label_archive": str(archive),
        "batch_manifest": str(batch_manifest),
        "instructions": "MakeSense加载01_images全部图片，再导入02_yolo_import中的全部txt并逐图修框",
    }
    (output / "batch_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def refresh_batch_metadata(root: Path) -> dict[str, Any]:
    """Repair metadata for an already-created batch without touching labels."""
    manifest_path = root / "manifests" / "frames.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    output = root / "review_batches" / "side_transition_all"
    if not records or not output.is_dir():
        raise FileNotFoundError("Review batch has not been created")
    batch_manifest = output / "batch_manifest.json"
    batch_manifest.write_text(
        json.dumps({"sample_ids": [record["sample_id"] for record in records], "images": len(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"images": len(records), "batch_manifest": str(batch_manifest)}


def normalize_manifest(root: Path) -> dict[str, Any]:
    """Update source paths after repairing legacy filename encoding."""
    path = root / "manifests" / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        relative = next((row[0] for row in SOURCE_PLAN if row[2] == record.get("source_group")), record.get("source_video", ""))
        video = resolve_source(relative, record.get("source_group", ""))
        if video.is_file():
            record["source_video"] = (Path("raw_videos") / video.relative_to(SOURCE_ROOT)).as_posix()
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
    (root / "manifest.jsonl").write_text(payload, encoding="utf-8")
    return {"records": len(records), "manifest": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="准备侧面视频YOLOv8n过渡训练数据")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--max-width", type=int, default=1920)
    label_parser = subparsers.add_parser("pseudo-label")
    label_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    label_parser.add_argument("--imgsz", type=int, default=640)
    subparsers.add_parser("review-batch")
    subparsers.add_parser("refresh-batch-metadata")
    subparsers.add_parser("normalize-manifest")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "extract":
        report = extract(root, args.max_width)
    elif args.command == "pseudo-label":
        report = pseudo_label(root, args.model.resolve(), args.imgsz)
    elif args.command == "review-batch":
        report = review_batch(root)
    elif args.command == "refresh-batch-metadata":
        report = refresh_batch_metadata(root)
    else:
        report = normalize_manifest(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
