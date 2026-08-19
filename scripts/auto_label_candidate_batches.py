"""Generate YOLO pre-labels for candidate annotation batches.

Pre-labels are intentionally written beside each batch and never replace the
reviewed legacy dataset.  They are a starting point for MakeSense review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = PROJECT / "dataset" / "tea_sop_front_v1" / "annotation_batches" / "detection_v1"
DEFAULT_MODEL = PROJECT / "models" / "low_vram" / "current10_640_20260731" / "weights" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_result(result, target: Path) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    boxes = result.boxes
    count = 0
    low_conf = 0
    lines: list[str] = []
    if boxes is not None and len(boxes) > 0:
        xywhn = boxes.xywhn.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        for coords, class_id, confidence in zip(xywhn, classes, confidences):
            if confidence < 0.20:
                low_conf += 1
            values = [int(class_id), *[max(0.0, min(1.0, float(value))) for value in coords]]
            lines.append(f"{values[0]} " + " ".join(f"{value:.6f}" for value in values[1:]))
            count += 1
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return count, low_conf


def main() -> int:
    parser = argparse.ArgumentParser(description="使用18类茶具模型为MakeSense批次生成预标注")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = YOLO(str(model_path))
    names = {int(key): value for key, value in model.names.items()} if isinstance(model.names, dict) else dict(enumerate(model.names))
    summary: list[dict[str, object]] = []
    for batch in sorted(path for path in batch_root.glob("batch_*") if path.is_dir()):
        image_dir = batch / "01_images"
        if not image_dir.is_dir():
            continue
        output_dir = batch / "02_auto_labels"
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"预标注目录非空，拒绝覆盖: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        total_boxes = 0
        total_low_conf = 0
        results = model.predict(
            source=[str(path) for path in images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            stream=True,
        )
        processed = 0
        for image, result in zip(images, results):
            image_name = image.name
            boxes, low_conf = write_result(result, output_dir / f"{Path(image_name).stem}.txt")
            total_boxes += boxes
            total_low_conf += low_conf
            processed += 1
        if processed != len(images):
            raise RuntimeError(f"{batch.name}: 推理结果不完整 expected={len(images)} actual={processed}")
        (batch / "02_auto_labels" / "labels.txt").write_text(
            "\n".join(names[index] for index in sorted(names)) + "\n", encoding="utf-8"
        )
        row = {
            "batch": batch.name, "images": len(images), "boxes": total_boxes,
            "low_confidence_boxes": total_low_conf, "status": "needs_manual_review",
        }
        (batch / "auto_label_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))
    report = {"model": str(model_path), "conf": args.conf, "imgsz": args.imgsz, "batches": summary, "status": "needs_manual_review"}
    (batch_root / "auto_label_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成预标注: {sum(int(row['images']) for row in summary)}张图片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
