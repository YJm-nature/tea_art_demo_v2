"""Run the improved YOLO model on remaining review batches."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = PROJECT / "dataset" / "tea_sop_front_v1" / "annotation_batches" / "detection_v2_100"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_labels(result, target: Path) -> int:
    lines: list[str] = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xywhn = boxes.xywhn.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        rows = sorted(zip(confidences, classes, xywhn), key=lambda row: row[0], reverse=True)
        for confidence, class_id, coords in rows:
            if confidence < 0.20:
                continue
            values = [max(0.0, min(1.0, float(value))) for value in coords]
            lines.append(f"{int(class_id)} " + " ".join(f"{value:.6f}" for value in values))
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="用优化后的模型重做剩余批次预标注")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--first-batch", type=int, default=7)
    parser.add_argument("--last-batch", type=int, default=13)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--keep-old", action="store_true", help="不备份旧预标注，直接写入")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_images = 0
    total_boxes = 0
    for number in range(args.first_batch, args.last_batch + 1):
        batch = args.batch_root.resolve() / f"batch_{number:03d}"
        image_dir = batch / "01_images"
        output_dir = batch / "02_auto_labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            continue
        if not args.keep_old and output_dir.exists() and any(output_dir.iterdir()):
            backup = batch / f"02_auto_labels_previous_{stamp}"
            shutil.copytree(output_dir, backup)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = model.predict(
            source=[str(path) for path in images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.50,
            device=args.device,
            verbose=False,
            stream=True,
        )
        processed = 0
        boxes = 0
        for image, result in zip(images, results):
            boxes += write_labels(result, output_dir / f"{image.stem}.txt")
            processed += 1
        if processed != len(images):
            raise RuntimeError(f"{batch.name}: 预标注结果数量不完整")
        (output_dir / "labels.txt").write_text(
            "\n".join(str(names[index]) for index in sorted(names)) + "\n", encoding="utf-8"
        )
        total_images += processed
        total_boxes += boxes
        print({"batch": batch.name, "images": processed, "boxes": boxes, "output": str(output_dir)})
    print({"images": total_images, "boxes": total_boxes, "model": str(model_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
