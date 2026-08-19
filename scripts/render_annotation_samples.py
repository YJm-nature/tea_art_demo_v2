"""Render evenly spaced candidate-label samples for visual quality checks."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import parse_yolo_labels, read_manifest


COLORS = {
    0: (40, 220, 40),
    1: (220, 60, 220),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render cross-scene candidate label samples")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-source", type=int, default=12)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--batch-manifest", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    candidates = args.candidates.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_count = len((workspace / "classes.txt").read_text(encoding="utf-8-sig").splitlines())
    records = read_manifest(workspace)
    if args.batch_manifest:
        batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
        record_map = {record["sample_id"]: record for record in records}
        selected = [record_map[sample_id] for sample_id in batch["sample_ids"]]
    else:
        selected = []
        for source in sorted({record["source"] for record in records}):
            eligible = [
                record for record in records
                if record["source"] == source
                and record["review_status"] != "rejected"
                and record.get("requires_gaiwan_split")
            ]
            eligible.sort(key=lambda record: (record.get("frame_number") or -1, record["sample_id"]))
            for record in _even_sample(eligible, args.per_source):
                selected.append(record)

    pages = []
    for source in sorted({record["source"] for record in selected}):
        source_records = [record for record in selected if record["source"] == source]
        for page_index, start in enumerate(range(0, len(source_records), args.page_size), 1):
            page_records = source_records[start:start + args.page_size]
            tiles = [
                _render_tile(
                    workspace / record["image"],
                    candidates / f"{record['sample_id']}.txt",
                    class_count,
                    record["sample_id"],
                )
                for record in page_records
            ]
            while len(tiles) < args.page_size:
                tiles.append(np.full_like(tiles[0], 32))
            page = np.vstack(tiles)
            page_path = output / f"{source}_page_{page_index:02d}.jpg"
            cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
            pages.append(page_path.name)

    report = {
        "selected_images": len(selected),
        "sources": {
            source: sum(record["source"] == source for record in selected)
            for source in sorted({record["source"] for record in selected})
        },
        "pages": pages,
    }
    (output / "sample_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _even_sample(records: list[dict], limit: int) -> list[dict]:
    if len(records) <= limit:
        return records
    if limit <= 1:
        return [records[len(records) // 2]]
    indexes = {round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)}
    return [records[index] for index in sorted(indexes)]


def _render_tile(image_path: Path, label_path: Path, class_count: int, sample_id: str):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    for class_id, x, y, box_width, box_height in parse_yolo_labels(label_path, class_count):
        x1 = int((x - box_width / 2) * width)
        y1 = int((y - box_height / 2) * height)
        x2 = int((x + box_width / 2) * width)
        y2 = int((y + box_height / 2) * height)
        color = COLORS.get(class_id, (40 + class_id * 23 % 180, 180, 240 - class_id * 17 % 160))
        thickness = max(2, round(min(width, height) / 500))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            image,
            str(class_id),
            (x1, max(24, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, min(width, height) / 1000),
            color,
            thickness,
            cv2.LINE_AA,
        )
    scale = min(960 / width, 500 / height)
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((550, 960, 3), 24, dtype=np.uint8)
    x_offset = (960 - image.shape[1]) // 2
    canvas[42:42 + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    cv2.putText(canvas, sample_id, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
