"""Create review-only YOLO prelabels for the IMG_4901 holdout frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = PROJECT / "dataset" / "tea_sop_front_v1" / "releases" / "detection" / "front_detect_img4901_holdout_v1"
DEFAULT_MODEL = PROJECT / "models" / "low_vram" / "merged_kettle_stage1_640" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    release = args.release.resolve()
    model = YOLO(str(args.model.resolve()))
    counts: dict[str, int] = {}
    for split in ("val", "test"):
        image_dir = release / "pending_holdout" / split / "images"
        label_dir = release / "pending_holdout" / split / "prelabels"
        label_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        results = model.predict(
            source=[str(path) for path in sorted(image_dir.glob("*.jpg"))],
            imgsz=args.imgsz,
            conf=args.conf,
            device=0,
            batch=4,
            workers=0,
            verbose=False,
        )
        for result in results:
            output = label_dir / f"{Path(result.path).stem}.txt"
            rows: list[str] = []
            if result.boxes is not None:
                for cls, box, confidence in zip(
                    result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), result.boxes.conf.tolist()
                ):
                    rows.append(
                        f"{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}"
                    )
            output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            total += len(rows)
        counts[split] = total
    summary = {
        "model": str(args.model.resolve()),
        "confidence": args.conf,
        "imgsz": args.imgsz,
        "purpose": "review_prelabels_only",
        "detections_by_split": counts,
        "warning": "These labels must be manually corrected and are not accepted validation labels.",
    }
    (release / "prelabel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
