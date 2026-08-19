"""Train from reviewed seeds and generate conservative pseudo labels."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_annotation import (
    Prediction,
    add_missing_legacy_split_boxes,
    apply_candidate_labels,
    learn_legacy_split_transforms,
    merge_baseline_with_predictions,
    prepare_seed_dataset,
    read_seed_archive,
)
from src.dataset_rebuild import parse_yolo_labels, read_manifest, write_yolo_labels


PROJECT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = PROJECT / "output" / "ultralytics_config"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="从人工种子生成待审核伪标签")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("output", type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("seed_dataset", type=Path)
    train.add_argument("--model", type=Path, default=PROJECT / "yolo26n.pt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--name", default="focus50_current10")

    predict = subparsers.add_parser("predict")
    predict.add_argument("workspace", type=Path)
    predict.add_argument("archive", type=Path)
    predict.add_argument("model", type=Path)
    predict.add_argument("output", type=Path)
    predict.add_argument("--imgsz", type=int, default=640)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("workspace", type=Path)
    apply_parser.add_argument("candidates", type=Path)
    apply_parser.add_argument("--run-name", default="auto_v1")

    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_seed_dataset(args.workspace, args.archive, args.output)
    elif args.command == "train":
        report = train_seed_model(args)
    elif args.command == "predict":
        report = predict_candidates(args)
    else:
        report = apply_candidate_labels(args.workspace, args.candidates, args.run_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def train_seed_model(args) -> dict:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝用CPU长时间训练")
    seed_dataset = args.seed_dataset.resolve()
    project = seed_dataset / "models"
    model = YOLO(str(args.model.resolve()))
    started = time.time()
    result = model.train(
        data=str(seed_dataset / "data.yaml"),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=0,
        workers=0,
        cache=False,
        amp=True,
        optimizer="AdamW",
        lr0=0.001,
        close_mosaic=10,
        cos_lr=True,
        pretrained=True,
        seed=42,
        project=str(project),
        name=args.name,
        exist_ok=False,
        save=True,
        save_period=10,
        mixup=0.0,
    )
    best = project / args.name / "weights" / "best.pt"
    return {
        "best_model": str(best),
        "exists": best.exists(),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "note": "仅用于伪标注，验证集与训练集来自同一session，指标不作为验收依据",
        "trainer_result": str(result),
    }


def predict_candidates(args) -> dict:
    from ultralytics import YOLO

    workspace = args.workspace.resolve()
    output = args.output.resolve()
    candidate_dir = output / "candidate_labels"
    if candidate_dir.exists() and any(candidate_dir.iterdir()):
        raise FileExistsError(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    class_names = workspace.joinpath("classes.txt").read_text(encoding="utf-8-sig").splitlines()
    seeds = read_seed_archive(args.archive.resolve(), workspace, len(class_names))
    active_classes = {row[0] for rows in seeds.values() for row in rows}
    split_transforms = learn_legacy_split_transforms(workspace, seeds)
    records = read_manifest(workspace)
    remaining = [record for record in records if record["sample_id"] not in seeds]
    model = YOLO(str(args.model.resolve()))

    reports = []
    totals = Counter()
    processed = 0
    for record in remaining:
        baseline = parse_yolo_labels(workspace / record["detect_label"], len(class_names))
        if record["review_status"] == "rejected":
            write_yolo_labels(candidate_dir / f"{record['sample_id']}.txt", baseline)
            reports.append({
                "sample_id": record["sample_id"],
                "source": record["source"],
                "seed": False,
                "rejected": True,
                "baseline_boxes": len(baseline),
                "prediction_boxes": 0,
                "output_boxes": len(baseline),
                "added": 0,
                "refined": 0,
                "retained": len(baseline),
                "derived_split_boxes": 0,
                "mean_accepted_confidence": None,
            })
            processed += 1
            totals["skipped_rejected"] += 1
            continue
        image_path = str((workspace / record["image"]).resolve())
        result = model.predict(
            source=image_path,
            imgsz=args.imgsz,
            conf=0.20,
            iou=0.50,
            device=0,
            verbose=False,
        )[0]
        processed += 1
        predictions = []
        if result.boxes is not None:
            for values in result.boxes.data.detach().cpu().tolist():
                x1, y1, x2, y2, confidence, class_id = values
                height, width = result.orig_shape
                predictions.append(Prediction(
                    class_id=int(class_id),
                    x=((x1 + x2) / 2) / width,
                    y=((y1 + y2) / 2) / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height,
                    confidence=float(confidence),
                ))
        merged, report = merge_baseline_with_predictions(
            baseline, predictions, active_classes=active_classes
        )
        derived_split_boxes = 0
        if record.get("requires_gaiwan_split"):
            legacy = parse_yolo_labels(workspace / record["legacy_label"], 9)
            merged, derived_split_boxes = add_missing_legacy_split_boxes(
                merged, legacy, split_transforms
            )
        write_yolo_labels(candidate_dir / f"{record['sample_id']}.txt", merged)
        report["output_boxes"] = len(merged)
        report.update(
            sample_id=record["sample_id"],
            source=record["source"],
            seed=False,
            rejected=False,
            derived_split_boxes=derived_split_boxes,
        )
        reports.append(report)
        totals.update({key: report[key] for key in ("added", "refined", "retained")})
        totals["derived_split_boxes"] += derived_split_boxes

    if processed != len(remaining):
        raise RuntimeError(f"预测结果不完整: expected={len(remaining)} actual={processed}")

    for stem, rows in seeds.items():
        write_yolo_labels(candidate_dir / f"{stem}.txt", rows)
        reports.append({
            "sample_id": stem,
            "source": "focus",
            "seed": True,
            "baseline_boxes": None,
            "prediction_boxes": None,
            "output_boxes": len(rows),
            "added": 0,
            "refined": 0,
            "retained": 0,
            "derived_split_boxes": 0,
            "mean_accepted_confidence": None,
        })

    reports.sort(key=lambda row: row["sample_id"])
    (output / "prediction_report.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reports), encoding="utf-8"
    )
    summary = {
        "images": len(reports),
        "seed_images": len(seeds),
        "pseudo_images": len(remaining),
        "active_class_ids": sorted(active_classes),
        "inactive_class_ids": [index for index in range(len(class_names)) if index not in active_classes],
        "added_boxes": totals["added"],
        "refined_boxes": totals["refined"],
        "retained_boxes": totals["retained"],
        "derived_split_boxes": totals["derived_split_boxes"],
        "skipped_rejected_images": totals["skipped_rejected"],
        "legacy_split_transforms": {
            str(class_id): [round(value, 4) for value in transform]
            for class_id, transform in sorted(split_transforms.items())
        },
        "candidate_labels": str(candidate_dir),
    }
    (output / "prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
