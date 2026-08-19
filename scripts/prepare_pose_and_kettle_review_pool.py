"""Prepare unlabelled vessel-pose and kettle-box review images.

The output is a review pool only. It is not a training release until the
annotator exports and checks the pose/keypoint and optional kettle boxes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "tea_sop_front_v1"
ACTION_ROOT = DATASET / "derived" / "action_pool" / "new_front_full_202608"
POSE_RELEASE = DATASET / "releases" / "pose" / "front_vessel_pose_prototype_v1"
OUTPUT = DATASET / "annotation_batches" / "pose_and_kettle_review_v2"


def digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def find_existing_label(name: str) -> Path | None:
    candidates = list(
        (DATASET / "releases" / "detection" / "front_detect_reviewed_v1").glob(
            f"*/labels/{Path(name).stem}.txt"
        )
    )
    if candidates:
        return candidates[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {output}")
    images_out = output / "01_images_for_cvat"
    labels_out = output / "02_existing_detection_labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    existing_pose = {
        path.name
        for split in ("train", "val")
        for path in (POSE_RELEASE / split / "images").glob("*")
    }
    seen_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    skipped_existing = 0
    skipped_blur = 0
    skipped_duplicate = 0

    for action_dir in sorted(ACTION_ROOT.iterdir()):
        if not action_dir.is_dir() or action_dir.name in {"warm_clean", "water_injection", "cup_layout"}:
            continue
        for variant_dir in sorted(action_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for image in sorted(variant_dir.glob("*.jpg")):
                if image.name in existing_pose:
                    skipped_existing += 1
                    continue
                image_hash = digest(image)
                if image_hash in seen_hashes:
                    skipped_duplicate += 1
                    continue
                score = sharpness(image)
                if score < args.min_sharpness:
                    skipped_blur += 1
                    continue
                seen_hashes.add(image_hash)
                target = images_out / image.name
                shutil.copy2(image, target)
                label = find_existing_label(image.name)
                has_label = "yes" if label else "no"
                if label:
                    shutil.copy2(label, labels_out / label.name)
                target_suggestions = {
                    "tea_lotus_to_gaiwan": "tea_lotus,gaiwan_body",
                    "shake_aroma": "tea_lotus,gaiwan_body",
                    "open_lid_smell": "gaiwan_body",
                    "gaiwan_to_pitcher": "gaiwan_body,pitcher",
                    "tea_distribution": "pitcher",
                }.get(action_dir.name, "all_visible_pose_classes")
                rows.append(
                    {
                        "image_name": image.name,
                        "source_path": str(image.relative_to(PROJECT)).replace("\\", "/"),
                        "action": action_dir.name,
                        "variant": variant_dir.name,
                        "suggested_pose_classes": target_suggestions,
                        "sharpness_laplacian": f"{score:.2f}",
                        "existing_detection_label": has_label,
                        "review_status": "pending",
                    }
                )

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(rows[0]) if rows else ["image_name", "source_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images_for_pose_review": len(rows),
        "excluded_already_pose_labelled": skipped_existing,
        "excluded_blurry": skipped_blur,
        "excluded_exact_duplicates": skipped_duplicate,
        "source": str(ACTION_ROOT),
        "pose_classes": ["kettle", "pitcher", "gaiwan_body", "tea_lotus"],
        "instruction": "Annotate every visible pose class; add kettle detection boxes to the copied labels when visible.",
    }
    (output / "summary.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Vessel pose and kettle-box review pool\n\n"
        "These images come from the new front workflow videos and exclude the 126\n"
        "images already used by the first pose prototype. In CVAT, annotate every\n"
        "visible kettle, pitcher, gaiwan body and tea lotus skeleton. For images\n"
        "with a visible kettle, also add or correct the kettle detection box in\n"
        "the copied YOLO label. Do not publish this pool until all annotations\n"
        "have been reviewed.\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
