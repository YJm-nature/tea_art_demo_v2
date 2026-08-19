"""Build a pending detection release holding IMG_4901 out of training.

Reviewed images from the current merged release become train. All 128 frames
extracted from IMG_4901 are split chronologically into pending val/test images.
No data.yaml is emitted until those holdout labels have been manually reviewed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FRONT = PROJECT / "dataset" / "tea_sop_front_v1"
MODULAR = PROJECT / "dataset" / "tea_sop_modular_v1"
SOURCE = FRONT / "releases" / "detection" / "front_detect_merged_kettle_v3"
OUTPUT = FRONT / "releases" / "detection" / "front_detect_img4901_holdout_v1"
HOLDOUT_GLOB = "612b5db104*.jpg"
FRAME_PATTERN = re.compile(r"__f(\d+)_")


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_score(label: Path) -> tuple[int, int]:
    lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    return int(any(line.split()[0] == "9" for line in lines)), len(lines)


def frame_index(path: Path) -> int:
    match = FRAME_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"cannot parse frame index: {path.name}")
    return int(match.group(1))


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Collapse the 126 exact duplicates introduced by the kettle supplement,
    # retaining the label that contains the reviewed kettle box.
    selected: dict[str, tuple[Path, Path, tuple[int, int]]] = {}
    duplicate_inputs = 0
    for image in sorted(SOURCE.glob("*/images/*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = SOURCE / image.parent.parent.name / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        digest = sha1(image)
        candidate = (image, label, label_score(label))
        previous = selected.get(digest)
        if previous is not None:
            duplicate_inputs += 1
        if previous is None or candidate[2] > previous[2]:
            selected[digest] = candidate

    train_images = OUTPUT / "train" / "images"
    train_labels = OUTPUT / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    train_rows: list[dict[str, str]] = []
    for digest, (image, label, _) in sorted(selected.items(), key=lambda item: item[1][0].name):
        shutil.copy2(image, train_images / image.name)
        shutil.copy2(label, train_labels / label.name)
        train_rows.append(
            {
                "split": "train",
                "image_name": image.name,
                "source": str(image.relative_to(PROJECT)).replace("\\", "/"),
                "sha1": digest,
                "review_status": "accepted",
            }
        )

    holdout_source = MODULAR / "derived" / "detection_pool" / "images"
    holdout = sorted(holdout_source.glob(HOLDOUT_GLOB), key=frame_index)
    if len(holdout) != 128:
        raise ValueError(f"expected 128 IMG_4901 frames, found {len(holdout)}")
    holdout_rows: list[dict[str, str | int]] = []
    split_point = len(holdout) // 2
    for index, image in enumerate(holdout):
        split = "val" if index < split_point else "test"
        target = OUTPUT / "pending_holdout" / split / "images" / image.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        holdout_rows.append(
            {
                "split": split,
                "image_name": image.name,
                "frame_index": frame_index(image),
                "timestamp_order": index,
                "source_video": "tea_sop_modular_v1/raw_videos/00_inbox/已加速- IMG_4901.MOV",
                "source": str(image.relative_to(PROJECT)).replace("\\", "/"),
                "sha1": sha1(image),
                "review_status": "pending_box_review",
            }
        )

    shutil.copy2(SOURCE / "classes.txt", OUTPUT / "classes.txt")
    with (OUTPUT / "train_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_rows[0]))
        writer.writeheader()
        writer.writerows(train_rows)
    with (OUTPUT / "holdout_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(holdout_rows[0]))
        writer.writeheader()
        writer.writerows(holdout_rows)

    summary = {
        "schema_version": "1.0",
        "ready_for_training": False,
        "reason": "IMG_4901 val/test frames still require complete manual box review",
        "train_images": len(train_rows),
        "pending_val_images": split_point,
        "pending_test_images": len(holdout) - split_point,
        "exact_duplicate_train_inputs_collapsed": duplicate_inputs,
        "holdout_video": "tea_sop_modular_v1/raw_videos/00_inbox/已加速- IMG_4901.MOV",
        "holdout_video_id": "612b5db104",
        "split_policy": "chronological first half val, second half test; no IMG_4901 image is train",
    }
    (OUTPUT / "release_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# IMG_4901 scene holdout release (pending)\n\n"
        "`train` contains reviewed images from all other available scenes. The 128\n"
        "frames from `IMG_4901.MOV` are isolated under `pending_holdout`: the first\n"
        "64 chronological frames are val and the final 64 are test. They currently\n"
        "have no accepted labels, so this directory intentionally has no data.yaml.\n"
        "After full box review, publish labels into val/test and generate data.yaml.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
