"""Move user-selected train images into a labelled val/test holdout.

The selected images are expected in ``test&val`` under the IMG_4901 release.
Their labels are still in ``train/labels``.  The operation is deterministic,
keeps the original source metadata, and leaves the separate pending IMG_4901
holdout untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = (
    PROJECT
    / "dataset"
    / "tea_sop_front_v1"
    / "releases"
    / "detection"
    / "front_detect_img4901_holdout_v1"
)


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_number(source_reference: str, image_name: str) -> int:
    match = re.search(r"frame[_-](\d+)", source_reference or "", re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:f|frame)[_-]?(\d+)", image_name, re.I)
    return int(match.group(1)) if match else 10**9


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--val-ratio", type=float, default=0.5)
    args = parser.parse_args()

    release = args.release.resolve()
    selected_dir = release / "test&val"
    train_images = release / "train" / "images"
    train_labels = release / "train" / "labels"
    if not selected_dir.exists():
        raise FileNotFoundError(f"selected image directory not found: {selected_dir}")

    selected = sorted(
        [path for path in selected_dir.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )
    if not selected:
        raise ValueError(f"no selected images in {selected_dir}")

    manifest_path = release / "train_manifest.csv"
    manifest_rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    by_name = {row["image_name"]: row for row in manifest_rows}
    missing_manifest = [path.name for path in selected if path.name not in by_name]
    if missing_manifest:
        raise ValueError(f"selected images missing from train_manifest.csv: {missing_manifest[:5]}")

    missing_labels = [
        path.name
        for path in selected
        if not (train_labels / f"{path.stem}.txt").is_file()
    ]
    if missing_labels:
        raise ValueError(f"selected images missing labels: {missing_labels[:5]}")

    # Keep chronological ordering where the old manifest retains original frame metadata.
    selected.sort(
        key=lambda path: (
            frame_number(by_name[path.name].get("source_reference", ""), path.name),
            path.name,
        )
    )
    val_count = max(1, min(len(selected) - 1, round(len(selected) * args.val_ratio)))
    split_paths = {"val": selected[:val_count], "test": selected[val_count:]}

    for split in split_paths:
        (release / split / "images").mkdir(parents=True, exist_ok=True)
        (release / split / "labels").mkdir(parents=True, exist_ok=True)

    output_rows: dict[str, list[dict[str, str]]] = {"val": [], "test": []}
    selected_names = {path.name for path in selected}
    for split, paths in split_paths.items():
        for image in paths:
            label = train_labels / f"{image.stem}.txt"
            target_image = release / split / "images" / image.name
            target_label = release / split / "labels" / label.name
            if target_image.exists() or target_label.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {target_image}")
            shutil.move(str(image), str(target_image))
            shutil.move(str(label), str(target_label))

            row = dict(by_name[image.name])
            row["split"] = split
            row["source"] = f"{split}/images/{image.name}"
            row["label"] = f"{split}/labels/{label.name}"
            row["sha1"] = sha1(target_image)
            output_rows[split].append(row)

    remaining_train = [row for row in manifest_rows if row["image_name"] not in selected_names]
    train_manifest_backup = release / "train_manifest.before_selected_holdout.csv"
    if not train_manifest_backup.exists():
        shutil.copy2(manifest_path, train_manifest_backup)

    fields = list(manifest_rows[0]) if manifest_rows else ["split", "image_name"]
    if "label" not in fields:
        fields.append("label")
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in remaining_train:
            row = dict(row)
            row["split"] = "train"
            writer.writerow({field: row.get(field, "") for field in fields})

    for split, rows in output_rows.items():
        with (release / f"{split}_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    classes = [line.strip() for line in (release / "classes.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    data_lines = [
        f"path: {release.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(classes)}",
        "names:",
    ]
    data_lines.extend(f"  {index}: {yaml_quote(name)}" for index, name in enumerate(classes))
    (release / "data.yaml").write_text("\n".join(data_lines) + "\n", encoding="utf-8")

    summary_path = release / "release_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "ready_for_training": True,
            "reason": "selected 150-image holdout has reviewed matching labels; IMG_4901 pending set remains separate",
            "train_images": len(remaining_train),
            "val_images": len(output_rows["val"]),
            "test_images": len(output_rows["test"]),
            "selected_holdout": {
                "source_session": "legacy_reviewed_original",
                "selected_images": len(selected),
                "val": len(output_rows["val"]),
                "test": len(output_rows["test"]),
                "policy": "user_selected_images_sorted_by_original_frame_number_then_split",
                "directory": "test&val",
            },
            "pending_img4901_holdout_preserved": True,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (release / "SELECTED_HOLDOUT_README.md").write_text(
        "# Selected detection holdout\n\n"
        f"The user-selected {len(selected)} images formerly under `test&val` were moved into\n"
        f"`val` ({len(output_rows['val'])}) and `test` ({len(output_rows['test'])}). Their YOLO\n"
        "labels were moved from the old train label directory with matching basenames.\n"
        "The separate `pending_holdout` IMG_4901 candidate set was preserved and is not\n"
        "included by `data.yaml`.\n\n"
        "This holdout is a selected subset of the `legacy_reviewed_original` session; it\n"
        "is useful for checking the current model but is not an independent new recording.\n",
        encoding="utf-8",
    )
    print(json.dumps({"release": str(release), "train": len(remaining_train), "val": len(output_rows["val"]), "test": len(output_rows["test"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
