"""Build MakeSense batches for adding kettle boxes to reviewed action frames.

Existing corrected utensil boxes are recovered from the prior MakeSense ZIP
exports by source video hash and frame number. Source images and review ZIPs are
never modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shutil
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_KEY_RE = re.compile(r"__(?P<video_hash>[0-9a-f]{10})__f(?P<frame>\d+)$")


def source_key(stem: str) -> str:
    match = SOURCE_KEY_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot recover source key from: {stem}")
    return f"{match.group('video_hash')}:{int(match.group('frame'))}"


def validate_label(text: str, source: str) -> str:
    normalized: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not 0 <= class_id < 18:
            raise ValueError(f"{source}:{line_number}: class ID outside 0..17")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{source}:{line_number}: coordinate outside 0..1")
        normalized.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
    return "\n".join(normalized) + ("\n" if normalized else "")


def corrected_labels(batch_root: Path) -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    for batch in sorted(batch_root.glob("batch_*")):
        zip_files = sorted((batch / "03_corrected_export").glob("*.zip"))
        if not zip_files:
            continue
        zip_path = zip_files[-1]
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".txt"):
                    continue
                name = Path(info.filename).name
                if name.lower() in {"labels.txt", "classes.txt"}:
                    continue
                key = source_key(Path(name).stem)
                text = validate_label(
                    archive.read(info).decode("utf-8-sig"),
                    f"{zip_path}:{info.filename}",
                )
                previous = labels.get(key)
                if previous and previous[0] != text:
                    raise ValueError(
                        f"Conflicting corrected labels for source frame {key}: "
                        f"{previous[1]} and {zip_path}"
                    )
                labels[key] = (text, f"{zip_path.name}:{name}")
    return labels


def action_name(image: Path, action_root: Path) -> str:
    return image.relative_to(action_root).parts[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare reviewed action frames for kettle box supplementation"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--output-name", default="kettle_box_supplement_v1",
        help="Directory name below annotation_batches",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=None,
        help="Only include these action directories, for example warm_clean water_injection",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    root = args.root.resolve()
    action_root = root / "derived" / "action_pool" / "new_front_full_202608"
    batch_root = root / "annotation_batches" / "detection_v2_100"
    output = root / "annotation_batches" / args.output_name
    classes_path = PROJECT / "dataset" / "tea_dataset_v1_reviewed" / "classes.txt"

    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"Output is not empty; use --force: {output}")
        shutil.rmtree(output)

    labels = corrected_labels(batch_root)
    selected_actions = set(args.actions or [])
    available_actions = {path.name for path in action_root.iterdir() if path.is_dir()}
    unknown_actions = sorted(selected_actions - available_actions)
    if unknown_actions:
        raise ValueError(f"Unknown action directories: {unknown_actions}")

    grouped: dict[str, list[Path]] = {}
    for image in sorted(action_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        action = action_name(image, action_root)
        if selected_actions and action not in selected_actions:
            continue
        grouped.setdefault(source_key(image.stem), []).append(image)
    if not grouped:
        raise ValueError("No action images matched the requested filters")

    missing = sorted(set(grouped) - set(labels))
    if missing:
        raise ValueError(f"Missing corrected labels for {len(missing)} source frames")

    records: list[dict[str, str]] = []
    for index, key in enumerate(sorted(grouped), 1):
        images = grouped[key]
        image = images[0]
        batch_number = (index - 1) // args.batch_size + 1
        batch = output / f"batch_{batch_number:03d}"
        image_dir = batch / "01_images"
        label_dir = batch / "02_yolo_import"
        corrected_dir = batch / "03_corrected_export"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        corrected_dir.mkdir(parents=True, exist_ok=True)

        target_image = image_dir / image.name
        target_label = label_dir / f"{image.stem}.txt"
        shutil.copy2(image, target_image)
        target_label.write_text(labels[key][0], encoding="utf-8")

        actions = sorted({action_name(item, action_root) for item in images})
        records.append({
            "batch": f"batch_{batch_number:03d}",
            "image_name": image.name,
            "label_name": target_label.name,
            "source_key": key,
            "actions": "|".join(actions),
            "duplicate_action_copies_removed": str(len(images) - 1),
            "label_source": labels[key][1],
            "required_review": "add kettle class 9; add kettle_display class 16 if readable; review all boxes",
        })

    batch_names = sorted({record["batch"] for record in records})
    classes_text = classes_path.read_text(encoding="utf-8-sig")
    for batch_name in batch_names:
        batch = output / batch_name
        (batch / "02_yolo_import" / "labels.txt").write_text(classes_text, encoding="utf-8")
        batch_records = [record for record in records if record["batch"] == batch_name]
        with (batch / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(batch_records[0]))
            writer.writeheader()
            writer.writerows(batch_records)

    with (output / "all_batches_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print({
        "action_images": sum(len(images) for images in grouped.values()),
        "unique_source_frames": len(grouped),
        "duplicate_action_copies_removed": sum(len(images) - 1 for images in grouped.values()),
        "actions": sorted(selected_actions) if selected_actions else sorted(available_actions),
        "batches": len(batch_names),
        "output": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
