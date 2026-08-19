"""Build a session-split, perceptually deduplicated front detection release.

Sources:
1. The current reviewed front release (1,238 images).
2. Accepted images from tea_dataset_v1_reviewed (844 images at the time this
   script was introduced).

Side-camera images are never added to this release. They are catalogued as
train-only candidates and must pass box review before a separate side model or
front-train augmentation release can use them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset"
CURRENT_RELEASE = (
    DATASET
    / "tea_sop_front_v1"
    / "releases"
    / "detection"
    / "front_detect_reviewed_v1"
)
REVIEWED_LEGACY = DATASET / "tea_dataset_v1_reviewed"
SIDE_DATASET = DATASET / "tea_sop_side_transition_v1"
DEFAULT_OUTPUT = (
    DATASET
    / "tea_sop_front_v1"
    / "releases"
    / "detection"
    / "front_detect_merged_dedup_v2"
)

SESSION_SPLITS = {
    "new_front_full_202608": "train",
    "legacy_reviewed_office": "train",
    "legacy_utensils_202606": "val",
    "legacy_root_25d00c9abd": "test",
    "legacy_root_a0ae6a2e54": "test",
    # focus and legacy_root_a0ae6a2e54 contain perceptually matching frames.
    "legacy_reviewed_focus": "test",
    "legacy_reviewed_original": "test",
}


@dataclass(frozen=True)
class Candidate:
    source_dataset: str
    session_id: str
    split: str
    image: Path
    label: Path
    source_reference: str
    priority: int


@dataclass
class KeptImage:
    candidate: Candidate
    sha1: str
    dhash: bytes
    output_name: str


def read_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            yield json.loads(line)


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> bytes:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    resized = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).reshape(-1)
    return np.packbits(bits).tobytes()


def hamming(left: bytes, right: bytes) -> int:
    return sum((a ^ b).bit_count() for a, b in zip(left, right))


def normalize_label(
    path: Path,
    class_count: int,
    active_class_ids: set[int],
) -> tuple[str, list[dict[str, str | int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: list[str] = []
    corrections: list[dict[str, str | int]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        if class_id not in active_class_ids:
            corrections.append(
                {
                    "source_label": project_relative(path),
                    "line_number": line_number,
                    "class_id": class_id,
                    "action": "removed_deferred_class",
                }
            )
            continue
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            corrections.append(
                {
                    "source_label": project_relative(path),
                    "line_number": line_number,
                    "class_id": class_id,
                    "action": "removed_zero_area_box",
                }
            )
            continue
        output.append(
            f"{class_id} " + " ".join(f"{value:.6f}" for value in coordinates)
        )
    return "\n".join(output) + ("\n" if output else ""), corrections


def current_candidates() -> list[Candidate]:
    rows: list[Candidate] = []
    with (CURRENT_RELEASE / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for record in csv.DictReader(handle):
            session = record["session"]
            split = SESSION_SPLITS.get(session)
            if split is None:
                raise ValueError(f"unassigned current session: {session}")
            image = CURRENT_RELEASE / record["split"] / "images" / record["image"]
            label = (
                CURRENT_RELEASE
                / record["split"]
                / "labels"
                / f"{Path(record['image']).stem}.txt"
            )
            rows.append(
                Candidate(
                    source_dataset="front_detect_reviewed_v1",
                    session_id=session,
                    split=split,
                    image=image,
                    label=label,
                    source_reference=record.get("source", ""),
                    priority=0,
                )
            )
    return rows


def legacy_candidates() -> list[Candidate]:
    rows: list[Candidate] = []
    for record in read_jsonl(REVIEWED_LEGACY / "manifest.jsonl"):
        if record.get("review_status") != "accepted":
            continue
        session = f"legacy_reviewed_{record.get('session_id', 'unknown')}"
        split = SESSION_SPLITS.get(session)
        if split is None:
            raise ValueError(f"unassigned accepted legacy session: {session}")
        rows.append(
            Candidate(
                source_dataset="tea_dataset_v1_reviewed",
                session_id=session,
                split=split,
                image=REVIEWED_LEGACY / record["image"],
                label=REVIEWED_LEGACY / record["detect_label"],
                source_reference=str(record.get("source_image", "")),
                priority=1,
            )
        )
    return rows


def select_images(
    candidates: list[Candidate], threshold: int
) -> tuple[list[KeptImage], list[dict[str, str | int]]]:
    kept: list[KeptImage] = []
    exact: dict[str, KeptImage] = {}
    excluded: list[dict[str, str | int]] = []

    # Preserve the already reviewed current release when a legacy candidate is
    # identical or nearly identical. Within each priority, keep deterministic order.
    candidates.sort(
        key=lambda item: (
            item.priority,
            item.split,
            item.session_id,
            item.image.name,
        )
    )
    for candidate in candidates:
        if not candidate.image.is_file():
            raise FileNotFoundError(candidate.image)
        digest = file_sha1(candidate.image)
        if digest in exact:
            match = exact[digest]
            excluded.append(exclusion_row(candidate, match, 0, "exact_duplicate"))
            continue

        perceptual = difference_hash(candidate.image)
        nearest: KeptImage | None = None
        nearest_distance = 10_000
        for existing in kept:
            distance = hamming(perceptual, existing.dhash)
            if distance < nearest_distance:
                nearest = existing
                nearest_distance = distance
                if distance == 0:
                    break
        if nearest is not None and nearest_distance <= threshold:
            excluded.append(
                exclusion_row(
                    candidate,
                    nearest,
                    nearest_distance,
                    "perceptual_near_duplicate",
                )
            )
            continue

        safe_session = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in candidate.session_id
        )
        output_name = f"{safe_session}__{digest[:12]}{candidate.image.suffix.lower()}"
        item = KeptImage(candidate, digest, perceptual, output_name)
        kept.append(item)
        exact[digest] = item
    return kept, excluded


def exclusion_row(
    candidate: Candidate,
    match: KeptImage,
    distance: int,
    reason: str,
) -> dict[str, str | int]:
    return {
        "reason": reason,
        "dhash_distance": distance,
        "excluded_split": candidate.split,
        "excluded_session": candidate.session_id,
        "excluded_source_dataset": candidate.source_dataset,
        "excluded_image": project_relative(candidate.image),
        "kept_split": match.candidate.split,
        "kept_session": match.candidate.session_id,
        "kept_source_dataset": match.candidate.source_dataset,
        "kept_image": project_relative(match.candidate.image),
    }


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_side_candidates(output: Path) -> dict[str, int]:
    rows: list[dict[str, str]] = []
    status_counts: dict[str, int] = {}
    metadata_path = SIDE_DATASET / "manifests" / "frames.jsonl"
    if not metadata_path.is_file():
        write_csv(output / "side_train_candidates.csv", rows)
        return status_counts
    for record in read_jsonl(metadata_path):
        status = str(record.get("review_status", "pending"))
        status_counts[status] = status_counts.get(status, 0) + 1
        image = SIDE_DATASET / record["image"]
        label = SIDE_DATASET / record["detect_label"]
        eligible = status == "accepted" and image.is_file() and label.is_file()
        rows.append(
            {
                "session_id": str(record.get("session_id", "")),
                "source_group": str(record.get("source_group", "")),
                "review_status": status,
                "eligible_for_train": str(eligible).lower(),
                "image": project_relative(image),
                "label": project_relative(label),
                "policy": "train_only_never_front_val_test",
                "reason": "" if eligible else "box_review_not_accepted",
            }
        )
    write_csv(output / "side_train_candidates.csv", rows)
    return status_counts


def verify_no_leakage(kept: list[KeptImage], threshold: int) -> None:
    session_splits: dict[str, set[str]] = {}
    for item in kept:
        session_splits.setdefault(item.candidate.session_id, set()).add(
            item.candidate.split
        )
    leaking_sessions = {
        session: sorted(splits)
        for session, splits in session_splits.items()
        if len(splits) > 1
    }
    if leaking_sessions:
        raise ValueError(f"session leakage: {leaking_sessions}")

    for index, left in enumerate(kept):
        for right in kept[index + 1 :]:
            if left.candidate.split == right.candidate.split:
                continue
            distance = hamming(left.dhash, right.dhash)
            if distance <= threshold:
                raise ValueError(
                    "cross-split perceptual duplicate remained: "
                    f"{left.candidate.image} vs {right.candidate.image}, distance={distance}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the merged leak-free front YOLO18 detection release"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=4,
        help="maximum 256-bit dHash distance treated as a near duplicate",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    classes = (
        REVIEWED_LEGACY / "classes.txt"
    ).read_text(encoding="utf-8-sig").splitlines()
    if len(classes) != 18:
        raise ValueError(f"expected fixed 18 classes, found {len(classes)}")
    ontology = yaml.safe_load(
        (REVIEWED_LEGACY / "ontology_v1.yaml").read_text(encoding="utf-8")
    )
    active_class_ids = {
        int(value)
        for value in ontology["training_phase"]["active_detect_class_ids"]
    }

    candidates = current_candidates() + legacy_candidates()
    kept, excluded = select_images(candidates, args.dhash_threshold)
    verify_no_leakage(kept, args.dhash_threshold)

    manifest_rows: list[dict[str, str | int]] = []
    validator_rows: list[dict[str, str]] = []
    label_corrections: list[dict[str, str | int]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    source_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    for item in kept:
        candidate = item.candidate
        image_target = output / candidate.split / "images" / item.output_name
        label_target = (
            output
            / candidate.split
            / "labels"
            / f"{Path(item.output_name).stem}.txt"
        )
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.image, image_target)
        normalized_label, corrections = normalize_label(
            candidate.label, len(classes), active_class_ids
        )
        label_target.write_text(normalized_label, encoding="utf-8")
        for correction in corrections:
            label_corrections.append(
                {
                    **correction,
                    "output_label": label_target.relative_to(output).as_posix(),
                }
            )
        split_counts[candidate.split] += 1
        source_counts[candidate.source_dataset] = (
            source_counts.get(candidate.source_dataset, 0) + 1
        )
        session_counts[candidate.session_id] = (
            session_counts.get(candidate.session_id, 0) + 1
        )
        manifest_rows.append(
            {
                "split": candidate.split,
                "session_id": candidate.session_id,
                "source_dataset": candidate.source_dataset,
                "output_image": item.output_name,
                "source_image": project_relative(candidate.image),
                "source_label": project_relative(candidate.label),
                "source_reference": candidate.source_reference,
                "sha1": item.sha1,
                "dhash_hex": item.dhash.hex(),
            }
        )
        validator_rows.append(
            {
                "sample_id": Path(item.output_name).stem,
                "split": candidate.split,
                "session_id": candidate.session_id,
                "image": image_target.relative_to(output).as_posix(),
                "label": label_target.relative_to(output).as_posix(),
                "sha256": file_sha256(image_target),
                "source_dataset": candidate.source_dataset,
                "source_image": project_relative(candidate.image),
            }
        )

    data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
        "session_split": SESSION_SPLITS,
        "deduplication": {
            "algorithm": "256_bit_dhash",
            "maximum_duplicate_distance": args.dhash_threshold,
        },
        "side_data_policy": "excluded_from_release; train-only after accepted review",
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )
    shutil.copy2(REVIEWED_LEGACY / "ontology_v1.yaml", output / "ontology_v1.yaml")
    write_csv(output / "manifest.csv", manifest_rows)
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in validator_rows) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "excluded_duplicates.csv", excluded)
    write_csv(output / "label_corrections.csv", label_corrections)
    write_csv(
        output / "session_assignments.csv",
        [
            {"session_id": session, "split": split, "images": session_counts.get(session, 0)}
            for session, split in SESSION_SPLITS.items()
        ],
    )
    side_statuses = write_side_candidates(output)

    summary = {
        "schema_version": "1.0",
        "candidate_images": len(candidates),
        "kept_images": len(kept),
        "excluded_duplicates": len(excluded),
        "label_corrections": len(label_corrections),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "session_counts": session_counts,
        "side_candidate_statuses": side_statuses,
        "side_images_in_release": 0,
        "reviewed_front_flow_policy": (
            "Only images already present in front_detect_reviewed_v1 are included; "
            "action labels alone never authorize detector training."
        ),
    }
    (output / "release_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Front detection merged release v2\n\n"
        "This release combines the reviewed front release with accepted legacy "
        "YOLO18 boxes. It is perceptually deduplicated and split only by complete "
        "recording session. Side-camera samples are not included. See "
        "`excluded_duplicates.csv`, `session_assignments.csv`, and "
        "`side_train_candidates.csv` for the audit trail.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"release: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
