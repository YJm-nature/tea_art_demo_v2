"""Initialize and validate the action-observation dataset workspace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / "dataset" / "action_observations_v1"
VALID_SPLITS = {"train", "val", "test"}
VALID_CAMERAS = {"tabletop", "side"}
VALID_SAMPLE_KINDS = {"positive", "error", "hard_negative"}
VALID_REVIEW_STATUSES = {"pending", "needs_fix", "accepted", "rejected"}
VALID_HAND_SIDES = {"left", "right"}
JEWELRY_CLASSES = ("ring", "bracelet_or_bangle", "watch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SEGMENT_REQUIRED_FIELDS = {
    "schema_version",
    "segment_id",
    "session_id",
    "split",
    "camera_role",
    "video_path",
    "observation_point_id",
    "start_ms",
    "end_ms",
    "sample_kind",
    "target_utensil",
    "review_status",
    "reviewer",
    "reviewed_at",
    "review_note",
}

JEWELRY_REQUIRED_FIELDS = {
    "schema_version",
    "roi_id",
    "session_id",
    "split",
    "image",
    "label",
    "source_video",
    "frame_ms",
    "hand_side",
    "sample_kind",
    "review_status",
    "reviewer",
    "reviewed_at",
    "review_note",
}


def initialize_dataset(target: Path = DEFAULT_DATASET) -> Path:
    """Create a reusable workspace without overwriting existing files."""
    target = target.resolve()
    if target != DEFAULT_DATASET.resolve():
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {target}")
        shutil.copytree(DEFAULT_DATASET, target, dirs_exist_ok=True)

    directories = [
        "annotations",
        "schemas",
        "templates",
        "splits",
        "raw_sessions",
        "jewelry_roi/train/images",
        "jewelry_roi/train/labels",
        "jewelry_roi/val/images",
        "jewelry_roi/val/labels",
        "jewelry_roi/test/images",
        "jewelry_roi/test/labels",
    ]
    for relative in directories:
        (target / relative).mkdir(parents=True, exist_ok=True)
    return target


def validate_dataset(root: Path, require_media: bool = False) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        "dataset_info.yaml",
        "annotations/segments.jsonl",
        "splits/session_assignments.csv",
        "jewelry_roi/classes.txt",
        "jewelry_roi/data.yaml",
        "jewelry_roi/manifest.jsonl",
        "jewelry_roi/train_6gb.yaml",
    ]
    for relative in required_files:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    assignments = _read_assignments(root / "splits/session_assignments.csv", errors)
    segment_records = _read_jsonl(root / "annotations/segments.jsonl", errors)
    jewelry_records = _read_jsonl(root / "jewelry_roi/manifest.jsonl", errors)

    session_splits: dict[str, set[str]] = defaultdict(set)
    for session_id, split in assignments.items():
        session_splits[session_id].add(split)

    segment_ids: set[str] = set()
    segment_kinds: Counter[str] = Counter()
    observation_points: Counter[str] = Counter()
    for line_number, record in segment_records:
        prefix = f"annotations/segments.jsonl:{line_number}"
        _validate_segment(
            root, record, prefix, assignments, session_splits, segment_ids,
            errors, warnings, require_media,
        )
        if isinstance(record.get("sample_kind"), str):
            segment_kinds[record["sample_kind"]] += 1
        if isinstance(record.get("observation_point_id"), str):
            observation_points[record["observation_point_id"]] += 1

    jewelry_ids: set[str] = set()
    jewelry_kinds: Counter[str] = Counter()
    class_instances: Counter[int] = Counter()
    for line_number, record in jewelry_records:
        prefix = f"jewelry_roi/manifest.jsonl:{line_number}"
        _validate_jewelry_record(
            root, record, prefix, assignments, session_splits, jewelry_ids,
            class_instances, errors, warnings, require_media,
        )
        if isinstance(record.get("sample_kind"), str):
            jewelry_kinds[record["sample_kind"]] += 1

    _validate_class_file(root / "jewelry_roi/classes.txt", errors)
    leaking_sessions = {
        session: sorted(splits)
        for session, splits in sorted(session_splits.items())
        if len(splits) > 1
    }
    if leaking_sessions:
        errors.append(f"session split leakage: {leaking_sessions}")

    referenced_sessions = {
        record.get("session_id")
        for _, record in segment_records + jewelry_records
        if isinstance(record.get("session_id"), str)
    }
    unused_assignments = sorted(set(assignments) - referenced_sessions)
    if unused_assignments:
        warnings.append(f"session assignments without records: {unused_assignments}")

    return {
        "schema_version": "1.0",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "segments": len(segment_records),
            "jewelry_rois": len(jewelry_records),
            "sessions": len(referenced_sessions),
            "segment_sample_kinds": dict(sorted(segment_kinds.items())),
            "jewelry_sample_kinds": dict(sorted(jewelry_kinds.items())),
            "observation_points": dict(sorted(observation_points.items())),
            "jewelry_class_instances": {
                JEWELRY_CLASSES[index]: class_instances[index]
                for index in range(len(JEWELRY_CLASSES))
            },
        },
        "session_splits": {
            session: sorted(splits) for session, splits in sorted(session_splits.items())
        },
    }


def _read_jsonl(path: Path, errors: list[str]) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return []
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}:{line_number}: record must be an object")
            continue
        records.append((line_number, value))
    return records


def _read_assignments(path: Path, errors: list[str]) -> dict[str, str]:
    if not path.is_file():
        return {}
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["session_id", "split"]:
            errors.append("splits/session_assignments.csv: expected header session_id,split")
            return assignments
        for line_number, row in enumerate(reader, 2):
            session_id = (row.get("session_id") or "").strip()
            split = (row.get("split") or "").strip()
            prefix = f"splits/session_assignments.csv:{line_number}"
            if not _valid_session_id(session_id):
                errors.append(f"{prefix}: invalid session_id")
                continue
            if split not in VALID_SPLITS:
                errors.append(f"{prefix}: split must be train, val, or test")
                continue
            previous = assignments.get(session_id)
            if previous and previous != split:
                errors.append(f"{prefix}: session {session_id!r} assigned to two splits")
            assignments[session_id] = split
    return assignments


def _validate_segment(
    root: Path,
    record: dict[str, Any],
    prefix: str,
    assignments: dict[str, str],
    session_splits: dict[str, set[str]],
    seen_ids: set[str],
    errors: list[str],
    warnings: list[str],
    require_media: bool,
) -> None:
    _require_fields(record, SEGMENT_REQUIRED_FIELDS, prefix, errors)
    _validate_common_record(record, prefix, assignments, session_splits, errors)
    _validate_unique_id(record.get("segment_id"), "segment_id", prefix, seen_ids, errors)

    if record.get("camera_role") not in VALID_CAMERAS:
        errors.append(f"{prefix}: camera_role must be tabletop or side")
    if not _nonempty_string(record.get("observation_point_id")):
        errors.append(f"{prefix}: observation_point_id must be a non-empty string")
    if record.get("sample_kind") not in VALID_SAMPLE_KINDS:
        errors.append(f"{prefix}: invalid sample_kind")
    target = record.get("target_utensil")
    if target is not None and not _nonempty_string(target):
        errors.append(f"{prefix}: target_utensil must be null or a non-empty string")

    start_ms = record.get("start_ms")
    end_ms = record.get("end_ms")
    if not _is_nonnegative_int(start_ms):
        errors.append(f"{prefix}: start_ms must be a non-negative integer")
    if not _is_nonnegative_int(end_ms) or (
        _is_nonnegative_int(start_ms) and end_ms <= start_ms
    ):
        errors.append(f"{prefix}: end_ms must be an integer greater than start_ms")

    video_path = record.get("video_path")
    expected_parts = None
    if _valid_relative_path(video_path, prefix, "video_path", errors):
        expected_parts = (
            "raw_sessions",
            record.get("session_id"),
            record.get("camera_role"),
            "videos",
        )
        parts = PurePosixPath(video_path).parts
        if len(parts) < 5 or tuple(parts[:4]) != expected_parts:
            errors.append(
                f"{prefix}: video_path must be raw_sessions/<session_id>/<camera_role>/videos/<file>"
            )
        media_path = root / Path(*parts)
        if not media_path.is_file():
            message = f"{prefix}: referenced video does not exist: {video_path}"
            (errors if require_media else warnings).append(message)


def _validate_jewelry_record(
    root: Path,
    record: dict[str, Any],
    prefix: str,
    assignments: dict[str, str],
    session_splits: dict[str, set[str]],
    seen_ids: set[str],
    class_instances: Counter[int],
    errors: list[str],
    warnings: list[str],
    require_media: bool,
) -> None:
    _require_fields(record, JEWELRY_REQUIRED_FIELDS, prefix, errors)
    _validate_common_record(record, prefix, assignments, session_splits, errors)
    _validate_unique_id(record.get("roi_id"), "roi_id", prefix, seen_ids, errors)
    if record.get("hand_side") not in VALID_HAND_SIDES:
        errors.append(f"{prefix}: hand_side must be left or right")
    if record.get("sample_kind") not in VALID_SAMPLE_KINDS:
        errors.append(f"{prefix}: invalid sample_kind")
    if not _is_nonnegative_int(record.get("frame_ms")):
        errors.append(f"{prefix}: frame_ms must be a non-negative integer")

    split = record.get("split")
    image_rel = record.get("image")
    label_rel = record.get("label")
    image_valid = _valid_relative_path(image_rel, prefix, "image", errors)
    label_valid = _valid_relative_path(label_rel, prefix, "label", errors)
    if image_valid:
        expected = ("jewelry_roi", split, "images")
        image_parts = PurePosixPath(image_rel).parts
        if len(image_parts) != 4 or tuple(image_parts[:3]) != expected:
            errors.append(f"{prefix}: image must be jewelry_roi/<split>/images/<file>")
        image_path = root / Path(*image_parts)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            errors.append(f"{prefix}: unsupported ROI image extension")
        if not image_path.is_file():
            errors.append(f"{prefix}: ROI image does not exist: {image_rel}")
    if label_valid:
        expected = ("jewelry_roi", split, "labels")
        label_parts = PurePosixPath(label_rel).parts
        if len(label_parts) != 4 or tuple(label_parts[:3]) != expected:
            errors.append(f"{prefix}: label must be jewelry_roi/<split>/labels/<file>")
        label_path = root / Path(*label_parts)
        if label_path.suffix.lower() != ".txt":
            errors.append(f"{prefix}: jewelry label must be a .txt file")
        if not label_path.is_file():
            errors.append(f"{prefix}: ROI label does not exist: {label_rel}")
        else:
            _validate_yolo_label(label_path, prefix, class_instances, errors)

    source_video = record.get("source_video")
    if _valid_relative_path(source_video, prefix, "source_video", errors):
        source_parts = PurePosixPath(source_video).parts
        expected = ("raw_sessions", record.get("session_id"), "side", "videos")
        if len(source_parts) < 5 or tuple(source_parts[:4]) != expected:
            errors.append(
                f"{prefix}: source_video must be raw_sessions/<session_id>/side/videos/<file>"
            )
        source_path = root / Path(*source_parts)
        if not source_path.is_file():
            message = f"{prefix}: referenced source video does not exist: {source_video}"
            (errors if require_media else warnings).append(message)
    if image_valid and label_valid:
        if PurePosixPath(image_rel).stem != PurePosixPath(label_rel).stem:
            errors.append(f"{prefix}: image and label stems must match")


def _validate_common_record(
    record: dict[str, Any],
    prefix: str,
    assignments: dict[str, str],
    session_splits: dict[str, set[str]],
    errors: list[str],
) -> None:
    if record.get("schema_version") != "1.0":
        errors.append(f"{prefix}: schema_version must be 1.0")
    session_id = record.get("session_id")
    split = record.get("split")
    if not _valid_session_id(session_id):
        errors.append(f"{prefix}: invalid session_id")
    if split not in VALID_SPLITS:
        errors.append(f"{prefix}: split must be train, val, or test")
    if _valid_session_id(session_id) and split in VALID_SPLITS:
        session_splits[session_id].add(split)
        assigned_split = assignments.get(session_id)
        if assigned_split is None:
            errors.append(f"{prefix}: session {session_id!r} missing from session_assignments.csv")
        elif assigned_split != split:
            errors.append(
                f"{prefix}: split {split!r} conflicts with assigned split {assigned_split!r}"
            )

    status = record.get("review_status")
    if status not in VALID_REVIEW_STATUSES:
        errors.append(f"{prefix}: invalid review_status")
    reviewer = record.get("reviewer")
    reviewed_at = record.get("reviewed_at")
    if reviewer is not None and not _nonempty_string(reviewer):
        errors.append(f"{prefix}: reviewer must be null or a non-empty string")
    if reviewed_at is not None and not _valid_iso_datetime(reviewed_at):
        errors.append(f"{prefix}: reviewed_at must be null or an ISO-8601 datetime")
    if status != "pending" and (not _nonempty_string(reviewer) or not _valid_iso_datetime(reviewed_at)):
        errors.append(f"{prefix}: reviewed records require reviewer and reviewed_at")
    if not isinstance(record.get("review_note"), str):
        errors.append(f"{prefix}: review_note must be a string")


def _validate_yolo_label(
    path: Path, prefix: str, class_instances: Counter[int], errors: list[str]
) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        label_prefix = f"{prefix} ({path.name}:{line_number})"
        if len(fields) != 5:
            errors.append(f"{label_prefix}: YOLO row must have 5 fields")
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{label_prefix}: YOLO fields must be numeric")
            continue
        if class_id not in range(len(JEWELRY_CLASSES)):
            errors.append(f"{label_prefix}: class id must be 0..2")
            continue
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label_prefix}: coordinates must be normalized to 0..1")
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{label_prefix}: width and height must be greater than zero")
            continue
        class_instances[class_id] += 1


def _validate_class_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    classes = tuple(
        line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    )
    if classes != JEWELRY_CLASSES:
        errors.append(f"jewelry_roi/classes.txt must contain: {', '.join(JEWELRY_CLASSES)}")


def _require_fields(
    record: dict[str, Any], required: Iterable[str], prefix: str, errors: list[str]
) -> None:
    missing = sorted(set(required) - set(record))
    if missing:
        errors.append(f"{prefix}: missing fields: {missing}")


def _validate_unique_id(
    value: Any, field: str, prefix: str, seen: set[str], errors: list[str]
) -> None:
    if not _nonempty_string(value):
        errors.append(f"{prefix}: {field} must be a non-empty string")
        return
    if value in seen:
        errors.append(f"{prefix}: duplicate {field}: {value}")
    seen.add(value)


def _valid_relative_path(value: Any, prefix: str, field: str, errors: list[str]) -> bool:
    if not _nonempty_string(value) or "\\" in value:
        errors.append(f"{prefix}: {field} must be a non-empty POSIX-style relative path")
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        errors.append(f"{prefix}: {field} must not be absolute or contain traversal")
        return False
    return True


def _valid_session_id(value: Any) -> bool:
    return (
        _nonempty_string(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _valid_iso_datetime(value: Any) -> bool:
    if not _nonempty_string(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create a new dataset workspace")
    init_parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)

    validate_parser = subparsers.add_parser("validate", help="validate schemas and splits")
    validate_parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    validate_parser.add_argument(
        "--require-media",
        action="store_true",
        help="treat missing source videos as errors instead of warnings",
    )
    validate_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "init":
        output = initialize_dataset(args.dataset)
        print(f"Initialized: {output}")
        return 0

    report = validate_dataset(args.dataset, require_media=args.require_media)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
