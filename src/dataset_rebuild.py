"""无损数据重建、人工审核状态和正式数据发布的核心逻辑。"""

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .dataset_quality import IMAGE_SUFFIXES, infer_frame_number, infer_source


MANIFEST_NAME = "manifest.jsonl"
VALID_REVIEW_STATUSES = {"pending", "needs_fix", "accepted", "rejected"}


def load_ontology(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = data.get("detect_classes", {})
    normalized = {int(key): value for key, value in classes.items()}
    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError("detect_classes 必须从0开始连续编号")
    data["detect_classes"] = normalized
    data["legacy_tea9_mapping"] = {
        int(key): int(value) for key, value in data.get("legacy_tea9_mapping", {}).items()
    }
    phase = data.get("training_phase") or {}
    all_class_ids = set(normalized)
    active_class_ids = {int(value) for value in phase.get("active_detect_class_ids", all_class_ids)}
    deferred_class_ids = {
        int(value) for value in phase.get("deferred_detect_class_ids", all_class_ids - active_class_ids)
    }
    if active_class_ids & deferred_class_ids:
        raise ValueError("training_phase的active和deferred类别不能重叠")
    if active_class_ids | deferred_class_ids != all_class_ids:
        raise ValueError("training_phase必须完整覆盖detect_classes")
    data["training_phase"] = {
        **phase,
        "name": phase.get("name", "all_classes"),
        "active_detect_class_ids": sorted(active_class_ids),
        "deferred_detect_class_ids": sorted(deferred_class_ids),
    }
    return data


def read_manifest(root: Path) -> List[dict]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"审核清单不存在: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        required = _requires_second_review(record)
        record.setdefault("second_review_required", required)
        record.setdefault("second_review_status", "pending" if required else "not_required")
        record.setdefault("second_reviewer", None)
        record.setdefault("second_reviewed_at", None)
        record.setdefault("second_review_note", "")
    return records


def write_manifest(root: Path, records: Sequence[dict]) -> Path:
    root = Path(root)
    path = root / MANIFEST_NAME
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def prepare_review_workspace(
    source_root: Path,
    output_root: Path,
    ontology_path: Path,
    link_mode: str = "hardlink",
) -> dict:
    """合并旧train/val并创建可续做的人工审核工作区。"""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {output_root}")
    ontology = load_ontology(ontology_path)
    legacy_mapping = ontology["legacy_tea9_mapping"]
    class_count = len(ontology["detect_classes"])

    image_out_dir = output_root / "pool" / "images"
    legacy_out_dir = output_root / "pool" / "labels" / "legacy_tea9"
    detect_out_dir = output_root / "pool" / "labels" / "detect"
    for directory in (image_out_dir, legacy_out_dir, detect_out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    seen_ids = set()
    for old_split in ("train", "val"):
        image_dir = source_root / old_split / "images"
        label_dir = source_root / old_split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sample_id = image_path.stem
            if sample_id in seen_ids:
                raise ValueError(f"train/val出现同名样本: {sample_id}")
            seen_ids.add(sample_id)
            label_path = label_dir / f"{sample_id}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"缺少标签: {label_path}")

            legacy_rows = parse_yolo_labels(label_path, class_count=9)
            has_legacy_gaiwan = any(row[0] == 0 for row in legacy_rows)
            legacy_classes = {row[0] for row in legacy_rows}
            migrated_rows = [
                (legacy_mapping[row[0]], *row[1:])
                for row in legacy_rows
                if row[0] in legacy_mapping
            ]

            image_out = image_out_dir / image_path.name
            legacy_out = legacy_out_dir / label_path.name
            detect_out = detect_out_dir / label_path.name
            _link_or_copy(image_path, image_out, link_mode)
            _link_or_copy(label_path, legacy_out, link_mode)
            write_yolo_labels(detect_out, migrated_rows)

            annotation_origin = sample_id.split("__", 1)[0] if "__" in sample_id else "unknown"
            session_id = infer_source(sample_id)
            priority = 0
            if "auto" in annotation_origin:
                priority += 50
            if legacy_classes & {5, 6, 7}:
                priority += 20
            if has_legacy_gaiwan:
                priority += 10
            record = {
                "sample_id": sample_id,
                "image": image_out.relative_to(output_root).as_posix(),
                "detect_label": detect_out.relative_to(output_root).as_posix(),
                "legacy_label": legacy_out.relative_to(output_root).as_posix(),
                "source_image": str(image_path.resolve()),
                "old_split": old_split,
                "annotation_origin": annotation_origin,
                "source": session_id,
                "session_id": session_id,
                "frame_number": infer_frame_number(sample_id),
                "sha256": _sha256(image_path),
                "phash": _perceptual_hash(image_path),
                "duplicate_of": None,
                "review_status": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "review_note": "",
                "priority": priority,
                "requires_gaiwan_split": has_legacy_gaiwan,
                "migration_warnings": ["需重新标注盖碗碗身和碗盖"] if has_legacy_gaiwan else [],
            }
            record["second_review_required"] = _requires_second_review(record)
            record["second_review_status"] = (
                "pending" if record["second_review_required"] else "not_required"
            )
            record["second_reviewer"] = None
            record["second_reviewed_at"] = None
            record["second_review_note"] = ""
            records.append(record)

    _mark_duplicates(records)
    records.sort(key=lambda row: (-row["priority"], row["sample_id"]))
    write_manifest(output_root, records)
    shutil.copy2(ontology_path, output_root / "ontology_v1.yaml")
    class_names = [
        ontology["detect_classes"][index]["name"]
        for index in range(len(ontology["detect_classes"]))
    ]
    (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_root / "README_REVIEW.md").write_text(
        "# 人工审核工作区\n\n"
        "- `pool/images`：旧数据合并后的图片，默认使用硬链接。\n"
        "- `pool/labels/legacy_tea9`：只读旧9类标签。\n"
        "- `pool/labels/detect`：待修订的18类标签。\n"
        "- `manifest.jsonl`：逐图审核状态，不要用表格软件改变编码。\n\n"
        "先在CVAT或MakeSense修订detect标签，再使用 `scripts/review_dataset.py` 标记状态。\n"
        "旧盖碗框未迁移；补齐class 0碗身和class 1碗盖前不能accepted。\n",
        encoding="utf-8",
    )
    summary = review_summary(records)
    (output_root / "workspace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_yolo_labels(path: Path, class_count: int) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path.name}:{line_number} 应为5列")
        class_id = int(fields[0])
        coords = tuple(float(value) for value in fields[1:])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path.name}:{line_number} 类别ID越界: {class_id}")
        x, y, width, height = coords
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path.name}:{line_number} 坐标越界")
        rows.append((class_id, x, y, width, height))
    return rows


def write_yolo_labels(path: Path, rows: Iterable[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for class_id, x, y, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_for_acceptance(root: Path, record: dict, class_count: int) -> List[str]:
    errors = []
    label_path = Path(root) / record["detect_label"]
    try:
        rows = parse_yolo_labels(label_path, class_count)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    classes = {row[0] for row in rows}
    reviewed_absent = {int(value) for value in record.get("reviewed_absent_class_ids", [])}
    if record.get("requires_gaiwan_split"):
        if 0 not in classes and 0 not in reviewed_absent:
            errors.append("旧图含盖碗，尚未标注盖碗碗身(class 0)")
        if 1 not in classes and 1 not in reviewed_absent:
            errors.append("旧图含盖碗，尚未标注盖碗碗盖(class 1)")
    return errors


def mark_reviewed_absent_classes(
    root: Path,
    sample_id: str,
    class_ids: Sequence[int],
    reviewer: str,
    note: str = "",
) -> dict:
    """Record that reviewed classes are intentionally not visible in this frame."""
    root = Path(root)
    ontology = load_ontology(root / "ontology_v1.yaml")
    class_count = len(ontology["detect_classes"])
    requested = {int(value) for value in class_ids}
    if not requested or any(value < 0 or value >= class_count for value in requested):
        raise ValueError(f"无效缺席类别: {sorted(requested)}")
    records = read_manifest(root)
    target = next((record for record in records if record["sample_id"] == sample_id), None)
    if target is None:
        raise KeyError(f"未找到样本: {sample_id}")
    present = {
        row[0] for row in parse_yolo_labels(root / target["detect_label"], class_count)
    }
    conflicts = requested & present
    if conflicts:
        raise ValueError(f"类别仍有标注框，不能同时标记缺席: {sorted(conflicts)}")
    target["reviewed_absent_class_ids"] = sorted(
        requested | {int(value) for value in target.get("reviewed_absent_class_ids", [])}
    )
    target["absence_reviewer"] = reviewer
    target["absence_reviewed_at"] = datetime.now().astimezone().isoformat()
    target["absence_review_note"] = note
    write_manifest(root, records)
    return target


def set_review_status(
    root: Path,
    sample_id: str,
    status: str,
    reviewer: str,
    note: str = "",
    ontology_path: Optional[Path] = None,
) -> dict:
    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"无效状态: {status}")
    root = Path(root)
    ontology = load_ontology(ontology_path or root / "ontology_v1.yaml")
    records = read_manifest(root)
    target = next((record for record in records if record["sample_id"] == sample_id), None)
    if target is None:
        raise KeyError(f"未找到样本: {sample_id}")
    if status == "accepted":
        errors = validate_for_acceptance(root, target, len(ontology["detect_classes"]))
        if errors:
            raise ValueError("；".join(errors))
    target.update(
        review_status=status,
        reviewer=reviewer,
        reviewed_at=datetime.now().astimezone().isoformat(),
        review_note=note,
    )
    write_manifest(root, records)
    return target


def set_batch_review_status(
    root: Path,
    sample_ids: Sequence[str],
    status: str,
    reviewer: str,
    note: str = "",
    ontology_path: Optional[Path] = None,
) -> dict:
    """Validate a complete batch first, then update all review states atomically."""
    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"无效状态: {status}")
    root = Path(root)
    ontology = load_ontology(ontology_path or root / "ontology_v1.yaml")
    records = read_manifest(root)
    record_map = {record["sample_id"]: record for record in records}
    unique_ids = list(dict.fromkeys(sample_ids))
    missing = [sample_id for sample_id in unique_ids if sample_id not in record_map]
    if missing:
        raise KeyError(f"未找到batch样本: {missing[:10]}")
    if status == "accepted":
        errors = {}
        for sample_id in unique_ids:
            sample_errors = validate_for_acceptance(
                root, record_map[sample_id], len(ontology["detect_classes"])
            )
            if sample_errors:
                errors[sample_id] = sample_errors
        if errors:
            raise ValueError(f"batch存在验收错误: {errors}")

    reviewed_at = datetime.now().astimezone().isoformat()
    for sample_id in unique_ids:
        record_map[sample_id].update(
            review_status=status,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            review_note=note,
        )
    write_manifest(root, records)
    return {
        "updated": len(unique_ids),
        "status": status,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
    }


def set_second_review_status(
    root: Path,
    sample_id: str,
    status: str,
    reviewer: str,
    note: str = "",
) -> dict:
    if status not in {"accepted", "rejected"}:
        raise ValueError("二审状态只能是 accepted 或 rejected")
    root = Path(root)
    records = read_manifest(root)
    target = next((record for record in records if record["sample_id"] == sample_id), None)
    if target is None:
        raise KeyError(f"未找到样本: {sample_id}")
    if not target["second_review_required"]:
        raise ValueError("该样本未被抽中二审")
    if target["review_status"] != "accepted":
        raise ValueError("主审尚未accepted")
    if target.get("reviewer") == reviewer:
        raise ValueError("二审人员不能与主审相同")
    target.update(
        second_review_status=status,
        second_reviewer=reviewer,
        second_reviewed_at=datetime.now().astimezone().isoformat(),
        second_review_note=note,
    )
    if status == "rejected":
        target["review_status"] = "needs_fix"
        target["review_note"] = f"二审退回: {note}" if note else "二审退回"
    write_manifest(root, records)
    return target


def review_summary(records: Sequence[dict]) -> dict:
    statuses = Counter(record["review_status"] for record in records)
    second_statuses = Counter(record["second_review_status"] for record in records)
    origins = Counter(record["annotation_origin"] for record in records)
    sessions = Counter(record["session_id"] for record in records)
    return {
        "total": len(records),
        "statuses": dict(sorted(statuses.items())),
        "second_review_statuses": dict(sorted(second_statuses.items())),
        "origins": dict(sorted(origins.items())),
        "sessions": dict(sorted(sessions.items())),
        "duplicates": sum(record.get("duplicate_of") is not None for record in records),
        "requires_gaiwan_split": sum(record.get("requires_gaiwan_split", False) for record in records),
    }


def assign_sessions(
    records: Sequence[dict],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    explicit: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """按session整体分配；显式映射优先，其余按目标图片数贪心分配。"""
    valid_splits = ("train", "val", "test")
    explicit = dict(explicit or {})
    if any(split not in valid_splits for split in explicit.values()):
        raise ValueError("显式split只能是 train/val/test")
    sizes = Counter(record["session_id"] for record in records if record["review_status"] == "accepted")
    unknown = set(explicit) - set(sizes)
    if unknown:
        raise ValueError(f"显式映射包含未知session: {sorted(unknown)}")

    assignment = dict(explicit)
    current = Counter()
    for session, split in assignment.items():
        current[split] += sizes[session]
    total = sum(sizes.values())
    targets = {split: total * ratio for split, ratio in zip(valid_splits, ratios)}
    remaining = [session for session in sizes if session not in assignment]
    random.Random(seed).shuffle(remaining)
    remaining.sort(key=lambda session: sizes[session], reverse=True)
    for session in remaining:
        split = max(valid_splits, key=lambda name: targets[name] - current[name])
        assignment[session] = split
        current[split] += sizes[session]
    return assignment


def publish_reviewed_dataset(
    review_root: Path,
    output_root: Path,
    explicit_assignments: Optional[Dict[str, str]] = None,
    allow_prototype: bool = False,
    allow_pending_second_review: bool = False,
    link_mode: str = "hardlink",
) -> dict:
    review_root = Path(review_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {output_root}")
    ontology = load_ontology(review_root / "ontology_v1.yaml")
    active_class_ids = set(ontology["training_phase"]["active_detect_class_ids"])
    records = read_manifest(review_root)
    blocking = [record for record in records if record["review_status"] in {"pending", "needs_fix"}]
    if blocking:
        raise ValueError(f"仍有 {len(blocking)} 张未完成审核，拒绝发布")
    second_blocking = [
        record for record in records
        if record["review_status"] == "accepted"
        and record["second_review_required"]
        and record["second_review_status"] != "accepted"
    ]
    if allow_pending_second_review and not allow_prototype:
        raise ValueError("仅原型发布可使用allow_pending_second_review")
    if second_blocking and not allow_pending_second_review:
        raise ValueError(f"仍有 {len(second_blocking)} 张未完成二审，拒绝发布")
    accepted = [record for record in records if record["review_status"] == "accepted"]
    if not accepted:
        raise ValueError("没有accepted样本")
    for record in accepted:
        errors = validate_for_acceptance(review_root, record, len(ontology["detect_classes"]))
        if errors:
            raise ValueError(f"{record['sample_id']}: {'；'.join(errors)}")

    record_map = {record["sample_id"]: record for record in records}
    deduplicated = []
    excluded_duplicates = []
    seen_duplicate_roots = set()
    for record in accepted:
        root_id = record["sample_id"]
        visited = set()
        while record_map.get(root_id, {}).get("duplicate_of") and root_id not in visited:
            visited.add(root_id)
            root_id = record_map[root_id]["duplicate_of"]
        if root_id in seen_duplicate_roots:
            excluded_duplicates.append(record["sample_id"])
            continue
        seen_duplicate_roots.add(root_id)
        deduplicated.append(record)

    assignment = assign_sessions(deduplicated, explicit=explicit_assignments)
    class_sessions: Dict[int, set] = defaultdict(set)
    for record in deduplicated:
        source_label = review_root / record["detect_label"]
        for row in parse_yolo_labels(source_label, len(ontology["detect_classes"])):
            if row[0] in active_class_ids:
                class_sessions[row[0]].add(record["session_id"])

    minimum_sessions = int(ontology.get("quality_gates", {}).get("min_sessions_per_class", 5))
    weak_classes = {
        ontology["detect_classes"][class_id]["name"]: len(class_sessions[class_id])
        for class_id in active_class_ids
        if len(class_sessions[class_id]) < minimum_sessions
    }
    if weak_classes and not allow_prototype:
        raise ValueError(f"类别独立session不足，使用--allow-prototype仅发布原型集: {weak_classes}")

    published_manifest = []
    excluded_deferred_instances = Counter()
    for record in deduplicated:
        split = assignment[record["session_id"]]
        source_image = review_root / record["image"]
        source_label = review_root / record["detect_label"]
        image_out = output_root / split / "images" / source_image.name
        label_out = output_root / split / "labels" / source_label.name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(source_image, image_out, link_mode)
        rows = parse_yolo_labels(source_label, len(ontology["detect_classes"]))
        active_rows = []
        for row in rows:
            if row[0] in active_class_ids:
                active_rows.append(row)
            else:
                excluded_deferred_instances[row[0]] += 1
        write_yolo_labels(label_out, active_rows)
        published_manifest.append({
            "sample_id": record["sample_id"],
            "split": split,
            "session_id": record["session_id"],
            "source": record["source"],
            "frame_number": record.get("frame_number"),
            "sha256": record.get("sha256"),
            "image": image_out.relative_to(output_root).as_posix(),
            "label": label_out.relative_to(output_root).as_posix(),
        })

    class_names = [ontology["detect_classes"][index]["name"] for index in range(len(ontology["detect_classes"]))]
    _write_data_yaml(
        output_root,
        class_names,
        ontology["training_phase"],
        metadata={
            "prototype_pending_second_review": bool(second_blocking),
            "pending_second_review_samples": len(second_blocking),
            "duplicate_candidates_excluded": len(excluded_duplicates),
        },
    )
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in published_manifest),
        encoding="utf-8",
    )
    shutil.copy2(review_root / "ontology_v1.yaml", output_root / "ontology_v1.yaml")
    split_counts = Counter(row["split"] for row in published_manifest)
    report = {
        "samples": len(published_manifest),
        "accepted_source_samples": len(accepted),
        "split_counts": dict(split_counts),
        "session_assignment": assignment,
        "weak_classes": weak_classes,
        "prototype": bool(weak_classes),
        "provisional_pending_second_review": bool(second_blocking),
        "pending_second_review_samples": len(second_blocking),
        "excluded_duplicate_candidates": len(excluded_duplicates),
        "excluded_duplicate_sample_ids": excluded_duplicates,
        "excluded_deferred_instances": {
            ontology["detect_classes"][class_id]["name"]: count
            for class_id, count in sorted(excluded_deferred_instances.items())
        },
        "training_phase": ontology["training_phase"]["name"],
        "active_class_ids": sorted(active_class_ids),
        "deferred_class_ids": ontology["training_phase"]["deferred_detect_class_ids"],
    }
    (output_root / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def publish_temporal_prototype_dataset(
    review_root: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    gap_frames: int = 10,
    link_mode: str = "hardlink",
) -> dict:
    """Publish accepted samples from one session for prototype training only.

    The validation portion is the final contiguous time block. A frame gap is
    removed at the train/validation boundary to reduce adjacent-frame leakage.
    This split is useful for pipeline and model experiments, but it is not an
    independent evaluation set.
    """
    review_root = Path(review_root).resolve()
    output_root = Path(output_root).resolve()
    if not 0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")
    if gap_frames < 0:
        raise ValueError("gap_frames must be non-negative")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    ontology = load_ontology(review_root / "ontology_v1.yaml")
    class_count = len(ontology["detect_classes"])
    active_class_ids = set(ontology["training_phase"]["active_detect_class_ids"])
    records = read_manifest(review_root)
    accepted = [record for record in records if record["review_status"] == "accepted"]
    if len(accepted) < 20:
        raise ValueError("At least 20 accepted samples are required for a temporal prototype")

    sessions = {record["session_id"] for record in accepted}
    if len(sessions) != 1:
        raise ValueError(
            "Temporal prototype publishing requires exactly one accepted session; "
            f"found {sorted(sessions)}"
        )
    if any(record.get("frame_number") is None for record in accepted):
        raise ValueError("Every accepted sample must have a frame_number")

    seen_hashes = set()
    unique_records = []
    exact_duplicates = []
    for record in sorted(accepted, key=lambda row: (row["frame_number"], row["sample_id"])):
        errors = validate_for_acceptance(review_root, record, class_count)
        if errors:
            raise ValueError(f"{record['sample_id']}: {'; '.join(errors)}")
        rows = parse_yolo_labels(review_root / record["detect_label"], class_count)
        unexpected = sorted({row[0] for row in rows} - active_class_ids)
        if unexpected:
            raise ValueError(
                f"{record['sample_id']}: labels contain deferred class IDs {unexpected}"
            )
        image_path = review_root / record["image"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        digest = record.get("sha256") or _sha256(image_path)
        if digest in seen_hashes:
            exact_duplicates.append(record["sample_id"])
            continue
        seen_hashes.add(digest)
        unique_records.append(record)

    val_count = max(1, int(round(len(unique_records) * val_ratio)))
    val_records = unique_records[-val_count:]
    val_start_frame = int(val_records[0]["frame_number"])
    train_records = [
        record for record in unique_records[:-val_count]
        if int(record["frame_number"]) < val_start_frame - gap_frames
    ]
    selected_ids = {record["sample_id"] for record in train_records + val_records}
    gap_records = [
        record for record in unique_records
        if record["sample_id"] not in selected_ids
    ]
    if not train_records or not val_records:
        raise ValueError("Temporal split produced an empty train or validation set")

    published_manifest = []
    class_instances = {"train": Counter(), "val": Counter()}
    for split, split_records in (("train", train_records), ("val", val_records)):
        for record in split_records:
            source_image = review_root / record["image"]
            source_label = review_root / record["detect_label"]
            image_out = output_root / split / "images" / source_image.name
            label_out = output_root / split / "labels" / source_label.name
            _link_or_copy(source_image, image_out, link_mode)
            _link_or_copy(source_label, label_out, link_mode)
            for row in parse_yolo_labels(source_label, class_count):
                class_instances[split][row[0]] += 1
            published_manifest.append({
                "sample_id": record["sample_id"],
                "split": split,
                "session_id": record["session_id"],
                "frame_number": record["frame_number"],
                "sha256": record.get("sha256"),
                "image": image_out.relative_to(output_root).as_posix(),
                "label": label_out.relative_to(output_root).as_posix(),
                "prototype_same_session_holdout": True,
            })

    class_names = [
        ontology["detect_classes"][index]["name"] for index in range(class_count)
    ]
    _write_data_yaml(
        output_root,
        class_names,
        ontology["training_phase"],
        include_test=False,
        metadata={
            "prototype_same_session_holdout": True,
            "evaluation_warning": "same_session_temporal_holdout_not_independent_test",
        },
    )
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in published_manifest),
        encoding="utf-8",
    )
    shutil.copy2(review_root / "ontology_v1.yaml", output_root / "ontology_v1.yaml")

    status_counts = Counter(record["review_status"] for record in records)
    second_review_pending = sum(
        record.get("second_review_required")
        and record.get("second_review_status") != "accepted"
        for record in accepted
    )
    report = {
        "prototype": True,
        "prototype_same_session_holdout": True,
        "evaluation_warning": "Validation is from the same session and is not formal model evidence.",
        "session_id": next(iter(sessions)),
        "accepted_available": len(accepted),
        "split_counts": {"train": len(train_records), "val": len(val_records)},
        "val_ratio": val_ratio,
        "gap_frames": gap_frames,
        "train_frame_range": [train_records[0]["frame_number"], train_records[-1]["frame_number"]],
        "val_frame_range": [val_records[0]["frame_number"], val_records[-1]["frame_number"]],
        "excluded_boundary_samples": len(gap_records),
        "excluded_exact_duplicates": len(exact_duplicates),
        "unfinished_samples_not_published": status_counts["pending"] + status_counts["needs_fix"],
        "rejected_samples_not_published": status_counts["rejected"],
        "accepted_samples_pending_second_review": second_review_pending,
        "active_class_ids": sorted(active_class_ids),
        "class_instances": {
            split: {
                class_names[class_id]: class_instances[split][class_id]
                for class_id in sorted(active_class_ids)
            }
            for split in ("train", "val")
        },
    }
    (output_root / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def validate_published_dataset(dataset_root: Path, min_sessions_per_class: int = 5) -> dict:
    """Check files, split isolation, and coverage for the ontology's active training phase."""
    dataset_root = Path(dataset_root).resolve()
    ontology = load_ontology(dataset_root / "ontology_v1.yaml")
    active_class_ids = set(ontology["training_phase"]["active_detect_class_ids"])
    deferred_class_ids = set(ontology["training_phase"]["deferred_detect_class_ids"])
    manifest_path = dataset_root / "manifest.jsonl"
    records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = []
    warnings = []
    session_splits: Dict[str, set] = defaultdict(set)
    hash_splits: Dict[str, set] = defaultdict(set)
    class_sessions: Dict[int, set] = defaultdict(set)
    class_instances = Counter()
    split_counts = Counter()

    for record in records:
        split = record.get("split")
        session = record.get("session_id")
        if split not in {"train", "val", "test"}:
            errors.append(f"{record.get('sample_id')}: 无效split {split}")
            continue
        split_counts[split] += 1
        session_splits[session].add(split)
        if record.get("sha256"):
            hash_splits[record["sha256"]].add(split)
        image_path = dataset_root / record["image"]
        label_path = dataset_root / record["label"]
        if not image_path.exists():
            errors.append(f"缺少图片: {record['image']}")
        if not label_path.exists():
            errors.append(f"缺少标签: {record['label']}")
            continue
        try:
            rows = parse_yolo_labels(label_path, len(ontology["detect_classes"]))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for row in rows:
            class_instances[row[0]] += 1
            class_sessions[row[0]].add(session)

    leaking_sessions = {
        session: sorted(splits) for session, splits in session_splits.items() if len(splits) > 1
    }
    if leaking_sessions:
        errors.append(f"session跨集合泄漏: {leaking_sessions}")
    leaking_hashes = sum(1 for splits in hash_splits.values() if len(splits) > 1)
    if leaking_hashes:
        errors.append(f"有 {leaking_hashes} 个相同SHA256图片跨集合")
    missing_splits = {"train", "val", "test"} - set(split_counts)
    if missing_splits:
        errors.append(f"缺少数据划分: {sorted(missing_splits)}")

    weak_classes = {}
    zero_active_classes = []
    unexpected_deferred_classes = []
    for class_id, config in ontology["detect_classes"].items():
        session_count = len(class_sessions[class_id])
        if class_id in active_class_ids and class_instances[class_id] == 0:
            zero_active_classes.append(config["name"])
        if class_id in active_class_ids and session_count < min_sessions_per_class:
            weak_classes[config["name"]] = session_count
        if class_id in deferred_class_ids and class_instances[class_id] > 0:
            unexpected_deferred_classes.append(config["name"])
    if zero_active_classes:
        errors.append(f"无实例的当前训练类别: {zero_active_classes}")
    if unexpected_deferred_classes:
        errors.append(f"延期类别意外混入当前数据: {unexpected_deferred_classes}")
    if weak_classes:
        warnings.append(f"类别独立session少于{min_sessions_per_class}: {weak_classes}")

    return {
        "dataset_root": str(dataset_root),
        "valid": not errors,
        "samples": len(records),
        "training_phase": ontology["training_phase"]["name"],
        "active_class_ids": sorted(active_class_ids),
        "deferred_class_ids": sorted(deferred_class_ids),
        "split_counts": dict(split_counts),
        "session_counts": dict(Counter(record.get("session_id") for record in records)),
        "class_instances": {
            ontology["detect_classes"][class_id]["name"]: class_instances[class_id]
            for class_id in ontology["detect_classes"]
        },
        "class_sessions": {
            ontology["detect_classes"][class_id]["name"]: len(class_sessions[class_id])
            for class_id in ontology["detect_classes"]
        },
        "errors": errors,
        "warnings": warnings,
    }


def _write_data_yaml(
    root: Path,
    names: Sequence[str],
    training_phase: dict,
    include_test: bool = True,
    metadata: Optional[dict] = None,
) -> None:
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    active_ids = training_phase["active_detect_class_ids"]
    deferred_ids = training_phase["deferred_detect_class_ids"]
    test_line = "test: test/images\n" if include_test else ""
    metadata_lines = "".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}\n"
        for key, value in (metadata or {}).items()
    )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\n{test_line}\n"
        f"nc: {len(names)}\nnames:\n{names_yaml}\n\n"
        f"training_phase: {training_phase['name']}\n"
        f"active_class_ids: {active_ids}\n"
        f"deferred_class_ids: {deferred_ids}\n"
        f"{metadata_lines}",
        encoding="utf-8",
    )


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            shutil.copy2(source, target)
            return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    raise ValueError("link_mode必须是 hardlink 或 copy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[:8, :8]
    values = dct.flatten()[1:]
    bits = values > np.median(values)
    number = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{number:016x}"


def _mark_duplicates(records: List[dict], max_hamming: int = 2) -> None:
    exact_seen = {}
    by_session: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        if record["sha256"] in exact_seen:
            record["duplicate_of"] = exact_seen[record["sha256"]]
            record["priority"] -= 30
        else:
            exact_seen[record["sha256"]] = record["sample_id"]
        by_session[record["session_id"]].append(record)
    for session_records in by_session.values():
        for index, record in enumerate(session_records):
            if record["duplicate_of"] is not None:
                continue
            value = int(record["phash"], 16)
            for previous in session_records[:index]:
                if (value ^ int(previous["phash"], 16)).bit_count() <= max_hamming:
                    record["duplicate_of"] = previous["sample_id"]
                    record["priority"] -= 20
                    break


def _requires_second_review(record: dict) -> bool:
    """高风险样本稳定抽50%，其它样本稳定抽20%。"""
    high_risk = "auto" in record.get("annotation_origin", "") or record.get("priority", 0) >= 20
    rate = 50 if high_risk else 20
    digest = record.get("sha256") or hashlib.sha256(record["sample_id"].encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < rate
