"""YOLO 数据集质量审计工具，仅读取数据，不修改训练集。"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_NUMBER_RE = re.compile(r"(\d+)$")


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    instances: int = 0
    empty_labels: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    orphan_labels: List[str] = field(default_factory=list)
    invalid_labels: List[dict] = field(default_factory=list)
    class_instances: Counter = field(default_factory=Counter)
    class_images: Counter = field(default_factory=Counter)
    source_images: Counter = field(default_factory=Counter)
    source_sequences: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))


def parse_class_names(data_yaml: Path) -> List[str]:
    """解析 Ultralytics 常见的多行 names 映射，不依赖 PyYAML。"""
    names: Dict[int, str] = {}
    for line in data_yaml.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    if not names:
        raise ValueError(f"无法从 {data_yaml} 读取类别 names 映射")
    expected = list(range(max(names) + 1))
    if sorted(names) != expected:
        raise ValueError(f"类别 ID 不连续: {sorted(names)}")
    return [names[index] for index in expected]


def infer_source(stem: str) -> str:
    """从合并数据集文件名中还原场景；auto/manual 仅是标注方式，不是新场景。"""
    source = stem.split("__", 1)[0] if "__" in stem else "unknown"
    source = re.sub(r"_(auto|manual)$", "", source)
    return source or "unknown"


def infer_frame_number(stem: str) -> Optional[int]:
    match = FRAME_NUMBER_RE.search(stem)
    return int(match.group(1)) if match else None


def audit_dataset(dataset_root: Path, near_frame_gap: int = 2) -> dict:
    root = Path(dataset_root).resolve()
    class_names = parse_class_names(root / "data.yaml")
    split_names = ["train", "val"]
    if (root / "test" / "images").exists():
        split_names.append("test")
    splits = {
        split: _scan_split(root, split, len(class_names))
        for split in split_names
    }

    class_rows = []
    for class_id, name in enumerate(class_names):
        row = {
            "id": class_id,
            "name": name,
            "total_instances": sum(splits[split].class_instances[class_id] for split in split_names),
        }
        for split in split_names:
            row[f"{split}_instances"] = splits[split].class_instances[class_id]
            row[f"{split}_images"] = splits[split].class_images[class_id]
        class_rows.append(row)

    all_sources = sorted(set().union(*(set(stats.source_images) for stats in splits.values())))
    source_rows = []
    for source in all_sources:
        row = {"source": source}
        for split in split_names:
            row[f"{split}_images"] = splits[split].source_images[source]
        source_rows.append(row)
    sources_in_both = [
        row["source"] for row in source_rows
        if sum(bool(row[f"{split}_images"]) for split in split_names) > 1
    ]
    near_pairs = _find_near_frame_leakage(splits, near_frame_gap)

    totals = [row["total_instances"] for row in class_rows]
    nonzero = [count for count in totals if count > 0]
    imbalance_ratio = max(nonzero) / min(nonzero) if nonzero else None

    warnings = []
    if sources_in_both:
        warnings.append("同一来源场景同时出现在 train/val，独立场景泛化指标会偏乐观")
    if near_pairs["count"]:
        warnings.append("相邻视频帧跨 train/val，存在高相似帧泄漏")
    if imbalance_ratio is not None and imbalance_ratio > 3:
        warnings.append("类别实例最大/最小比超过 3:1")
    if any(count == 0 for count in totals):
        warnings.append("至少一个类别没有标注实例")
    if any(splits[s].invalid_labels for s in splits):
        warnings.append("存在格式或坐标非法的 YOLO 标注")

    return {
        "schema_version": "1.0",
        "dataset_root": str(root),
        "class_names": class_names,
        "summary": {
            split: {
                "images": stats.images,
                "labels": stats.labels,
                "instances": stats.instances,
            }
            for split, stats in splits.items()
        },
        "classes": class_rows,
        "sources": source_rows,
        "imbalance": {
            "max_min_instance_ratio": round(imbalance_ratio, 3) if imbalance_ratio else None,
            "zero_instance_classes": [row["name"] for row in class_rows if row["total_instances"] == 0],
        },
        "integrity": {
            split: {
                "empty_labels": stats.empty_labels,
                "missing_labels": stats.missing_labels,
                "orphan_labels": stats.orphan_labels,
                "invalid_labels": stats.invalid_labels,
            }
            for split, stats in splits.items()
        },
        "leakage": {
            "sources_in_both_splits": sources_in_both,
            "near_frame_gap": near_frame_gap,
            "near_frame_pairs": near_pairs,
        },
        "warnings": warnings,
    }


def _scan_split(root: Path, split: str, class_count: int) -> SplitStats:
    stats = SplitStats()
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    images = {
        path.stem: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    } if image_dir.exists() else {}
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.exists() else {}
    stats.images, stats.labels = len(images), len(labels)
    stats.missing_labels = sorted(set(images) - set(labels))
    stats.orphan_labels = sorted(set(labels) - set(images))

    for stem in images:
        source = infer_source(stem)
        stats.source_images[source] += 1
        number = infer_frame_number(stem)
        if number is not None:
            stats.source_sequences[source].append(number)

    for stem, label_path in labels.items():
        if stem not in images:
            continue
        try:
            label_text = label_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            stats.invalid_labels.append({
                "file": label_path.name,
                "line": 0,
                "error": f"标签不是UTF-8/ASCII文本: {exc}",
            })
            continue
        lines = [line.strip() for line in label_text.splitlines() if line.strip()]
        if not lines:
            stats.empty_labels.append(stem)
            continue
        image_classes = set()
        for line_number, line in enumerate(lines, 1):
            error, class_id = _validate_yolo_line(line, class_count)
            if error:
                stats.invalid_labels.append({"file": label_path.name, "line": line_number, "error": error})
                continue
            stats.instances += 1
            stats.class_instances[class_id] += 1
            image_classes.add(class_id)
        for class_id in image_classes:
            stats.class_images[class_id] += 1
    return stats


def _validate_yolo_line(line: str, class_count: int) -> Tuple[Optional[str], Optional[int]]:
    fields = line.split()
    if len(fields) != 5:
        return f"应为5列，实际{len(fields)}列", None
    try:
        class_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError:
        return "包含非数字字段", None
    if not 0 <= class_id < class_count:
        return f"类别ID {class_id} 超出 0..{class_count - 1}", None
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return "归一化坐标或宽高越界", None
    if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
        return "边界框横向超出图像", None
    if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
        return "边界框纵向超出图像", None
    return None, class_id


def _find_near_frame_leakage(splits: Dict[str, SplitStats], gap: int) -> dict:
    examples = []
    count = 0
    split_names = list(splits)
    for split_a_index, split_a in enumerate(split_names):
        for split_b in split_names[split_a_index + 1:]:
            shared = set(splits[split_a].source_sequences) & set(splits[split_b].source_sequences)
            for source in sorted(shared):
                numbers_a = sorted(splits[split_a].source_sequences[source])
                numbers_b = sorted(splits[split_b].source_sequences[source])
                left = 0
                for number_b in numbers_b:
                    while left < len(numbers_a) and numbers_a[left] < number_b - gap:
                        left += 1
                    index = left
                    while index < len(numbers_a) and numbers_a[index] <= number_b + gap:
                        count += 1
                        if len(examples) < 20:
                            examples.append({
                                "source": source,
                                "split_a": split_a,
                                "frame_a": numbers_a[index],
                                "split_b": split_b,
                                "frame_b": number_b,
                            })
                        index += 1
    return {"count": count, "examples": examples}


def save_audit(report: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
