"""Publish fresh front-session labels into leak-free YOLO train/val/test releases."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = {
    **{f"front_s{index:02d}": "train" for index in range(1, 7)},
    "front_s07": "val",
    "front_s08": "test",
}


def session_id(path: Path) -> str:
    value = path.stem.split("__", 1)[0]
    if value not in SPLITS:
        raise ValueError(f"图片名缺少有效session前缀: {path.name}")
    return value


def validate_label(path: Path, task: str, class_count: int) -> None:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        values = raw.split()
        expected = 5 if task == "detect" else 14
        if len(values) != expected:
            raise ValueError(f"{path}:{line_number} 应有{expected}列，实际{len(values)}列")
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number} 类别编号越界: {class_id}")
        coordinates = [float(value) for value in values[1:]]
        if task == "detect":
            if any(not 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(f"{path}:{line_number} 坐标必须归一化到0至1")
        else:
            normalized = coordinates[:4] + [
                value for index, value in enumerate(coordinates[4:])
                if index % 3 != 2
            ]
            if any(not 0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(f"{path}:{line_number} 框和关键点坐标必须归一化到0至1")
            visibility = coordinates[6::3]
            if any(value not in {0.0, 1.0, 2.0} for value in visibility):
                raise ValueError(f"{path}:{line_number} 关键点可见性必须为0/1/2")


def publish(root: Path, task: str, version: str, allow_incomplete: bool) -> Path:
    pool_name = "detection" if task == "detect" else "pose"
    images_root = root / "derived" / pool_name / "images"
    labels_root = root / "annotations" / (
        "detection_yolo18" if task == "detect" else "pose_yolo"
    ) / "labels"
    release = root / "releases" / pool_name / version
    if release.exists() and any(release.iterdir()):
        raise FileExistsError(f"发布目录已存在且非空: {release}")

    ontology = yaml.safe_load((PROJECT / "config" / "ontology_v1.yaml").read_text(encoding="utf-8"))
    if task == "detect":
        names = {int(key): value["name"] for key, value in ontology["detect_classes"].items()}
        phase = ontology["front_training_phase"]
    else:
        pose = yaml.safe_load((PROJECT / "config" / "vessel_keypoints_v1.yaml").read_text(encoding="utf-8"))
        names = {int(key): value["name"] for key, value in pose["classes"].items()}
        phase = None

    counts = {"train": 0, "val": 0, "test": 0}
    images = [path for path in sorted(images_root.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError(f"没有待发布图片: {images_root}")
    for image in images:
        label = labels_root / f"{image.stem}.txt"
        if not label.exists():
            continue
        validate_label(label, task, len(names))
        split = SPLITS[session_id(image)]
        target_images = release / split / "images"
        target_labels = release / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_images / image.name)
        shutil.copy2(label, target_labels / label.name)
        counts[split] += 1
    if counts["train"] == 0:
        raise ValueError("发布集中没有train标注")
    if not allow_incomplete and (counts["val"] == 0 or counts["test"] == 0):
        raise ValueError(f"正式发布必须包含val和test，当前数量: {counts}")

    data = {
        "path": release.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
        "fresh_front_sessions_only": True,
        "session_split": SPLITS,
    }
    if task == "detect":
        data["active_class_ids"] = phase["active_detect_class_ids"]
        data["deferred_class_ids"] = phase["deferred_detect_class_ids"]
    else:
        data["kpt_shape"] = [3, 3]
        data["flip_idx"] = [2, 1, 0]
    (release / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (release / "release_summary.yaml").write_text(
        yaml.safe_dump({"task": task, "version": version, "counts": counts}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description="发布正面YOLO检测或关键点数据集")
    parser.add_argument("task", choices=["detect", "pose"])
    parser.add_argument("version", help="例如 front_detect_v1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true", help="仅供front_s01冒烟训练")
    args = parser.parse_args()
    release = publish(args.root.resolve(), args.task, args.version, args.allow_incomplete)
    print(f"发布完成: {release}")
    print(f"训练配置: {release / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
