"""按完整来源场景重建 train/val，避免同一视频相邻帧泄漏。"""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_quality import IMAGE_SUFFIXES, infer_source, parse_class_names


def main() -> int:
    parser = argparse.ArgumentParser(description="将现有 YOLO 数据集按来源场景隔离拆分到新目录")
    parser.add_argument("dataset", type=Path, help="现有含 train/val 的数据集")
    parser.add_argument("output", type=Path, help="全新输出目录，不会原地修改")
    parser.add_argument(
        "--val-sources",
        nargs="+",
        required=True,
        help="完整放入验证集的来源名，如 office；先运行 audit_dataset.py 查看",
    )
    args = parser.parse_args()

    source_root = args.dataset.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖: {output_root}")

    class_names = parse_class_names(source_root / "data.yaml")
    samples = _collect_samples(source_root)
    known_sources = sorted({sample["source"] for sample in samples.values()})
    unknown = sorted(set(args.val_sources) - set(known_sources))
    if unknown:
        raise SystemExit(f"未知来源 {unknown}；可用来源: {known_sources}")
    train_sources = set(known_sources) - set(args.val_sources)
    if not train_sources:
        raise SystemExit("训练集至少需要保留一个完整来源")

    manifest = []
    for stem, sample in sorted(samples.items()):
        split = "val" if sample["source"] in args.val_sources else "train"
        image_out = output_root / split / "images" / sample["image"].name
        label_out = output_root / split / "labels" / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["image"], image_out)
        shutil.copy2(sample["label"], label_out)
        manifest.append({"stem": stem, "source": sample["source"], "split": split})

    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (output_root / "data.yaml").write_text(
        f"path: {output_root.as_posix()}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_count = sum(row["split"] == "train" for row in manifest)
    val_count = len(manifest) - train_count
    print(f"完成: train={train_count}, val={val_count}")
    print(f"训练来源: {sorted(train_sources)}")
    print(f"验证来源: {sorted(args.val_sources)}")
    print(f"输出: {output_root}")
    return 0


def _collect_samples(root: Path) -> dict:
    samples = {}
    for split in ("train", "val"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists():
            continue
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise SystemExit(f"图片缺少标签文件: {image}")
            if image.stem in samples:
                raise SystemExit(f"train/val 存在同名样本: {image.stem}")
            samples[image.stem] = {
                "image": image,
                "label": label,
                "source": infer_source(image.stem),
            }
    if not samples:
        raise SystemExit(f"未发现样本: {root}")
    return samples


if __name__ == "__main__":
    raise SystemExit(main())
