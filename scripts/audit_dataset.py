"""命令行数据集审计：不会修改任何图片或标签。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_quality import audit_dataset, save_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 YOLO 数据集分布、完整性和视频帧泄漏")
    parser.add_argument("dataset", type=Path, help="包含 data.yaml 和 train/val 的数据集目录")
    parser.add_argument("--output", type=Path, default=None, help="JSON 报告路径")
    parser.add_argument("--near-frame-gap", type=int, default=2, help="相邻帧泄漏检测间隔")
    args = parser.parse_args()

    report = audit_dataset(args.dataset, near_frame_gap=args.near_frame_gap)
    output = args.output or args.dataset / "audit_report.json"
    save_audit(report, output)

    print(f"数据集: {report['dataset_root']}")
    for split, summary in report["summary"].items():
        print(f"  {split}: {summary['images']} images, {summary['instances']} instances")
    print("\n类别分布:")
    for row in report["classes"]:
        split_text = " ".join(
            f"{split}={row.get(f'{split}_instances', 0):<5}"
            for split in report["summary"]
        )
        print(f"  {row['id']:>2} {row['name']:<8} {split_text} total={row['total_instances']}")
    print("\n来源分布:")
    for row in report["sources"]:
        split_text = " ".join(
            f"{split}={row.get(f'{split}_images', 0):<5}"
            for split in report["summary"]
        )
        print(f"  {row['source']:<16} {split_text}")
    print(f"\n相邻帧跨集合: {report['leakage']['near_frame_pairs']['count']} 对")
    for warning in report["warnings"]:
        print(f"  [WARN] {warning}")
    print(f"\nJSON报告: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
