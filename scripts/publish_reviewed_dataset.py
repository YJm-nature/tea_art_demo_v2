"""将完成审核的阶段数据按session隔离发布为train/val/test。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import publish_reviewed_dataset


def _parse_assignments(values):
    assignments = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"映射应为 session=split: {value}")
        session, split = value.split("=", 1)
        assignments[session] = split
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description="发布固定18类编号的当前训练阶段YOLO数据集")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--assign",
        nargs="*",
        default=[],
        metavar="SESSION=SPLIT",
        help="例如 office=train original=val focus=test",
    )
    parser.add_argument("--allow-prototype", action="store_true", help="允许类别session少于5个")
    parser.add_argument(
        "--allow-pending-second-review",
        action="store_true",
        help="仅原型候选集允许发布尚未完成独立二审的accepted样本，并写入警告",
    )
    parser.add_argument("--copy", action="store_true", help="复制而非硬链接")
    args = parser.parse_args()
    report = publish_reviewed_dataset(
        args.workspace,
        args.output,
        explicit_assignments=_parse_assignments(args.assign),
        allow_prototype=args.allow_prototype,
        allow_pending_second_review=args.allow_pending_second_review,
        link_mode="copy" if args.copy else "hardlink",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
