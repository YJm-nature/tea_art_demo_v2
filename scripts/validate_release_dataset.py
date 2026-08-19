"""当前训练阶段开始前的数据质量门禁。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import validate_published_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="验证固定18类编号的阶段train/val/test发布物")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-sessions-per-class", type=int, default=5)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate_published_dataset(args.dataset, args.min_sessions_per_class)
    output = args.report or args.dataset / "release_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {output.resolve()}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
