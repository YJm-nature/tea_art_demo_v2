"""从旧tea9 train/val无损创建18类人工审核工作区。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import prepare_review_workspace


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="创建逐图审核工作区，不修改旧数据")
    parser.add_argument(
        "--source",
        type=Path,
        default=project / "dataset" / "final_tea9_dataset_20260723",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "dataset" / "tea_dataset_v1_reviewed",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=project / "config" / "ontology_v1.yaml",
    )
    parser.add_argument("--copy", action="store_true", help="复制文件；默认优先硬链接")
    args = parser.parse_args()
    summary = prepare_review_workspace(
        args.source,
        args.output,
        args.ontology,
        link_mode="copy" if args.copy else "hardlink",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"审核工作区: {args.output.resolve()}")
    print(f"下一步: python scripts/review_dataset.py {args.output} summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
