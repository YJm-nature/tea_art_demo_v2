"""Publish accepted reviewed images as a same-session temporal prototype."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import publish_temporal_prototype_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an experimental train/val set from one reviewed session"
    )
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--gap-frames", type=int, default=10)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking files")
    args = parser.parse_args()
    report = publish_temporal_prototype_dataset(
        args.workspace,
        args.output,
        val_ratio=args.val_ratio,
        gap_frames=args.gap_frames,
        link_mode="copy" if args.copy else "hardlink",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
