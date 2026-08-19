"""Replay recorded observation events without opening a camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sop_config import SopConfigError  # noqa: E402
from src.sop_replay import (  # noqa: E402
    SopReplayError,
    load_event_records,
    replay_sop_events,
    save_replay_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay JSON/JSONL observation events through the red-tea SOP."
    )
    parser.add_argument("--events", required=True, help="JSON or JSONL event file")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "sop_red_tea_v1.yaml"),
        help="SOP YAML config",
    )
    parser.add_argument(
        "--mode",
        choices=("free_observation", "strict"),
        default="free_observation",
    )
    parser.add_argument(
        "--include-deferred",
        action="store_true",
        help="include deferred runtime nodes for simulated future-capability replay",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="include runtime-disabled adapter nodes for full SOP simulation",
    )
    parser.add_argument(
        "--available-observation",
        action="append",
        default=None,
        help="observation allow-list; repeat for multiple IDs",
    )
    parser.add_argument("--sort-events", action="store_true")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "reports" / "sop_replay.json"),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return exit code 3 unless every configured node completes or is skipped",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_event_records(args.events)
        report = replay_sop_events(
            records,
            config_path=args.config,
            mode=args.mode,
            include_deferred=args.include_deferred,
            include_disabled=args.include_disabled,
            available_observation_ids=args.available_observation,
            sort_events=args.sort_events,
        )
        output = save_replay_report(report, args.output)
    except (SopConfigError, SopReplayError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        f"Replay: {summary['final_status']}; "
        f"accepted={summary['accepted_record_count']}; "
        f"ignored={summary['ignored_record_count']}; "
        f"review={summary['review_record_count']}"
    )
    print(f"Report: {output.resolve()}")
    if args.require_complete and not summary["is_complete"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
