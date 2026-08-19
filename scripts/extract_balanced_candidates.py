"""Extract sharp, low-redundancy candidate frames from the material manifest.

This script creates annotation candidates only.  It never assigns train/val/test
and never overwrites source files.  Use the resulting frame_candidates.csv as
the review queue, then publish only reviewed frames with a session-level split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "dataset" / "tea_sop_front_v1"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifests" / "all_materials.csv"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

# Equal per-video sampling avoids a 6-minute full SOP video dominating the pool.
DEFAULT_LIMITS = {
    "utensil_static": 180,
    "utensil_grouped": 120,
    "occlusion_handheld": 100,
    "smell": 80,
    "hold_lotus": 80,
    "cup_layout": 60,
    "full_sop": 300,
    "legacy_full_sop": 160,
    "utensil_occlusion": 80,
    "empty_negative": 40,
}


def dhash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_frames(path: Path, requested: int, min_sharpness: float, min_hash_distance: int) -> list[tuple[int, np.ndarray, float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        capture.release()
        return []
    # Leave two seconds at each end, where most clips contain setup/teardown.
    margin = min(int(fps * 2), max(0, total // 10))
    start, end = margin, max(margin, total - margin - 1)
    positions = np.linspace(start, end, max(requested * 3, requested), dtype=np.int64)
    target_positions = set(int(value) for value in positions.tolist())
    selected: list[tuple[int, np.ndarray, float]] = []
    hashes: list[np.ndarray] = []
    frame_index = 0
    while frame_index <= end:
        ok, frame = capture.read()
        if not ok:
            break
        current_index = frame_index
        frame_index += 1
        if current_index not in target_positions:
            continue
        score = sharpness(frame)
        if score < min_sharpness:
            continue
        current = dhash(frame)
        if hashes and min(hamming(current, previous) for previous in hashes) < min_hash_distance:
            continue
        selected.append((int(current_index), frame, score))
        hashes.append(current)
        if len(selected) >= requested:
            break
    capture.release()
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="按模块抽取清晰、低重复的候选帧")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--per-video", type=int, default=40, help="每个视频最多抽取多少张，随后受模块总额限制")
    parser.add_argument("--min-sharpness", type=float, default=35.0)
    parser.add_argument("--min-hash-distance", type=int, default=8)
    parser.add_argument("--module-limit", action="append", default=[], metavar="MODULE=N", help="覆盖模块总额，可重复指定")
    parser.add_argument("--module", action="append", default=[], help="只处理指定模块，可重复指定；适合分批运行")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    limits = dict(DEFAULT_LIMITS)
    for item in args.module_limit:
        module, value = item.split("=", 1)
        limits[module] = int(value)

    enabled = set(args.module) if args.module else set(limits)
    rows = [row for row in load_rows(manifest) if row.get("kind") == "video" and row.get("module") in enabled and row.get("module") in limits]
    by_module: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_module.setdefault(row["module"], []).append(row)
    output = root / "derived" / "detection_pool" / "full_frames"
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    for module, module_rows in sorted(by_module.items()):
        quota = limits[module]
        per_video = min(args.per_video, max(1, (quota + len(module_rows) - 1) // len(module_rows)))
        module_count = 0
        for row in module_rows:
            if module_count >= quota:
                break
            source = Path(row["relative_path"])
            if not source.is_absolute():
                source = PROJECT / "dataset" / source
            selected = pick_frames(source, min(per_video, quota - module_count), args.min_sharpness, args.min_hash_distance)
            source_id = hashlib.sha1(row["relative_path"].encode("utf-8")).hexdigest()[:10]
            target_dir = output / row["session_id"] / module
            target_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, frame, score in selected:
                name = f"{row['session_id']}__{module}__{source_id}__f{frame_index:07d}.jpg"
                target = target_dir / name
                if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    continue
                candidates.append({
                    "candidate_path": target.relative_to(root).as_posix(),
                    "source_relative_path": row["relative_path"],
                    "session_id": row["session_id"], "module": module,
                    "frame_index": frame_index, "sharpness": round(score, 2),
                    "review_status": "pending", "split": "pending",
                })
                module_count += 1
    out_manifest = root / "manifests" / "frame_candidates.csv"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate_path", "source_relative_path", "session_id", "module", "frame_index", "sharpness", "review_status", "split"]
    # Module-by-module runs must accumulate instead of replacing prior review
    # records.  Re-running one module replaces only identical candidate paths.
    existing: dict[str, dict[str, str]] = {}
    if out_manifest.exists():
        with out_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("candidate_path") and (root / row["candidate_path"]).exists():
                    existing[row["candidate_path"]] = row
    for row in candidates:
        existing[str(row["candidate_path"])] = {key: str(row.get(key, "")) for key in fields}
    merged = list(existing.values())
    with out_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    print(f"本次新增/更新候选帧: {len(candidates)}，累计候选帧: {len(merged)} -> {out_manifest}")
    for module in sorted(by_module):
        count = sum(1 for row in candidates if row["module"] == module)
        print(f"{module}: {count}/{limits[module]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
