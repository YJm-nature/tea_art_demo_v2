"""Validate and import a MakeSense YOLO batch without changing acceptance status."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_annotation import parse_yolo_text
from src.dataset_rebuild import parse_yolo_labels, read_manifest, write_manifest, write_yolo_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a validated MakeSense YOLO batch")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("batch_manifest", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--reviewer", default="make-sense")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    batch_manifest_path = args.batch_manifest.resolve()
    archive = args.archive.resolve()
    batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    expected = set(batch["sample_ids"])
    if not expected:
        raise ValueError("batch_manifest没有样本")
    class_count = len((workspace / "classes.txt").read_text(encoding="utf-8-sig").splitlines())
    records = read_manifest(workspace)
    record_map = {record["sample_id"]: record for record in records}
    missing_records = expected - set(record_map)
    if missing_records:
        raise ValueError(f"batch中存在未知样本: {sorted(missing_records)[:5]}")
    if any(record_map[stem]["review_status"] == "rejected" for stem in expected):
        raise ValueError("batch包含rejected样本，拒绝导入")

    rows_by_stem = _read_archive(archive, expected, class_count)
    run_name = args.run_name or f"makesense_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    detect_dir = workspace / "pool" / "labels" / "detect"
    backup_dir = workspace / "pool" / "labels" / f"detect_before_{run_name}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)

    # Validate and stage every output before changing the working labels.
    with tempfile.TemporaryDirectory(prefix="makesense_batch_", dir=workspace) as temp_dir:
        staged = Path(temp_dir)
        for stem, rows in rows_by_stem.items():
            write_yolo_labels(staged / f"{stem}.txt", rows)
        backup_dir.mkdir(parents=True)
        for stem in sorted(expected):
            shutil.copy2(detect_dir / f"{stem}.txt", backup_dir / f"{stem}.txt")
        for stem in sorted(expected):
            shutil.copy2(staged / f"{stem}.txt", detect_dir / f"{stem}.txt")

    imported_at = datetime.now().astimezone().isoformat()
    for stem in expected:
        record = record_map[stem]
        record["last_annotation_import"] = run_name
        record["annotation_imported_at"] = imported_at
        record["annotation_import_reviewer"] = args.reviewer
        record["review_note"] = f"{run_name} MakeSense修框已导入，等待人工确认"
    write_manifest(workspace, records)
    # The transition extractor keeps a namespaced manifest for provenance;
    # keep it synchronized with the review tool's canonical manifest too.
    transition_manifest = workspace / "manifests" / "frames.jsonl"
    if transition_manifest.parent.is_dir():
        shutil.copy2(workspace / "manifest.jsonl", transition_manifest)

    report = {
        "run_name": run_name,
        "archive": str(archive),
        "batch_manifest": str(batch_manifest_path),
        "imported_labels": len(rows_by_stem),
        "backup": str(backup_dir),
        "review_status_unchanged": True,
        "class_instances": _count_classes(rows_by_stem),
    }
    report_dir = workspace / "output" / "makesense_imports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{run_name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_archive(archive: Path, expected: set[str], class_count: int) -> dict[str, list[tuple]]:
    rows_by_stem = {}
    with ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir() or entry.filename.lower().endswith(("labels.txt", "classes.txt")):
                continue
            entry_path = Path(entry.filename)
            if entry_path.name != entry.filename or entry_path.suffix.lower() != ".txt":
                raise ValueError(f"ZIP包含非法标注路径: {entry.filename}")
            stem = entry_path.stem
            if stem not in expected:
                raise ValueError(f"ZIP包含不属于本批次的标注: {entry.filename}")
            if stem in rows_by_stem:
                raise ValueError(f"ZIP包含重复标注: {entry.filename}")
            rows_by_stem[stem] = parse_yolo_text(
                bundle.read(entry).decode("utf-8-sig"), entry.filename, class_count
            )
    missing = expected - set(rows_by_stem)
    if missing:
        raise ValueError(f"ZIP缺少标注: {sorted(missing)[:10]}")
    return rows_by_stem


def _count_classes(rows_by_stem: dict[str, list[tuple]]) -> dict[str, int]:
    counts = {}
    for rows in rows_by_stem.values():
        for row in rows:
            counts[str(row[0])] = counts.get(str(row[0]), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


if __name__ == "__main__":
    raise SystemExit(main())
