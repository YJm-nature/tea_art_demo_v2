"""逐图审核状态工具；修框使用CVAT/MakeSense，本工具记录审核闭环。"""

import argparse
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_rebuild import (
    load_ontology,
    mark_reviewed_absent_classes,
    parse_yolo_labels,
    read_manifest,
    review_summary,
    set_batch_review_status,
    set_second_review_status,
    set_review_status,
)


STATUS_COLORS = {
    "pending": (0, 200, 255),
    "needs_fix": (0, 120, 255),
    "accepted": (80, 210, 80),
    "rejected": (80, 80, 230),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="查看、统计并记录逐图审核状态")
    parser.add_argument("workspace", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary")

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("sample_id")
    set_parser.add_argument("status", choices=list(STATUS_COLORS))
    set_parser.add_argument("--reviewer", required=True)
    set_parser.add_argument("--note", default="")

    batch_parser = subparsers.add_parser("batch-set")
    batch_parser.add_argument("batch_manifest", type=Path)
    batch_parser.add_argument("status", choices=list(STATUS_COLORS))
    batch_parser.add_argument("--reviewer", required=True)
    batch_parser.add_argument("--note", default="")

    absent_parser = subparsers.add_parser("mark-absent")
    absent_parser.add_argument("sample_id")
    absent_parser.add_argument("class_ids", nargs="+", type=int)
    absent_parser.add_argument("--reviewer", required=True)
    absent_parser.add_argument("--note", default="")

    second_parser = subparsers.add_parser("second-set")
    second_parser.add_argument("sample_id")
    second_parser.add_argument("status", choices=["accepted", "rejected"])
    second_parser.add_argument("--reviewer", required=True)
    second_parser.add_argument("--note", default="")

    gui_parser = subparsers.add_parser("gui")
    gui_parser.add_argument("--reviewer", required=True)
    gui_parser.add_argument("--status", choices=["all", *STATUS_COLORS], default="pending")
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if args.command == "summary":
        print(json.dumps(review_summary(read_manifest(workspace)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "set":
        record = set_review_status(
            workspace, args.sample_id, args.status, args.reviewer, args.note
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch-set":
        batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
        report = set_batch_review_status(
            workspace, batch["sample_ids"], args.status, args.reviewer, args.note
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "mark-absent":
        record = mark_reviewed_absent_classes(
            workspace, args.sample_id, args.class_ids, args.reviewer, args.note
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    if args.command == "second-set":
        record = set_second_review_status(
            workspace, args.sample_id, args.status, args.reviewer, args.note
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    return run_gui(workspace, args.reviewer, args.status)


def run_gui(workspace: Path, reviewer: str, status_filter: str) -> int:
    ontology = load_ontology(workspace / "ontology_v1.yaml")
    class_names = {
        class_id: config["key"] for class_id, config in ontology["detect_classes"].items()
    }
    records = read_manifest(workspace)
    indexes = [
        index for index, record in enumerate(records)
        if status_filter == "all" or record["review_status"] == status_filter
    ]
    if not indexes:
        print("没有符合筛选条件的图片")
        return 0
    cursor = 0
    message = "A accept | F needs_fix | R reject | N/P navigate | Q quit"
    cv2.namedWindow("Tea Dataset Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tea Dataset Review", 1400, 900)
    while True:
        record = records[indexes[cursor]]
        canvas = _render(workspace, record, class_names, cursor + 1, len(indexes), message)
        cv2.imshow("Tea Dataset Review", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):
            cursor = min(len(indexes) - 1, cursor + 1)
            message = "next"
        elif key in (ord("p"), 81):
            cursor = max(0, cursor - 1)
            message = "previous"
        elif key in (ord("a"), ord("f"), ord("r")):
            status = {ord("a"): "accepted", ord("f"): "needs_fix", ord("r"): "rejected"}[key]
            try:
                set_review_status(workspace, record["sample_id"], status, reviewer)
                records = read_manifest(workspace)
                message = f"saved: {status}"
                if cursor < len(indexes) - 1:
                    cursor += 1
            except ValueError as exc:
                set_review_status(workspace, record["sample_id"], "needs_fix", reviewer, str(exc))
                records = read_manifest(workspace)
                message = f"cannot accept: {exc}"
        if cv2.getWindowProperty("Tea Dataset Review", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    print(json.dumps(review_summary(read_manifest(workspace)), ensure_ascii=False, indent=2))
    return 0


def _render(
    workspace: Path,
    record: dict,
    class_names: dict,
    position: int,
    total: int,
    message: str,
):
    image = cv2.imread(str(workspace / record["image"]))
    if image is None:
        raise ValueError(f"无法读取 {record['image']}")
    height, width = image.shape[:2]
    for class_id, x, y, box_width, box_height in parse_yolo_labels(
        workspace / record["detect_label"], len(class_names)
    ):
        x1 = int((x - box_width / 2) * width)
        y1 = int((y - box_height / 2) * height)
        x2 = int((x + box_width / 2) * width)
        y2 = int((y + box_height / 2) * height)
        color = (60 + class_id * 37 % 190, 210 - class_id * 19 % 150, 80 + class_id * 53 % 170)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_id}:{class_names[class_id]}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    scale = min(1400 / width, 800 / height, 1.0)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))
    panel = cv2.copyMakeBorder(image, 72, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 30))
    status = record["review_status"]
    header = f"[{position}/{total}] {record['sample_id']} | {status} | session={record['session_id']}"
    cv2.putText(panel, header, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                STATUS_COLORS[status], 1, cv2.LINE_AA)
    cv2.putText(panel, message[:180], (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (220, 220, 220), 1, cv2.LINE_AA)
    return panel


if __name__ == "__main__":
    raise SystemExit(main())
