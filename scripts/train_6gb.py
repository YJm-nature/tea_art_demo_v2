"""GTX 1060 6GB顺序训练、验证、基准测试和ONNX导出入口。"""

import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "config" / "training_6gb.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="6GB显存YOLO训练工作流")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印参数")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "smoke", "stage1", "stage2", "stage1280",
        "pose-smoke", "pose-stage1", "pose-stage2",
        "segment",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("data", type=Path)
        child.add_argument("--model", type=str, default=None)
        child.add_argument("--name", default=None)

    validate = subparsers.add_parser("validate")
    validate.add_argument("data", type=Path)
    validate.add_argument("model", type=Path)
    validate.add_argument("--split", choices=["val", "test"], default="test")
    validate.add_argument("--imgsz", type=int, default=960)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("data", type=Path)
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--imgsz", type=int, default=832)
    benchmark.add_argument("--limit", type=int, default=100)

    export = subparsers.add_parser("export")
    export.add_argument("model", type=Path)
    export.add_argument("--imgsz", type=int, default=832)

    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command in {
        "smoke", "stage1", "stage2", "stage1280",
        "pose-smoke", "pose-stage1", "pose-stage2",
        "segment",
    }:
        return run_training(args, config)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, default=str, indent=2))
        return 0
    _preflight()
    if args.command == "validate":
        return run_validation(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    return run_export(args)


def run_training(args, config: dict) -> int:
    data_yaml = args.data.resolve()
    is_pose = args.command.startswith("pose-")
    is_smoke = args.command in {"smoke", "pose-smoke"}
    _validate_data_yaml(
        data_yaml,
        require_test=not is_smoke,
        task="pose" if is_pose else ("segment" if args.command == "segment" else "detect"),
    )
    project = PROJECT / "models" / "low_vram"
    common = dict(config["common"])

    if is_smoke:
        task_config = "pose" if is_pose else "detect"
        stage = dict(config[task_config]["smoke"])
        smoke_dir = PROJECT / "output" / "smoke_dataset"
        data_yaml = _prepare_smoke_yaml(data_yaml, smoke_dir, limit=100)
        candidates = [stage]
        default_model = Path("yolov8n-pose.pt") if is_pose else PROJECT / "yolov8n.pt"
    elif args.command in {"stage1", "pose-stage1"}:
        task_config = "pose" if is_pose else "detect"
        stage = dict(config[task_config]["stage1"])
        candidates = [stage, {**stage, "batch": max(1, stage["batch"] // 2)}, {**stage, "batch": 1}]
        default_model = Path("yolov8n-pose.pt") if is_pose else PROJECT / "yolov8n.pt"
    elif args.command == "stage1280":
        stage = dict(config["detect"]["stage1280"])
        candidates = [stage] + [
            {**stage, **fallback} for fallback in config["detect"]["stage1280_oom_fallbacks"]
        ]
        default_model = PROJECT / "yolov8n.pt"
    elif args.command in {"stage2", "pose-stage2"}:
        task_config = "pose" if is_pose else "detect"
        stage = dict(config[task_config]["stage2"])
        candidates = [stage] + [
            {**stage, **fallback} for fallback in config[task_config]["stage2_oom_fallbacks"]
        ]
        default_model = _latest_stage1_weight(
            project, prefix="pose-stage1" if is_pose else "stage1"
        )
    else:
        stage = dict(config["segment"])
        fallback = stage.pop("oom_fallback")
        candidates = [stage, {**stage, **fallback}]
        default_model = Path("yolov8n-seg.pt")

    model_path = Path(args.model) if args.model else default_model
    name = args.name or f"{args.command}_{time.strftime('%Y%m%d_%H%M%S')}"
    payload = {
        "command": args.command,
        "model": str(model_path),
        "data": str(data_yaml),
        "project": str(project),
        "name": name,
        "common": common,
        "memory_fallbacks": candidates,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    _preflight()
    from ultralytics import YOLO

    last_error = None
    for attempt, candidate in enumerate(candidates, 1):
        run_name = name if attempt == 1 else f"{name}_fallback{attempt - 1}"
        parameters = {
            **common,
            **candidate,
            "data": str(data_yaml),
            "project": str(project),
            "name": run_name,
            "exist_ok": False,
            "save": True,
            "save_period": 10,
            "mixup": 0.0,
        }
        try:
            print(f"\n训练尝试 {attempt}/{len(candidates)}: imgsz={candidate['imgsz']} batch={candidate['batch']}")
            YOLO(str(model_path)).train(**parameters)
            print(f"训练完成: {project / run_name}")
            return 0
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_error = exc
            print(f"显存不足，清理CUDA缓存后尝试下一档: {exc}")
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    raise RuntimeError(f"所有6GB降级配置均显存不足: {last_error}")


def run_validation(args) -> int:
    from ultralytics import YOLO

    data_config = yaml.safe_load(args.data.resolve().read_text(encoding="utf-8"))
    if args.split == "test" and data_config.get("prototype_same_session_holdout"):
        raise ValueError("Prototype same-session holdout has no independent test split")
    _validate_data_yaml(args.data.resolve(), require_test=args.split == "test")
    metrics = YOLO(str(args.model.resolve())).val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=1,
        device=0,
        workers=0,
        plots=True,
    )
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    class_names = metrics.names
    per_class = {}
    for result_index, class_id in enumerate(class_indices):
        per_class[str(class_id)] = {
            "name": class_names[class_id],
            "precision": float(metrics.box.p[result_index]),
            "recall": float(metrics.box.r[result_index]),
            "map50": float(metrics.box.ap50[result_index]),
            "map50_95": float(metrics.box.maps[class_id]),
        }
    report = {
        "model": str(args.model.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
        "per_class": per_class,
    }
    output = PROJECT / "output" / f"validation_{args.model.stem}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {output}")
    return 0


def run_benchmark(args) -> int:
    import torch
    from ultralytics import YOLO

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data["path"])
    test_entry = Path(data.get("test", data["val"]))
    test_dir = test_entry if test_entry.is_absolute() else root / test_entry
    images = [
        path for path in sorted(test_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ][:args.limit]
    if not images:
        raise ValueError(f"没有基准图片: {test_dir}")
    model = YOLO(str(args.model.resolve()))
    torch.cuda.reset_peak_memory_stats()
    for path in images[:3]:
        model.predict(str(path), imgsz=args.imgsz, device=0, verbose=False)
    started = time.perf_counter()
    for path in images:
        model.predict(str(path), imgsz=args.imgsz, device=0, verbose=False)
    elapsed = time.perf_counter() - started
    report = {
        "model": str(args.model.resolve()),
        "images": len(images),
        "imgsz": args.imgsz,
        "fps": round(len(images) / elapsed, 2),
        "milliseconds_per_image": round(elapsed / len(images) * 1000, 2),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_export(args) -> int:
    from ultralytics import YOLO

    output = YOLO(str(args.model.resolve())).export(
        format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True, half=False
    )
    print(f"ONNX: {output}")
    return 0


def _preflight() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"正式训练要求Python 3.11，当前是 {platform.python_version()}。"
            "请使用 environment-6gb.yml 创建独立环境。"
        )
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝误用CPU开始长时间训练")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024 ** 3
    if total_gib > 7.0:
        print(f"提示: 检测到 {total_gib:.1f}GB 显存，仍沿用保守6GB配置")
    print(f"GPU: {properties.name}, VRAM: {total_gib:.1f}GB")


def _validate_data_yaml(path: Path, require_test: bool, task: str = "detect") -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("path", "train", "val", "names"):
        if key not in data:
            raise ValueError(f"data.yaml缺少字段: {key}")
    if (
        require_test
        and "test" not in data
        and not data.get("prototype_same_session_holdout", False)
    ):
        raise ValueError("正式训练数据必须包含独立test划分")
    class_count = len(data["names"])
    if task == "detect" and class_count != 18:
        raise ValueError(f"器具检测数据集应为固定18类，实际 {class_count} 类")
    if task == "segment" and class_count != 4:
        raise ValueError(f"分割数据集应为4类，实际 {class_count} 类")
    if task == "pose":
        if class_count != 4:
            raise ValueError(f"器具关键点数据集应为4类，实际 {class_count} 类")
        if data.get("kpt_shape") != [3, 3]:
            raise ValueError("器具关键点数据必须声明 kpt_shape: [3, 3]")
    if task == "detect" and (
        "active_class_ids" in data or "deferred_class_ids" in data
    ):
        if "active_class_ids" not in data or "deferred_class_ids" not in data:
            raise ValueError("阶段检测数据必须同时声明active_class_ids和deferred_class_ids")
        active = {int(value) for value in data["active_class_ids"]}
        deferred = {int(value) for value in data["deferred_class_ids"]}
        expected = set(range(18))
        if not active:
            raise ValueError("active_class_ids不能为空")
        if active & deferred:
            raise ValueError("active_class_ids和deferred_class_ids不能重叠")
        if active | deferred != expected:
            raise ValueError("active/deferred类别必须完整覆盖固定18类编号")


def _prepare_smoke_yaml(data_yaml: Path, output: Path, limit: int) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    train_entry = Path(data["train"])
    train_dir = train_entry if train_entry.is_absolute() else root / train_entry
    images = [
        path.resolve() for path in sorted(train_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ][:limit]
    if not images:
        raise ValueError(f"训练集为空: {train_dir}")
    output.mkdir(parents=True, exist_ok=True)
    train_list = output / "train.txt"
    val_list = output / "val.txt"
    train_list.write_text("\n".join(path.as_posix() for path in images) + "\n", encoding="utf-8")
    val_images = images[:min(20, len(images))]
    val_list.write_text("\n".join(path.as_posix() for path in val_images) + "\n", encoding="utf-8")
    smoke = {
        "path": root.as_posix(),
        "train": train_list.resolve().as_posix(),
        "val": val_list.resolve().as_posix(),
        "nc": len(data["names"]),
        "names": data["names"],
    }
    for key in ("kpt_shape", "flip_idx"):
        if key in data:
            smoke[key] = data[key]
    output_yaml = output / "data_smoke.yaml"
    output_yaml.write_text(yaml.safe_dump(smoke, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_yaml


def _latest_stage1_weight(project: Path, prefix: str = "stage1") -> Path:
    weights = sorted(
        project.glob(f"{prefix}_*/weights/best.pt"), key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not weights:
        raise FileNotFoundError("未找到stage1 best.pt，请先运行stage1或使用--model指定")
    return weights[0]


if __name__ == "__main__":
    raise SystemExit(main())
