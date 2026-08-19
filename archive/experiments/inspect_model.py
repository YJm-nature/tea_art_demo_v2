"""
查看 YOLOv8 .pt 模型文件的参数和结构。

用法:
    python scripts/inspect_model.py                          # 查看项目中所有 .pt 文件
    python scripts/inspect_model.py path/to/model.pt         # 查看指定文件
    python scripts/inspect_model.py -a                       # 查看所有 .pt + 显示每层详细结构
"""

import sys
import os
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch


def inspect_pt(filepath: str, verbose: bool = False) -> None:
    """检查一个 .pt 文件的结构和参数"""
    print("=" * 72)
    print(f"File: {filepath}")
    print("=" * 72)

    # 文件大小
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"文件大小: {size_mb:.2f} MB")

    # 加载
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    print(f"顶层类型: {type(data).__name__}")

    if not isinstance(data, dict):
        print(str(data)[:2000])
        print()
        return

    # ----- 顶层字典键 -----
    print(f"\n顶层键 ({len(data)} 个):")
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {type(v).__name__}, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict, keys={list(v.keys())[:10]}"
                  + ("..." if len(v) > 10 else ""))
        elif isinstance(v, list):
            print(f"  {k}: list, len={len(v)}")
        elif v is None:
            print(f"  {k}: None")
        elif isinstance(v, (int, float, str, bool)):
            s = str(v)
            if len(s) > 100:
                s = s[:100] + "..."
            print(f"  {k}: {type(v).__name__} = {s}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # ----- 元信息 -----
    print("\n[元信息]")
    for key in ["date", "version", "epoch", "best_fitness", "license"]:
        if key in data:
            print(f"  {key}: {data[key]}")

    # ----- 训练参数 -----
    if "train_args" in data and isinstance(data["train_args"], dict):
        ta = data["train_args"]
        print(f"\n[训练参数] (部分):")
        for k in ["task", "epochs", "batch", "imgsz", "patience", "lr0", "lrf",
                  "momentum", "weight_decay", "optimizer", "data"]:
            if k in ta:
                print(f"  {k}: {ta[k]}")

    # ----- 训练指标 -----
    if "train_metrics" in data and isinstance(data["train_metrics"], dict):
        tm = data["train_metrics"]
        print(f"\n[训练指标] (best fitness):")
        metric_keys = [k for k in tm if not k.startswith("val/")]
        for k in metric_keys:
            print(f"  {k}: {tm[k]}")
        val_keys = [k for k in tm if k.startswith("val/")]
        for k in val_keys:
            print(f"  {k}: {tm[k]}")

    # ----- 模型结构 -----
    model = None
    source = None
    for key in ["model", "ema"]:
        if key in data and data[key] is not None:
            model = data[key]
            source = key
            break

    if model is None:
        print("\n[警告] 未找到 model 或 ema 权重")
        print()
        return

    print(f"\n[模型权重来源]: data[\"{source}\"]")
    print(f"模型类名: {model.__class__.__name__}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")

    # YAML 配置
    if hasattr(model, "yaml"):
        y = model.yaml
        print(f"\n[YAML 配置]:")
        print(f"  输入通道: {y.get('ch', '?')}")
        print(f"  类别数 (nc): {y.get('nc', '?')}")
        print(f"  深度系数 (depth_multiple): {y.get('depth_multiple', '?')}")
        print(f"  宽度系数 (width_multiple): {y.get('width_multiple', '?')}")

    # 按层打印参数
    print(f"\n{'层号':<6} {'模块':<22} {'参数量':>10}  {'累计占比':>10}")
    print("-" * 60)
    cumulative = 0
    for idx, (name, child) in enumerate(model.model.named_children()):
        params = sum(p.numel() for p in child.parameters())
        cumulative += params
        pct = cumulative / total * 100 if total > 0 else 0
        print(f"{int(name):<6} {child.__class__.__name__:<22} {params:>10,}  {pct:>9.1f}%")

    # Detect 头详情
    last = model.model[-1]
    if hasattr(last, "nc"):
        print(f"\n[检测头 Detect]:")
        print(f"  类别数: {last.nc}")
        if hasattr(last, "reg_max"):
            print(f"  reg_max: {last.reg_max}")
        print(f"  参数量: {sum(p.numel() for p in last.parameters()):,}")
        if hasattr(last, "stride"):
            print(f"  检测步长: {last.stride}")

    # 逐层详细结构（verbose 模式）
    if verbose:
        print(f"\n[逐层详细结构]:")
        print(model)
        print()

    print()


def find_pt_files(root: str = ".") -> list:
    """查找项目下所有 .pt 文件"""
    root = Path(root)
    return sorted([str(p) for p in root.rglob("*.pt")])


def main():
    parser = argparse.ArgumentParser(
        description="查看 YOLOv8 .pt 模型文件的参数和结构"
    )
    parser.add_argument(
        "files", nargs="*",
        help="要查看的 .pt 文件路径（不指定则自动查找项目中的所有 .pt）"
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="显示每层的详细结构（verbose）"
    )
    args = parser.parse_args()

    targets = args.files if args.files else find_pt_files(".")
    if not targets:
        print("未找到任何 .pt 文件")
        return

    print(f"找到 {len(targets)} 个 .pt 文件\n")

    for path in targets:
        if not os.path.isfile(path):
            print(f"[警告] 文件不存在: {path}\n")
            continue
        try:
            inspect_pt(path, verbose=args.all)
        except Exception as e:
            print(f"[错误] 加载失败: {e}\n")


if __name__ == "__main__":
    main()
