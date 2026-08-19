"""
模型配置模块 — 统一管理 YOLO 权重路径与类别 Profile。

解决 9 类 / 13 类权重并存时的类别 ID 映射风险。
"""

from dataclasses import dataclass
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(PROJECT_ROOT, "config", "model_profiles.json")


@dataclass
class ModelProfile:
    """当前加载模型的类别配置。"""

    name: str
    display_name: str
    class_names: List[str]
    scoring_mode: str = "supported_classes_only"
    scoring_items: Optional[List[str]] = None
    item_aliases: Optional[Dict[str, str]] = None
    active_class_ids: Optional[List[int]] = None

    @property
    def supported_items(self) -> set:
        return set(self.scoring_items or self.class_names)

    @property
    def active_class_names(self) -> List[str]:
        if self.active_class_ids is None:
            return list(self.class_names)
        return [self.class_names[index] for index in self.active_class_ids]


@dataclass
class LoadedModelConfig:
    """YOLO 模型和其对应的类别配置。"""

    model_path: str
    yolo_model: object
    profile: ModelProfile
    model_names: List[str]


class ModelConfigError(RuntimeError):
    """模型类别配置错误。"""


_FALLBACK_CANDIDATES = [
    os.path.join(
        PROJECT_ROOT,
        "models",
        "low_vram",
        "front_detect_selected_holdout_stage1-2",
        "weights",
        "best.pt",
    ),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_final_20260723", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_final_20260722", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_office50_canister", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_train3", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_train2", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_best.pt"),
    os.path.join(PROJECT_ROOT, "models", "tea_ware_train", "weights", "best.pt"),
]

# Multiple 18-class profiles share the same class order but enable different
# classes at runtime. Bind known project weights explicitly so ``auto`` does
# not silently select the older tea18 profile and disable the kettle output.
_MODEL_PROFILE_OVERRIDES = {
    os.path.normcase(os.path.abspath(_FALLBACK_CANDIDATES[0])): "tea18_warm_clean",
}

_EN_TO_CN_CLASS = {
    "gaiwan": "盖碗",
    "pitcher": "公道杯",
    "cup": "品茗杯",
    "tea_lotus": "茶荷",
    "towel": "茶巾",
    "tongs": "茶夹",
    "pick": "茶拨",
    "tray": "茶盘",
    "kettle": "烧水壶",
    "waste_bowl": "建水",
    "scale": "电子秤",
    "thermometer": "温度计",
    "timer": "计时器",
    "盖碗（碗身）": "盖碗碗身",
    "盖碗（碗盖）": "盖碗碗盖",
}


def load_profiles(profile_path: Optional[str] = None) -> Dict[str, ModelProfile]:
    """读取 `config/model_profiles.json`。"""
    path = profile_path or PROFILE_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    profiles = {}
    for name, cfg in raw.items():
        class_names = list(cfg["class_names"])
        active_class_ids = cfg.get("active_class_ids")
        if active_class_ids is not None:
            active_class_ids = [int(value) for value in active_class_ids]
            invalid = [
                value for value in active_class_ids
                if value < 0 or value >= len(class_names)
            ]
            if invalid or len(set(active_class_ids)) != len(active_class_ids):
                raise ModelConfigError(
                    f"profile {name} 的 active_class_ids 无效: {active_class_ids}"
                )
        profiles[name] = ModelProfile(
            name=name,
            display_name=cfg.get("display_name", name),
            class_names=class_names,
            scoring_mode=cfg.get("scoring_mode", "supported_classes_only"),
            scoring_items=list(cfg.get("scoring_items", cfg["class_names"])),
            item_aliases=dict(cfg.get("item_aliases", {})),
            active_class_ids=active_class_ids,
        )
    return profiles


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """解析模型路径；优先使用显式路径，其次使用环境变量和默认候选。"""
    candidates = []
    if model_path:
        candidates.append(model_path)

    env_path = os.environ.get("TEA_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.extend(_FALLBACK_CANDIDATES)

    for candidate in candidates:
        if not candidate:
            continue
        abs_path = candidate
        if not os.path.isabs(abs_path):
            abs_path = os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(abs_path):
            return abs_path

    raise ModelConfigError(
        "未找到 YOLO 茶具模型，请使用 --model 指定权重路径。\n"
        + "已查找:\n  "
        + "\n  ".join(candidates)
    )


def profile_override_for_model(model_path: str) -> Optional[str]:
    """Return the runtime profile assigned to a known project weight."""
    normalized = os.path.normcase(os.path.abspath(model_path))
    return _MODEL_PROFILE_OVERRIDES.get(normalized)


def names_to_list(names: object) -> List[str]:
    """将 Ultralytics `model.names` 统一为按类别 ID 排序的 list。"""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise ModelConfigError(f"无法解析模型类别 names: {type(names)!r}")


def match_profile(
    model_names: Iterable[str],
    profiles: Optional[Dict[str, ModelProfile]] = None,
    requested_profile: str = "auto",
    strict: bool = True,
) -> ModelProfile:
    """根据模型类别名匹配 tea9 / tea13 profile。"""
    profiles = profiles or load_profiles()
    names = list(model_names)
    normalized_names = normalize_class_names(names)

    if requested_profile != "auto":
        if requested_profile not in profiles:
            raise ModelConfigError(f"未知 profile: {requested_profile}")
        profile = profiles[requested_profile]
        if normalized_names != profile.class_names:
            message = _format_mismatch(names, profile)
            if strict:
                raise ModelConfigError(message)
            print("[ModelConfig] 警告: " + message)
        return profile

    for profile in profiles.values():
        if normalized_names == profile.class_names:
            return profile

    if strict:
        expected = "\n".join(
            f"  - {p.name}: {len(p.class_names)}类 {p.class_names}"
            for p in profiles.values()
        )
        raise ModelConfigError(
            "模型类别与已知 tea9/tea13 profile 不匹配，已停止以避免类别错映射。\n"
            f"模型类别({len(names)}类): {names}\n"
            f"期望 profile:\n{expected}"
        )

    return ModelProfile(
        name="custom",
        display_name="自定义模型",
        class_names=normalized_names,
        scoring_mode="supported_classes_only",
        scoring_items=normalized_names,
        item_aliases={},
        active_class_ids=None,
    )


def normalize_class_names(names: Iterable[str]) -> List[str]:
    """将模型类别名规范化为项目使用的中文类别名。"""
    normalized = []
    for name in names:
        key = str(name).strip()
        normalized.append(_EN_TO_CN_CLASS.get(key, key))
    return normalized


def load_yolo_with_profile(
    model_path: Optional[str] = None,
    requested_profile: str = "auto",
    strict: bool = True,
) -> LoadedModelConfig:
    """加载 YOLO 模型并校验类别 profile。"""
    from ultralytics import YOLO

    resolved_path = resolve_model_path(model_path)
    yolo_model = YOLO(resolved_path)
    model_names = names_to_list(getattr(yolo_model, "names", []))
    normalized_names = normalize_class_names(model_names)
    effective_profile = requested_profile
    if requested_profile == "auto":
        effective_profile = profile_override_for_model(resolved_path) or "auto"
    profile = match_profile(
        model_names,
        requested_profile=effective_profile,
        strict=strict,
    )
    return LoadedModelConfig(
        model_path=resolved_path,
        yolo_model=yolo_model,
        profile=profile,
        model_names=normalized_names,
    )


def filter_items_config(items_config: dict, profile: ModelProfile) -> dict:
    """按当前模型支持类别过滤 tea_items 配置。"""
    if profile.scoring_mode == "full_tea_items":
        return dict(items_config)
    supported = profile.supported_items
    return {name: cfg for name, cfg in items_config.items() if name in supported}


def _format_mismatch(model_names: List[str], profile: ModelProfile) -> str:
    return (
        f"模型类别与 profile {profile.name} 不一致。\n"
        f"模型类别({len(model_names)}类): {model_names}\n"
        f"profile类别({len(profile.class_names)}类): {profile.class_names}"
    )
