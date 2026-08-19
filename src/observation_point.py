"""
观测点框架 — 茶艺红茶SOP自动评分系统

定义观测点抽象基类及全部19个观测点的数据结构。
当前已实现 2 个观测点：
  - obj_utensils_s1:   备具布席·物品准备检测 (Step 1)
  - result_scale_3g:   电子秤显示茶叶3g (Step 3)

其余 17 个观测点在 OBSERVATION_REGISTRY 中以注释占位，后续 Phase 逐个实现。
"""

from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import time
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 枚举定义
# ══════════════════════════════════════════════════════════════════════

class ObservationType(Enum):
    """观测点类型 — 对应需求文档中的4类判断"""
    OBJECT = "物品判断"       # 如：检测盖碗、公道杯等物品
    ACTION = "动作判断"       # 如：双手托举、冲洗盖碗
    SEQUENCE = "时序判断"     # 如：等待10秒、从左到右倒茶
    RESULT = "结果判断"       # 如：电子秤3g、桌面无溅水


class Verdict(Enum):
    """判定结果"""
    PASS = "✅ 合格"
    FAIL = "❌ 不合格"
    WARNING = "⚠️ 警告"      # 检测异常（如物品未找到、读数不可识别）


# ══════════════════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ObservationResult:
    """单个观测点的检测结果"""
    point_id: str                          # 观测点唯一标识
    point_name: str                        # 观测点名称（中文）
    sop_step: int                          # 所属SOP步骤编号 (1-6)
    obs_type: ObservationType              # 观测类型
    verdict: Verdict                       # 判定结果
    value: Any = None                      # 检测到的实际值（如 3.0g, 8/10）
    threshold: str = ""                    # 判定阈值描述
    confidence: float = 0.0                # 置信度 0-1
    detail: str = ""                       # 详细描述
    bbox: Optional[List[float]] = None     # 检测框 [x1, y1, x2, y2]
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展数据（如物品清单）
    timestamp: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════
# 观测点抽象基类
# ══════════════════════════════════════════════════════════════════════

class ObservationPoint(ABC):
    """
    观测点基类 — 所有19个观测点继承此类。

    设计原则：
    - 每个观测点封装自己的判定逻辑
    - detect() 接收帧+上下文，返回判定结果
    - 检测器/匹配器等重资源由 app 层管理，通过 context 传入结果
    - 可独立测试、独立部署
    """

    def __init__(
        self,
        point_id: str,
        point_name: str,
        sop_step: int,
        obs_type: ObservationType,
    ):
        self.point_id = point_id
        self.point_name = point_name
        self.sop_step = sop_step
        self.obs_type = obs_type

    @abstractmethod
    def detect(self, frame: np.ndarray, context: Dict[str, Any]) -> ObservationResult:
        """
        执行检测并返回判定结果。

        Args:
            frame: 当前视频帧 (BGR格式, numpy array)
            context: 上下文信息，由 pipeline 注入，包含：
                - 预检测结果（物品清单、OCR读数等）
                - current_step: 当前SOP步骤
                - step_start_time: 当前步骤开始时间
                - detected_objects: 已检测到的物品列表
                - previous_results: 历史检测结果

        Returns:
            ObservationResult: 检测与判定结果
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.point_name} (Step {self.sop_step})>"


# ══════════════════════════════════════════════════════════════════════
# ✅ 已实现：备具布席 · 物品准备检测 (Step 1)
# ══════════════════════════════════════════════════════════════════════

class UtensilsCheckObservation(ObservationPoint):
    """
    观测点：备具布席 — 物品准备检测

    所属步骤：步骤1 — 备具布席
    判定标准：10项必备品全部检出为合格（A级），≥8项为良好（B级），≥6项为合格（C级）

    技术链路：
    TeaDetector(YOLO+HSV) → ItemMatcher(几何模板匹配) → DetectionMemory(跨帧累积) → 本观测点评定

    context 期望字段：
        - checklist: Dict — ItemMatcher/DetectionMemory 生成的物品清单
        - essential_found: int — 检出的必备品数量
        - total_essential: int — 必备品总数
        - score: float — 得分 (0-100)
        - grade: str — 等级标签
        - grade_color: str — 等级颜色

    纯逻辑测试：
        UtensilsCheckObservation.evaluate(essential_found=8, total_essential=10)
    """

    def __init__(self):
        super().__init__(
            point_id="obj_utensils_s1",
            point_name="备具布席·物品准备检测",
            sop_step=1,
            obs_type=ObservationType.OBJECT,
        )

    def detect(self, frame: np.ndarray, context: Dict[str, Any]) -> ObservationResult:
        """从 context 中获取预检测结果并做出判定"""
        checklist = context.get("checklist", {})
        essential_found = context.get("essential_found", 0)
        total_essential = context.get("total_essential", 10)
        score = context.get("score", 0.0)
        grade = context.get("grade", "-")

        # 判定
        verdict = self.evaluate(essential_found, total_essential)

        # 构造详情
        missing_items = [
            info.get("name_cn", name)
            for name, info in checklist.items()
            if info.get("essential") and not info.get("detected")
        ]
        found_items = [
            info.get("name_cn", name)
            for name, info in checklist.items()
            if info.get("essential") and info.get("detected")
        ]

        if missing_items:
            detail = (
                f"必备品检出 {essential_found}/{total_essential}，"
                f"缺失: {', '.join(missing_items)}"
            )
        else:
            detail = f"全部必备品已备齐 ({essential_found}/{total_essential})"

        return ObservationResult(
            point_id=self.point_id,
            point_name=self.point_name,
            sop_step=self.sop_step,
            obs_type=self.obs_type,
            verdict=verdict,
            value=f"{essential_found}/{total_essential}",
            threshold="10/10 必备品 → A级",
            confidence=essential_found / max(total_essential, 1),
            detail=detail,
            metadata={
                "checklist": checklist,
                "found_items": found_items,
                "missing_items": missing_items,
                "score": score,
                "grade": grade,
            },
        )

    @classmethod
    def evaluate(cls, essential_found: int, total_essential: int = 10) -> Verdict:
        """
        纯数据判定（不依赖图像上下文，便于单元测试）。

        Args:
            essential_found: 检出的必备品数量
            total_essential: 必备品总数

        Returns:
            Verdict: PASS (≥6), FAIL (<6)
        """
        if total_essential == 0:
            return Verdict.PASS
        ratio = essential_found / total_essential
        if ratio >= 0.6:
            return Verdict.PASS
        elif ratio >= 0.4:
            return Verdict.FAIL
        else:
            return Verdict.FAIL


# ══════════════════════════════════════════════════════════════════════
# ✅ 已实现：电子秤显示茶叶3g (Step 3)
# ══════════════════════════════════════════════════════════════════════

class ScaleReadingObservation(ObservationPoint):
    """
    观测点：电子秤显示茶叶3g

    所属步骤：步骤3 — 投茶准备
    判定标准：电子秤显示数值 ∈ [2.7, 3.3]g

    技术链路：
    YOLO检测电子秤 → 裁切屏幕ROI → PaddleOCR读取数字 → 数值判定

    context 期望字段：
        - scale_bbox: List[float] or None — 电子秤检测框
        - scale_ocr_value: float or None — OCR读取值
        - scale_ocr_confidence: float — OCR置信度

    纯逻辑测试：
        ScaleReadingObservation.evaluate(3.0) → PASS
        ScaleReadingObservation.evaluate(2.3) → FAIL
        ScaleReadingObservation.evaluate(None) → WARNING
    """

    TARGET_VALUE = 3.0       # 目标值（克）
    TOLERANCE = 0.3          # 容差范围 ±0.3g

    def __init__(self):
        super().__init__(
            point_id="result_scale_3g",
            point_name="电子秤显示茶叶3g",
            sop_step=3,
            obs_type=ObservationType.RESULT,
        )

    def detect(self, frame: np.ndarray, context: Dict[str, Any]) -> ObservationResult:
        """从 context 中获取 OCR 读数并做出判定"""
        scale_bbox = context.get("scale_bbox", None)
        ocr_value = context.get("scale_ocr_value", None)
        ocr_conf = context.get("scale_ocr_confidence", 0.0)

        # 未检测到电子秤
        if scale_bbox is None:
            return ObservationResult(
                point_id=self.point_id,
                point_name=self.point_name,
                sop_step=self.sop_step,
                obs_type=self.obs_type,
                verdict=Verdict.WARNING,
                value=None,
                threshold=f"{self.TARGET_VALUE}±{self.TOLERANCE}g",
                confidence=0.0,
                detail="未检测到电子秤，请确认电子秤在摄像头视野内",
            )

        # 电子秤屏幕无法识别
        if ocr_value is None:
            return ObservationResult(
                point_id=self.point_id,
                point_name=self.point_name,
                sop_step=self.sop_step,
                obs_type=self.obs_type,
                verdict=Verdict.WARNING,
                value=None,
                threshold=f"{self.TARGET_VALUE}±{self.TOLERANCE}g",
                confidence=ocr_conf,
                detail="检测到电子秤但无法读取屏幕数值，请调整光线或角度",
                bbox=scale_bbox,
            )

        # 数值判定
        diff = abs(ocr_value - self.TARGET_VALUE)
        if diff <= self.TOLERANCE:
            verdict = Verdict.PASS
            detail = f"茶叶重量 {ocr_value:.1f}g，在目标范围 {self.TARGET_VALUE}±{self.TOLERANCE}g 内"
        else:
            verdict = Verdict.FAIL
            direction = "偏高" if ocr_value > self.TARGET_VALUE else "偏低"
            detail = f"茶叶重量 {ocr_value:.1f}g，{direction}（目标 {self.TARGET_VALUE}±{self.TOLERANCE}g）"

        return ObservationResult(
            point_id=self.point_id,
            point_name=self.point_name,
            sop_step=self.sop_step,
            obs_type=self.obs_type,
            verdict=verdict,
            value=ocr_value,
            threshold=f"{self.TARGET_VALUE}±{self.TOLERANCE}g",
            confidence=ocr_conf,
            detail=detail,
            bbox=scale_bbox,
            metadata={"raw_diff": diff, "target": self.TARGET_VALUE},
        )

    @classmethod
    def evaluate(cls, value: Optional[float]) -> Verdict:
        """纯数值判定（不依赖图像上下文，便于单元测试）"""
        if value is None:
            return Verdict.WARNING
        if abs(value - cls.TARGET_VALUE) <= cls.TOLERANCE:
            return Verdict.PASS
        return Verdict.FAIL


# ══════════════════════════════════════════════════════════════════════
# 观测点注册表（完整19个观测点 — 已实现2个 + 占位17个）
# ══════════════════════════════════════════════════════════════════════

OBSERVATION_REGISTRY: Dict[str, ObservationPoint] = {
    # ── Step 1: 备具布席 ──
    "obj_utensils_s1": UtensilsCheckObservation(),     # ✅ 物品准备检测

    # ── Step 2: 温杯洁具 ──
    # TODO: "obj_utensils_s2": UtensilsCheckObservation(step=2, ...),
    # TODO: "result_temp_s2": TemperatureObservation(),
    # TODO: "seq_warm_order": WarmCupSequenceObservation(),

    # ── Step 3: 投茶准备 ──
    "result_scale_3g": ScaleReadingObservation(),       # ✅ 电子秤显示茶叶3g
    # TODO: "obj_utensils_s3": UtensilsCheckObservation(step=3, ...),
    # TODO: "action_hold_lotus": HandHoldObservation(),

    # ── Step 4: 投茶闻香 ──
    # TODO: "obj_utensils_s4": UtensilsCheckObservation(step=4, ...),
    # TODO: "action_tilt_lotus": AngleObservation("茶荷15°", 15, (10, 30)),
    # TODO: "action_open_lid": LidOpenSmellObservation(),
    # TODO: "result_no_spill_s4": NoSpillObservation(),

    # ── Step 5: 注水冲泡 ──
    # TODO: "obj_utensils_s5": UtensilsCheckObservation(step=5, ...),
    # TODO: "action_rinse_gaiwan": RinseGaiwanObservation(),
    # TODO: "seq_wait_10s": WaitTimerObservation(),
    # TODO: "result_water_submerge": WaterSubmergeObservation(),
    # TODO: "result_no_splash": NoSplashObservation(),

    # ── Step 6: 分茶与奉茶 ──
    # TODO: "obj_utensils_s6": UtensilsCheckObservation(step=6, ...),
    # TODO: "seq_pour_order": PourOrderObservation(),
    # TODO: "action_hold_tray": HandHoldTrayObservation(),
    # TODO: "result_cup_layout": CupLayoutObservation(),
    # TODO: "result_no_spill_s6": NoSpillObservation(),
}


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def get_observations_for_step(sop_step: int) -> List[ObservationPoint]:
    """获取指定SOP步骤的所有已注册观测点"""
    return [obs for obs in OBSERVATION_REGISTRY.values() if obs.sop_step == sop_step]


def get_all_observation_ids() -> List[str]:
    """获取所有已注册的观测点ID"""
    return list(OBSERVATION_REGISTRY.keys())


def get_registered_steps() -> Dict[int, List[str]]:
    """获取各步骤的观测点覆盖情况

    Returns:
        {step_number: [point_id, ...]}
    """
    steps: Dict[int, List[str]] = {}
    for pid, obs in OBSERVATION_REGISTRY.items():
        steps.setdefault(obs.sop_step, []).append(pid)
    return dict(sorted(steps.items()))


def run_pure_logic_tests() -> bool:
    """
    运行所有已实现观测点的纯逻辑测试（不依赖任何AI模型）。

    Returns:
        True 如果全部通过
    """
    all_pass = True

    # ── UtensilsCheckObservation 测试 ──
    print("\n── UtensilsCheckObservation ──")
    test_cases_utensils = [
        (10, 10, Verdict.PASS, "全部备齐"),
        (8, 10, Verdict.PASS, "8/10 良好"),
        (6, 10, Verdict.PASS, "6/10 合格边缘"),
        (5, 10, Verdict.FAIL, "5/10 不合格"),
        (3, 10, Verdict.FAIL, "3/10 严重不足"),
        (0, 10, Verdict.FAIL, "0/10 全未检出"),
    ]
    for found, total, expected, desc in test_cases_utensils:
        actual = UtensilsCheckObservation.evaluate(found, total)
        ok = "[PASS]" if actual == expected else "[FAIL]"
        if actual != expected:
            all_pass = False
        print(f"  {ok} {desc:20s} found={found}/{total}  expected={expected.value:10s}  got={actual.value}")

    # ── ScaleReadingObservation 测试 ──
    print("\n── ScaleReadingObservation ──")
    test_cases_scale = [
        (3.0, Verdict.PASS, "Exactly 3.0g"),
        (2.7, Verdict.PASS, "Lower bound 2.7g"),
        (3.3, Verdict.PASS, "Upper bound 3.3g"),
        (2.6, Verdict.FAIL, "Below tolerance 2.6g"),
        (3.4, Verdict.FAIL, "Above tolerance 3.4g"),
        (0.0, Verdict.FAIL, "Zero reading"),
        (None, Verdict.WARNING, "No reading"),
    ]
    for value, expected, desc in test_cases_scale:
        actual = ScaleReadingObservation.evaluate(value)
        ok = "[PASS]" if actual == expected else "[FAIL]"
        if actual != expected:
            all_pass = False
        print(f"  {ok} {desc:30s} value={str(value):8s}  expected={expected.value:10s}  got={actual.value}")

    if all_pass:
        print("\n  [OK] All pure logic tests PASSED!")
    else:
        print("\n  [FAIL] Some tests FAILED!")

    return all_pass


# ══════════════════════════════════════════════════════════════════════
# 模块自检（直接运行此文件时执行纯逻辑测试）
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("观测点框架 — 纯逻辑测试")
    print(f"已注册观测点: {len(OBSERVATION_REGISTRY)} 个")
    for pid, obs in OBSERVATION_REGISTRY.items():
        print(f"  {pid:25s} → {obs.point_name}")
    print(f"\n各步骤覆盖: {get_registered_steps()}")
    print("=" * 60)
    run_pure_logic_tests()
