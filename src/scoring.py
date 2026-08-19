"""可审计的茶艺 SOP 评分规则。

步骤一的模型得分与完整 SOP 验收分是两个概念。本模块同时输出：
- observable score: 当前模型可观测项目内的表现分；
- requirement coverage: 需求中必备项目被模型覆盖的比例；
- coverage-adjusted score: 将尚不可观测项目按未完成处理的保守参考分；
- evidence reliability: 检测置信度与跨帧稳定性，只描述证据质量，不给学员加分。
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ScoreReport:
    """单个步骤的机器评分报告。"""

    step_id: int
    step_name: str
    detected_essential: int
    total_essential: int
    detected_optional: int
    total_optional: int
    score: float
    grade: str
    grade_color: str
    checklist: Dict[str, dict] = field(default_factory=dict)
    dimension_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    detail: str = ""
    requirement_coverage: float = 1.0
    coverage_adjusted_score: float = 0.0
    evidence_reliability: float = 0.0
    score_status: str = "final"
    supported_requirements: List[str] = field(default_factory=list)
    unsupported_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["schema_version"] = "1.0"
        data["timestamp_iso"] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()
        return data

    def save_json(self, path: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output


class ScoringEngine:
    """步骤评分引擎。权重为 demo 暂定口径，须在正式验收前由教师确认。"""

    STEP_NAMES = {
        1: "备具布席",
        2: "温杯洁具",
        3: "投茶准备",
        4: "投茶闻香",
        5: "注水冲泡",
        6: "分茶与奉茶",
    }

    PREPARATION_REQUIRED = [
        "盖碗", "公道杯", "品茗杯", "茶荷", "茶巾",
        "茶夹", "茶拨", "茶盘", "烧水壶", "建水",
    ]
    PREPARATION_OPTIONAL = ["电子秤", "温度计", "计时器"]
    PREPARATION_WEIGHTS = {
        "object_completeness": 0.80,
        "quantity_correctness": 0.10,
        "placement_heuristic": 0.10,
    }
    MIN_EVIDENCE_FRAMES = 15

    # 旧调用方兼容常量。该三维公式不再用于步骤一主流程。
    WEIGHTS = {
        "item_completeness": 0.40,
        "placement_reasonableness": 0.30,
        "area_normality": 0.30,
    }

    @classmethod
    def evaluate_preparation_step(
        cls,
        checklist: Dict[str, dict],
        supported_items: Iterable[str],
        placement_score: Optional[float] = None,
    ) -> ScoreReport:
        """评估步骤一，并显式披露模型覆盖率和证据可靠度。"""
        supported_set = set(supported_items)
        supported_required = [
            name for name in cls.PREPARATION_REQUIRED if name in supported_set
        ]
        unsupported_required = [
            name for name in cls.PREPARATION_REQUIRED if name not in supported_set
        ]
        supported_optional = [
            name for name in cls.PREPARATION_OPTIONAL if name in supported_set
        ]

        present_count = sum(
            1 for name in supported_required
            if cls._is_present(checklist.get(name, {}))
        )
        quantity_ok_count = sum(
            1 for name in supported_required
            if checklist.get(name, {}).get("detected", False)
        )
        optional_count = sum(
            1 for name in supported_optional
            if cls._is_present(checklist.get(name, {}))
        )

        required_count = len(supported_required)
        object_score = cls._ratio_score(present_count, required_count)
        quantity_score = cls._ratio_score(quantity_ok_count, required_count)
        placement = cls._clamp01(placement_score) * 100 if placement_score is not None else None

        dimensions: Dict[str, Optional[float]] = {
            "object_completeness": round(object_score, 1),
            "quantity_correctness": round(quantity_score, 1),
            "placement_heuristic": round(placement, 1) if placement is not None else None,
        }
        available = {
            "object_completeness": object_score,
            "quantity_correctness": quantity_score,
        }
        if placement is not None:
            available["placement_heuristic"] = placement

        # 缺少某个技术维度时对剩余权重归一化，不能用中性分伪造证据。
        weight_sum = sum(cls.PREPARATION_WEIGHTS[name] for name in available)
        observable_score = sum(
            value * cls.PREPARATION_WEIGHTS[name] for name, value in available.items()
        ) / weight_sum

        coverage = required_count / len(cls.PREPARATION_REQUIRED)
        coverage_adjusted = observable_score * coverage
        reliability = cls._evidence_reliability(checklist, supported_required)
        grade, color = cls._determine_grade_from_score(observable_score)
        status = "final" if coverage >= 1.0 else "provisional"

        return ScoreReport(
            step_id=1,
            step_name=cls.STEP_NAMES[1],
            detected_essential=quantity_ok_count,
            total_essential=required_count,
            detected_optional=optional_count,
            total_optional=len(supported_optional),
            score=round(observable_score, 1),
            grade=grade,
            grade_color=color,
            checklist=checklist,
            dimension_scores=dimensions,
            requirement_coverage=round(coverage, 3),
            coverage_adjusted_score=round(coverage_adjusted, 1),
            evidence_reliability=round(reliability, 1),
            score_status=status,
            supported_requirements=supported_required,
            unsupported_requirements=unsupported_required,
            detail=(
                f"当前模型必备品 {quantity_ok_count}/{required_count}，"
                f"需求覆盖 {required_count}/{len(cls.PREPARATION_REQUIRED)}；"
                "布局分为单机位启发式结果，未标定实际1.5米操作半径"
            ),
        )

    @classmethod
    def evaluate_step(
        cls,
        step_id: int,
        checklist: Dict[str, dict],
        essential_items: list,
        optional_items: list,
        placement_score: float = None,
        normality_score: float = None,
    ) -> ScoreReport:
        """兼容旧入口的通用物品评分。新代码优先调用具体步骤规则。"""
        detected_essential = sum(
            1 for name in essential_items
            if checklist.get(name, {}).get("detected", False)
        )
        detected_optional = sum(
            1 for name in optional_items
            if checklist.get(name, {}).get("detected", False)
        )
        item_score = cls._ratio_score(detected_essential, len(essential_items))
        dimensions: Dict[str, Optional[float]] = {
            "item_completeness": round(item_score, 1),
            "placement": None,
            "normality": None,
        }
        score = item_score
        if placement_score is not None and normality_score is not None:
            placement = cls._clamp01(placement_score) * 100
            normality = cls._clamp01(normality_score) * 100
            dimensions.update(placement=round(placement, 1), normality=round(normality, 1))
            score = (
                item_score * cls.WEIGHTS["item_completeness"]
                + placement * cls.WEIGHTS["placement_reasonableness"]
                + normality * cls.WEIGHTS["area_normality"]
            )
        grade, color = cls._determine_grade_from_score(score)
        return ScoreReport(
            step_id=step_id,
            step_name=cls.STEP_NAMES.get(step_id, f"步骤{step_id}"),
            detected_essential=detected_essential,
            total_essential=len(essential_items),
            detected_optional=detected_optional,
            total_optional=len(optional_items),
            score=round(score, 1),
            grade=grade,
            grade_color=color,
            checklist=checklist,
            dimension_scores=dimensions,
            coverage_adjusted_score=round(score, 1),
            detail=f"必备品: {detected_essential}/{len(essential_items)}",
        )

    @classmethod
    def _evidence_reliability(cls, checklist: Dict[str, dict], names: List[str]) -> float:
        if not names:
            return 0.0
        evidence = []
        for name in names:
            item = checklist.get(name, {})
            if not cls._is_present(item):
                evidence.append(0.0)
                continue
            confidence = cls._clamp01(item.get("confidence", 0.0))
            frames = max(0, int(item.get("seen_frames", 0)))
            stability = min(1.0, frames / cls.MIN_EVIDENCE_FRAMES)
            evidence.append(confidence * stability)
        return sum(evidence) / len(evidence) * 100

    @staticmethod
    def _is_present(item: dict) -> bool:
        if "present" in item:
            return bool(item["present"])
        return item.get("count", 0) > 0 or item.get("detected", False)

    @staticmethod
    def _ratio_score(found: int, total: int) -> float:
        return found / total * 100 if total > 0 else 0.0

    @staticmethod
    def _clamp01(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _determine_grade(cls, found: int, total: int) -> Tuple[str, str]:
        score = cls._ratio_score(found, total) if total else 100.0
        return cls._determine_grade_from_score(score)

    @staticmethod
    def _determine_grade_from_score(score: float) -> Tuple[str, str]:
        if score >= 90:
            return "A · 优秀", "#27ae60"
        if score >= 80:
            return "B · 良好", "#2ecc71"
        if score >= 60:
            return "C · 合格", "#f39c12"
        if score >= 40:
            return "D · 不合格", "#e74c3c"
        return "E · 严重不足", "#c0392b"

    @classmethod
    def compute_total_score(cls, step_reports: list) -> float:
        if not step_reports:
            return 0.0
        return sum(r.score for r in step_reports) / len(step_reports)
