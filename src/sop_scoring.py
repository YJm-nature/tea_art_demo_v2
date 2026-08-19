"""Provisional scoring-data ledger built from observation events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class CriterionEvidence:
    observation_id: str
    status: str
    confidence: float
    value: Any = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    rule_version: str = "unknown"
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
            "value": self.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "model_version": self.model_version,
            "rule_version": self.rule_version,
            "event_id": self.event_id,
        }


class SopScoreLedger:
    """Store auditable pass/fail evidence without claiming a formal score."""

    TERMINAL_PHASES = {"completed", "failed", "uncertain"}

    def __init__(self, weights: Optional[Mapping[str, float]] = None):
        self.weights = {str(key): float(value) for key, value in (weights or {}).items()}
        self.criteria: Dict[str, CriterionEvidence] = {}

    @staticmethod
    def _read(event: Any, key: str, default: Any = None) -> Any:
        if isinstance(event, Mapping):
            return event.get(key, default)
        return getattr(event, key, default)

    def consume(self, events: Iterable[Any]) -> None:
        for event in events:
            phase = self._read(event, "phase", "")
            phase = str(getattr(phase, "value", phase))
            if phase not in self.TERMINAL_PHASES:
                continue
            observation_id = str(self._read(event, "observation_id", ""))
            if not observation_id:
                continue
            self.criteria[observation_id] = CriterionEvidence(
                observation_id=observation_id,
                status=phase,
                confidence=float(self._read(event, "confidence", 0.0)),
                value=self._read(event, "value"),
                start_time=self._read(event, "start_time"),
                end_time=self._read(event, "end_time"),
                metrics=dict(self._read(event, "metrics", {}) or {}),
                model_version=str(self._read(event, "model_version", "unknown")),
                rule_version=str(self._read(event, "rule_version", "unknown")),
                event_id=self._read(event, "event_id"),
            )

    def to_dict(self) -> Dict[str, Any]:
        passed = sum(row.status == "completed" for row in self.criteria.values())
        failed = sum(row.status == "failed" for row in self.criteria.values())
        uncertain = sum(row.status == "uncertain" for row in self.criteria.values())
        weighted_score = None
        covered_weight = sum(
            self.weights.get(key, 0.0) for key in self.criteria
            if self.criteria[key].status in {"completed", "failed"}
        )
        if self.weights and covered_weight > 0:
            earned = sum(
                self.weights.get(key, 0.0)
                for key, row in self.criteria.items()
                if row.status == "completed"
            )
            weighted_score = round(earned / covered_weight * 100.0, 2)
        return {
            "schema_version": "1.0",
            "score_status": "provisional" if self.weights else "evidence_only",
            "formal_scoring_enabled": False,
            "weighted_score": weighted_score,
            "summary": {
                "observed_criteria": len(self.criteria),
                "passed": passed,
                "failed": failed,
                "uncertain": uncertain,
            },
            "criteria": {
                key: value.to_dict() for key, value in self.criteria.items()
            },
        }

    def reset(self) -> None:
        self.criteria.clear()

