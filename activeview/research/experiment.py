"""Experiment identity and lifecycle model for controlled research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class ExperimentStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    FINAL_FROZEN = "FINAL_FROZEN"


class Decision(str, Enum):
    NA = "NA"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


def _enum_value(value: Enum | str, enum_type: type[Enum], field_name: str) -> str:
    candidate = value.value if isinstance(value, enum_type) else str(value)
    if candidate not in {item.value for item in enum_type}:
        raise ValueError(f"Invalid {field_name}: {candidate}")
    return candidate


@dataclass
class Experiment:
    """Serializable experiment record with explicit scientific fields."""

    experiment_id: str
    name: str
    stage: str
    status: ExperimentStatus | str = ExperimentStatus.PLANNED
    hypothesis: str = ""
    motivation: str = ""
    baseline: str = ""
    core_change: str = ""
    frozen_items: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    rejection_criteria: list[str] = field(default_factory=list)
    source_dir: str = ""
    runtime_dir: str = ""
    created_at: str = ""
    completed_at: str | None = None
    decision: Decision | str = Decision.NA

    def validate(self) -> None:
        """Raise ``ValueError`` when identity or enum fields are invalid."""
        if not self.experiment_id.startswith("EXP") or not self.experiment_id[3:].isdigit():
            raise ValueError(f"Invalid experiment_id: {self.experiment_id}")
        for field_name in ("name", "stage", "source_dir", "runtime_dir", "created_at"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Missing required field: {field_name}")
        _enum_value(self.status, ExperimentStatus, "status")
        _enum_value(self.decision, Decision, "decision")
        if not isinstance(self.frozen_items, list) or not isinstance(self.metrics, list):
            raise ValueError("frozen_items and metrics must be lists")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["status"] = _enum_value(self.status, ExperimentStatus, "status")
        payload["decision"] = _enum_value(self.decision, Decision, "decision")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Experiment":
        required = {"experiment_id", "name", "stage", "status", "hypothesis", "motivation", "baseline", "core_change", "frozen_items", "metrics", "acceptance_criteria", "rejection_criteria", "source_dir", "runtime_dir", "created_at", "completed_at", "decision"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Missing experiment fields: {sorted(missing)}")
        instance = cls(**{key: payload[key] for key in required})
        instance.validate()
        return instance
