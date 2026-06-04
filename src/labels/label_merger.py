from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.labels.judge_schema import JudgeLabel
from src.labels.label_validation import validation_errors, validate_judge_label


LabelSourceName = Literal["benchmark_metadata", "oracle_diagnostic", "llm_judge"]


@dataclass(frozen=True)
class SourceLabel:
    source: LabelSourceName
    binary_risk_label: int | None = None
    severity_score: float | None = None
    first_risk_step: int | None = None
    risk_category: str | None = None
    judge_version: str | None = None
    residual_concern: bool = False
    raw: dict[str, Any] | None = None


def _normalize_binary(value: Any) -> int | None:
    if value in {0, 1}:
        return int(value)
    if value is True:
        return 1
    if value is False:
        return 0
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "risky", "risk"}:
        return 1
    if text in {"0", "false", "safe", "none"}:
        return 0
    return None


def source_label_from_judge(payload: dict[str, Any]) -> tuple[SourceLabel | None, list[str]]:
    errors = validation_errors(payload)
    if errors:
        return None, errors
    label = validate_judge_label(payload)
    return (
        SourceLabel(
            source="llm_judge",
            binary_risk_label=label.binary_risk_label,
            severity_score=label.severity_score,
            first_risk_step=label.first_risk_step,
            risk_category=label.risk_category,
            judge_version=label.judge_version,
            residual_concern=label.residual_concern,
            raw=payload,
        ),
        [],
    )


def source_label_from_metadata(payload: dict[str, Any]) -> SourceLabel:
    binary = _normalize_binary(payload.get("binary_risk_label", payload.get("risk_event")))
    first_risk_step = payload.get("first_risk_step", payload.get("t_risk"))
    if first_risk_step is not None:
        first_risk_step = int(first_risk_step)
    severity = payload.get("severity_score", payload.get("severity"))
    if severity is not None:
        severity = float(severity)
    return SourceLabel(
        source="benchmark_metadata",
        binary_risk_label=binary,
        severity_score=severity,
        first_risk_step=first_risk_step,
        risk_category=payload.get("risk_category"),
        raw=payload,
    )


def source_label_from_oracle(payload: dict[str, Any]) -> SourceLabel:
    binary = _normalize_binary(payload.get("oracle_violation"))
    return SourceLabel(
        source="oracle_diagnostic",
        binary_risk_label=binary,
        severity_score=float(payload.get("severity_score", 0.0)) if binary else 0.0,
        first_risk_step=int(payload["first_risk_step"]) if payload.get("first_risk_step") is not None else None,
        risk_category=payload.get("risk_category", "policy_violation" if binary else "none"),
        raw=payload,
    )


def _conflicts(metadata: SourceLabel | None, judge: SourceLabel | None, include_oracle: bool) -> list[str]:
    conflicts: list[str] = []
    if judge is not None:
        if judge.binary_risk_label == 0 and (judge.severity_score or 0.0) > 0 and not judge.residual_concern:
            conflicts.append("binary_risk_label=0 but severity_score>0 without residual concern policy")
        if judge.binary_risk_label == 1 and judge.first_risk_step is None:
            conflicts.append("risky label but missing first_risk_step")
    if metadata is not None and judge is not None:
        if judge.binary_risk_label == 1 and metadata.binary_risk_label == 0:
            conflicts.append("judge risky but metadata safe")
        if metadata.binary_risk_label == 1 and judge.binary_risk_label == 0:
            conflicts.append("metadata risky but judge safe")
    if include_oracle:
        conflicts.append("oracle labels are diagnostic by default; explicit inclusion requested")
    return conflicts


def merge_label_sources(
    *,
    episode_id: str,
    benchmark_metadata: dict[str, Any] | None = None,
    oracle_diagnostic: dict[str, Any] | None = None,
    llm_judge: dict[str, Any] | None = None,
    suffix_policy: str = "future_only",
    allow_oracle_as_label: bool = False,
) -> dict[str, Any]:
    metadata_label = source_label_from_metadata(benchmark_metadata or {}) if benchmark_metadata is not None else None
    judge_label = None
    judge_errors: list[str] = []
    if llm_judge is not None:
        judge_label, judge_errors = source_label_from_judge(llm_judge)
    oracle_label = source_label_from_oracle(oracle_diagnostic or {}) if oracle_diagnostic is not None else None

    selected = judge_label or metadata_label
    if selected is None and allow_oracle_as_label:
        selected = oracle_label

    conflicts = list(judge_errors)
    conflicts.extend(_conflicts(metadata_label, judge_label, include_oracle=allow_oracle_as_label and oracle_label is not None))
    if selected is None:
        conflicts.append("no usable label source")

    conflict_status = "conflict" if conflicts else "ok"
    return {
        "episode_id": episode_id,
        "binary_risk_label": selected.binary_risk_label if selected else None,
        "severity_score": selected.severity_score if selected else None,
        "risk_category": selected.risk_category if selected else None,
        "first_risk_step": selected.first_risk_step if selected else None,
        "label_source": selected.source if selected else None,
        "judge_version": selected.judge_version if selected else None,
        "suffix_policy": suffix_policy,
        "label_conflict_status": conflict_status,
        "conflicts": conflicts,
        "available_sources": {
            "benchmark_metadata": metadata_label is not None,
            "oracle_diagnostic": oracle_label is not None,
            "llm_judge": judge_label is not None,
        },
        "oracle_used_as_label": bool(selected is oracle_label and allow_oracle_as_label),
    }

