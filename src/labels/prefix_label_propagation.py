from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from src.labels.judge_schema import JudgeLabel
from src.labels.label_validation import validate_judge_label


SuffixPolicy = Literal["future_only"]


PREFIX_LABEL_COLUMNS = [
    "future_risk_label",
    "future_severity",
    "severity_weighted_future_risk",
    "risk_category",
    "deception_indicator",
    "surface_benign_indicator",
    "first_risk_step",
]


def propagated_prefix_label(
    step_id: int,
    label: JudgeLabel | dict[str, Any],
    suffix_policy: SuffixPolicy = "future_only",
) -> dict[str, Any]:
    if suffix_policy != "future_only":
        raise ValueError(f"Unsupported suffix policy: {suffix_policy}")
    judge_label = label if isinstance(label, JudgeLabel) else validate_judge_label(label)

    future_risk = 0
    if judge_label.binary_risk_label == 1 and judge_label.first_risk_step is not None:
        future_risk = int(step_id <= judge_label.first_risk_step)
    future_severity = float(judge_label.severity_score) if future_risk else 0.0
    return {
        "future_risk_label": future_risk,
        "future_severity": future_severity,
        "severity_weighted_future_risk": future_risk * future_severity,
        "risk_category": judge_label.risk_category if future_risk else "none",
        "deception_indicator": bool(judge_label.deception_indicator) if future_risk else False,
        "surface_benign_indicator": bool(judge_label.surface_benign_indicator) if future_risk else False,
        "first_risk_step": judge_label.first_risk_step,
    }


def propagate_label_to_prefix_dataframe(
    prefix_df: pd.DataFrame,
    label: JudgeLabel | dict[str, Any],
    suffix_policy: SuffixPolicy = "future_only",
) -> pd.DataFrame:
    if "step_id" not in prefix_df.columns:
        raise ValueError("prefix_df must contain step_id")
    judge_label = label if isinstance(label, JudgeLabel) else validate_judge_label(label)
    df = prefix_df.copy()
    for idx, row in df.iterrows():
        propagated = propagated_prefix_label(int(row["step_id"]), judge_label, suffix_policy=suffix_policy)
        for column, value in propagated.items():
            df.at[idx, column] = value
    return df


SUFFIX_POLICY_DESCRIPTION = {
    "future_only": (
        "Prefixes before or at first_risk_step are positive because risk is still future or present. "
        "Prefixes after first_risk_step are set to future_risk_label=0 and future_severity=0 because the target is "
        "future risk from that prefix onward rather than retrospective incident attribution."
    )
}
