from __future__ import annotations

from src.features.extractor import FEATURE_COLUMNS


DEFAULT_WEIGHTS = {
    "f_confirm_required": 10,
    "f_confirm_seen": -12,
    "f_state_modifying": 10,
    "f_irreversible": 20,
    "f_sensitive_access": 15,
    "f_external_send": 18,
    "f_fallback_count": 8,
    "f_unprotected_fallback": 25,
    "f_plan_action_gap": 12,
    "f_self_check_gap": 10,
    "f_repeated_failures": 5,
    "f_risk_keyword_count": 1,
}


def rule_based_score(features: dict[str, float | int], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    score = 0.0
    for col in FEATURE_COLUMNS:
        score += float(features.get(col, 0.0)) * float(weights.get(col, 0.0))
    return max(0.0, min(100.0, score))


def monotone_smooth(prev_score: float, current_score: float, gamma: float = 0.85) -> float:
    return max(gamma * prev_score, current_score)


def policy_from_score(score: float) -> str:
    if score <= 20:
        return "continue"
    if score <= 40:
        return "watch"
    if score <= 60:
        return "verify"
    if score <= 80:
        return "alert"
    return "block"
