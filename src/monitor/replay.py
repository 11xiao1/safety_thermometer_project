from __future__ import annotations

import pandas as pd
from src.monitor.logger import load_trace_events, group_by_episode
from src.monitor.schema import TraceEvent, trace_event_sort_key
from src.features.extractor import extract_step_features
from src.oracles.rules import evaluate_oracles
from src.oracles.scoring import rule_based_score, monotone_smooth, policy_from_score


def _episode_risk_steps(events: list[TraceEvent]) -> list[int]:
    risk_steps: set[int] = set()
    for event in events:
        if event.t_risk is not None:
            risk_steps.add(event.t_risk)
        if event.risk_event:
            risk_steps.add(event.step_id)

    oracle_results = evaluate_oracles(events)
    for result in oracle_results:
        if result.violation and result.step_id is not None:
            risk_steps.add(result.step_id)

    return sorted(risk_steps)


def _future_severity(events: list[TraceEvent], current_step: int) -> float:
    return max(
        [float(event.severity or 0.0) for event in events if event.step_id >= current_step] or [0.0]
    )


def make_prefix_dataset(trace_path: str) -> pd.DataFrame:
    events = load_trace_events(trace_path)
    rows = []
    for episode_id, eps in group_by_episode(events).items():
        eps_sorted = sorted(eps, key=trace_event_sort_key)
        risk_steps = _episode_risk_steps(eps_sorted)
        t_risk = risk_steps[0] if risk_steps else None
        cumulative_state_modifying_count = 0
        cumulative_irreversible_count = 0
        cumulative_sensitive_access_count = 0
        cumulative_external_send_count = 0
        cumulative_fallback_count = 0
        confirmation_seen_so_far = 0
        smoothed = 0.0
        max_risk_score_so_far = 0.0
        for idx, event in enumerate(eps_sorted):
            prefix = eps_sorted[: idx + 1]
            feat = extract_step_features(event)
            cumulative_state_modifying_count += int(feat["f_state_modifying"])
            cumulative_irreversible_count += int(feat["f_irreversible"])
            cumulative_sensitive_access_count += int(feat["f_sensitive_access"])
            cumulative_external_send_count += int(feat["f_external_send"])
            cumulative_fallback_count += int(feat["f_fallback_count"])
            confirmation_seen_so_far = int(
                bool(confirmation_seen_so_far) or bool(feat["f_confirm_seen"])
            )
            oracle_results = evaluate_oracles(prefix)
            oracle_violation = int(any(r.violation for r in oracle_results))
            raw_score = rule_based_score(feat)
            if oracle_violation:
                raw_score = max(raw_score, 75.0)
            smoothed = monotone_smooth(smoothed, raw_score)
            max_risk_score_so_far = max(max_risk_score_so_far, smoothed)
            future_risk = int(any(risk_step >= event.step_id for risk_step in risk_steps))
            future_severity = _future_severity(eps_sorted, event.step_id) if future_risk else 0.0
            lead_time = None if t_risk is None else t_risk - event.step_id
            rows.append({
                **feat,
                "cumulative_state_modifying_count": cumulative_state_modifying_count,
                "cumulative_irreversible_count": cumulative_irreversible_count,
                "cumulative_sensitive_access_count": cumulative_sensitive_access_count,
                "cumulative_external_send_count": cumulative_external_send_count,
                "cumulative_fallback_count": cumulative_fallback_count,
                "confirmation_seen_so_far": confirmation_seen_so_far,
                "max_risk_score_so_far": max_risk_score_so_far,
                "oracle_violation": oracle_violation,
                "oracle_rules": ";".join(r.rule_id for r in oracle_results if r.violation),
                "risk_score": smoothed,
                "policy": policy_from_score(smoothed),
                "future_risk_label": future_risk,
                "future_severity": future_severity,
                "t_risk": t_risk,
                "lead_time_if_alert_now": lead_time,
            })
    return pd.DataFrame(rows)
