from __future__ import annotations

import pandas as pd
from src.monitor.logger import load_trace_events, group_by_episode
from src.monitor.schema import trace_event_sort_key
from src.features.extractor import extract_step_features
from src.oracles.rules import evaluate_oracles
from src.oracles.scoring import rule_based_score, monotone_smooth, policy_from_score


def make_prefix_dataset(trace_path: str) -> pd.DataFrame:
    events = load_trace_events(trace_path)
    rows = []
    for episode_id, eps in group_by_episode(events).items():
        eps_sorted = sorted(eps, key=trace_event_sort_key)
        t_risk = next((e.t_risk for e in eps_sorted if e.t_risk is not None), None)
        severity = max([float(e.severity or 0.0) for e in eps_sorted] or [0.0])
        smoothed = 0.0
        for idx, event in enumerate(eps_sorted):
            prefix = eps_sorted[: idx + 1]
            feat = extract_step_features(event)
            oracle_results = evaluate_oracles(prefix)
            oracle_violation = int(any(r.violation for r in oracle_results))
            raw_score = rule_based_score(feat)
            if oracle_violation:
                raw_score = max(raw_score, 75.0)
            smoothed = monotone_smooth(smoothed, raw_score)
            future_risk = int(t_risk is not None and event.step_id <= t_risk)
            lead_time = None if t_risk is None else t_risk - event.step_id
            rows.append({
                **feat,
                "future_risk_label": future_risk,
                "future_severity": severity if future_risk else 0.0,
                "t_risk": t_risk,
                "lead_time_if_alert_now": lead_time,
                "oracle_violation": oracle_violation,
                "oracle_rules": ";".join(r.rule_id for r in oracle_results if r.violation),
                "risk_score": smoothed,
                "policy": policy_from_score(smoothed),
            })
    return pd.DataFrame(rows)
