"""RL policy skeleton for later weeks.

Recommended interpretation:
- Supervised thermometer estimates calibrated future risk.
- RL or contextual bandit learns intervention policy over states/scores.

Do not start here. Build trace + oracle + supervised baseline first.
"""

ACTIONS = ["continue", "watch", "verify", "alert", "block"]


def reward(action: str, future_risk_label: int, lead_time_if_alert_now: float | None) -> float:
    if action in {"alert", "block", "verify"}:
        if future_risk_label and lead_time_if_alert_now is not None and lead_time_if_alert_now > 0:
            return 1.0
        if future_risk_label:
            return 0.0
        return -0.3
    if action == "continue":
        return -1.0 if future_risk_label else 0.2
    return 0.0
