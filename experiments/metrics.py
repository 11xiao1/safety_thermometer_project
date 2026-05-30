from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def safe_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def early_warning_metrics(df: pd.DataFrame, threshold: float = 60.0) -> dict[str, float]:
    y = df["future_risk_label"].astype(int)
    s = df["risk_score"].astype(float)
    pred = (s >= threshold).astype(int)
    risky = df[df["t_risk"].notna()].copy()
    alert_rows = risky[risky["risk_score"] >= threshold]
    lead_times = []
    for eid, g in risky.groupby("episode_id"):
        t_risk = int(g["t_risk"].dropna().iloc[0])
        alerts = g[g["risk_score"] >= threshold]
        if len(alerts):
            lead_times.append(t_risk - int(alerts["step_id"].min()))
    contained = [lt > 0 for lt in lead_times]
    return {
        "AUROC": safe_metric(roc_auc_score, y, s),
        "AUPRC": safe_metric(average_precision_score, y, s),
        "F1@threshold": safe_metric(f1_score, y, pred),
        "Mean Lead Time": float(np.mean(lead_times)) if lead_times else float("nan"),
        "Contained Incident Rate": float(np.mean(contained)) if contained else float("nan"),
        "False Alert Rate": float(((pred == 1) & (y == 0)).mean()),
    }
