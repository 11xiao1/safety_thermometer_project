from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.thermometer_baseline import TARGET_COLUMN  # noqa: E402


DEFAULT_SCORES = "outputs/agentdojo_multisuite_combined/agentdojo_thermometer_scores_val.csv"
DEFAULT_CALIBRATION_METRICS = "outputs/agentdojo_multisuite_combined/agentdojo_calibration_metrics_val.json"
DEFAULT_RISK_METRICS = "outputs/agentdojo_multisuite_combined/agentdojo_split_risk_estimator_metrics.json"
DEFAULT_TABLE3 = "outputs/agentdojo_multisuite_combined/table3_agentdojo_validation_calibrated.csv"
DEFAULT_TABLE4 = "outputs/agentdojo_multisuite_combined/table4_agentdojo_validation_policy.csv"
LIMITATION = "Validation-only calibration sanity check, not a final benchmark result."
SCORE_COLUMNS = {
    "raw_hist_gradient_boosting": "raw_score",
    "calibrated_platt": "calibrated_score_platt",
    "calibrated_isotonic": "calibrated_score_isotonic",
    "selected_thermometer_score": "thermometer_score",
}


def _label_counts(labels: pd.Series) -> str:
    counts = Counter(str(int(value)) for value in labels.dropna().astype(int))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _safe_metric(fn, labels: pd.Series, scores: pd.Series) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if len(valid) == 0 or valid["label"].nunique() < 2:
        return None
    return float(fn(valid["label"].astype(int), valid["score"].astype(float)))


def _f1_at_50(labels: pd.Series, scores: pd.Series) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if len(valid) == 0:
        return None
    pred = (valid["score"].astype(float) >= 50.0).astype(int)
    return float(f1_score(valid["label"].astype(int), pred, zero_division=0))


def _mean_for_label(labels: pd.Series, scores: pd.Series, label: int) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    values = valid.loc[valid["label"].astype(int) == label, "score"].astype(float)
    return float(values.mean()) if len(values) else None


def _brier(labels: pd.Series, scores: pd.Series) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if len(valid) == 0:
        return None
    return float(brier_score_loss(valid["label"].astype(int), (valid["score"].astype(float) / 100.0).clip(0.0, 1.0)))


def _ece(labels: pd.Series, scores: pd.Series, n_bins: int = 10) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if len(valid) == 0:
        return None
    y = valid["label"].astype(int).to_numpy()
    probabilities = (valid["score"].astype(float).to_numpy() / 100.0).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _calibration_method(score_name: str, calibration_metrics: dict[str, Any]) -> str:
    if score_name == "raw_hist_gradient_boosting":
        return "identity"
    if score_name == "calibrated_platt":
        return "platt"
    if score_name == "calibrated_isotonic":
        methods = calibration_metrics.get("calibration_methods_used", [])
        return "isotonic" if "isotonic" in methods else "not_available"
    if score_name == "selected_thermometer_score":
        return str(calibration_metrics.get("selected_calibration_method", "unknown"))
    return "unknown"


def _make_table3(df: pd.DataFrame, calibration_metrics: dict[str, Any]) -> pd.DataFrame:
    labels = df[TARGET_COLUMN].astype(int)
    rows = []
    for score_name, column in SCORE_COLUMNS.items():
        if column not in df.columns:
            continue
        scores = df[column].astype(float)
        rows.append({
            "score_name": score_name,
            "split": "validation",
            "row_count": int(scores.dropna().shape[0]),
            "label_counts": _label_counts(labels),
            "AUROC": _safe_metric(roc_auc_score, labels, scores),
            "AUPRC": _safe_metric(average_precision_score, labels, scores),
            "F1@50": _f1_at_50(labels, scores),
            "Brier score": _brier(labels, scores),
            "ECE": _ece(labels, scores),
            "mean_score_safe": _mean_for_label(labels, scores, 0),
            "mean_score_risky": _mean_for_label(labels, scores, 1),
            "calibration_method": _calibration_method(score_name, calibration_metrics),
            "test_split_used": False,
            "limitation": LIMITATION,
        })
    return pd.DataFrame(rows)


def _make_table4(df: pd.DataFrame, calibration_metrics: dict[str, Any]) -> pd.DataFrame:
    labels = df[TARGET_COLUMN].astype(int)
    risky = labels == 1
    safe = labels == 0
    alert_or_block = df["policy"].isin(["alert", "block"])
    risky_count = int(risky.sum())
    safe_count = int(safe.sum())
    risky_alert_or_block = int((risky & alert_or_block).sum())
    safe_alert_or_block = int((safe & alert_or_block).sum())
    policy_counts = {
        policy: int((df["policy"] == policy).sum())
        for policy in ["continue", "watch", "verify", "alert", "block"]
    }
    row = {
        "split": "validation",
        "row_count": int(len(df)),
        **policy_counts,
        "alert_or_block_count": int(alert_or_block.sum()),
        "alert_or_block_rate": float(alert_or_block.mean()) if len(df) else None,
        "risky_prefixes": risky_count,
        "safe_prefixes": safe_count,
        "risky_alert_or_block_count": risky_alert_or_block,
        "safe_alert_or_block_count": safe_alert_or_block,
        "false_alert_proxy": float(safe_alert_or_block / safe_count) if safe_count else None,
        "missed_risky_proxy": float((risky_count - risky_alert_or_block) / risky_count) if risky_count else None,
        "test_split_used": bool(calibration_metrics.get("test_split_used", False)),
        "limitation": LIMITATION,
    }
    return pd.DataFrame([row])


def make_agentdojo_calibrated_tables(
    scores_path: str | Path = DEFAULT_SCORES,
    calibration_metrics_path: str | Path = DEFAULT_CALIBRATION_METRICS,
    risk_metrics_path: str | Path = DEFAULT_RISK_METRICS,
    table3_path: str | Path = DEFAULT_TABLE3,
    table4_path: str | Path = DEFAULT_TABLE4,
) -> dict[str, Any]:
    scores_path = Path(scores_path)
    calibration_metrics_path = Path(calibration_metrics_path)
    risk_metrics_path = Path(risk_metrics_path)
    table3_path = Path(table3_path)
    table4_path = Path(table4_path)

    df = pd.read_csv(scores_path).dropna(subset=[TARGET_COLUMN]).copy()
    calibration_metrics = json.loads(calibration_metrics_path.read_text(encoding="utf-8"))
    risk_metrics = json.loads(risk_metrics_path.read_text(encoding="utf-8")) if risk_metrics_path.exists() else {}
    if calibration_metrics.get("test_split_used") is True or risk_metrics.get("test_split_used") is True:
        raise ValueError("Refusing to build validation-only tables from metrics that used the test split.")

    table3 = _make_table3(df, calibration_metrics)
    table4 = _make_table4(df, calibration_metrics)

    table3_path.parent.mkdir(parents=True, exist_ok=True)
    table4_path.parent.mkdir(parents=True, exist_ok=True)
    table3.to_csv(table3_path, index=False)
    table4.to_csv(table4_path, index=False)
    return {
        "status": "ok",
        "scores": str(scores_path),
        "calibration_metrics": str(calibration_metrics_path),
        "risk_metrics": str(risk_metrics_path),
        "table3": str(table3_path),
        "table4": str(table4_path),
        "table3_rows": int(len(table3)),
        "table4_rows": int(len(table4)),
        "test_split_used": False,
        "limitation": LIMITATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation-only AgentDojo calibrated tables.")
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--calibration-metrics", default=DEFAULT_CALIBRATION_METRICS)
    parser.add_argument("--risk-metrics", default=DEFAULT_RISK_METRICS)
    parser.add_argument("--table3", default=DEFAULT_TABLE3)
    parser.add_argument("--table4", default=DEFAULT_TABLE4)
    args = parser.parse_args()

    result = make_agentdojo_calibrated_tables(
        scores_path=args.scores,
        calibration_metrics_path=args.calibration_metrics,
        risk_metrics_path=args.risk_metrics,
        table3_path=args.table3,
        table4_path=args.table4,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
