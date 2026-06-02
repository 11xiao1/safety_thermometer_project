from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.thermometer_baseline import TARGET_COLUMN  # noqa: E402
from src.oracles.scoring import policy_from_score  # noqa: E402


DEFAULT_PREDICTIONS = "outputs/agentdojo_multisuite_combined/agentdojo_split_risk_estimator_predictions.csv"
DEFAULT_METRICS_IN = "outputs/agentdojo_multisuite_combined/agentdojo_split_risk_estimator_metrics.json"
DEFAULT_OUT = "outputs/agentdojo_multisuite_combined/agentdojo_thermometer_scores_val.csv"
DEFAULT_METRICS_OUT = "outputs/agentdojo_multisuite_combined/agentdojo_calibration_metrics_val.json"
DEFAULT_SCORE_COLUMN = "risk_score_hist_gradient_boosting"
OUTPUT_COLUMNS = [
    "episode_id",
    "step_id",
    "hook_type",
    TARGET_COLUMN,
    "oracle_violation",
    "raw_score",
    "calibrated_score_platt",
    "calibrated_score_isotonic",
    "thermometer_score",
    "policy",
]


def _clip_probability(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _score_to_probability(values) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if len(raw) == 0:
        return raw
    if np.nanmax(raw) > 1.0:
        raw = raw / 100.0
    return _clip_probability(raw)


def _label_counts(values: pd.Series) -> dict[str, int]:
    return dict(sorted(Counter(str(int(value)) for value in values.dropna().astype(int)).items()))


def _fit_platt(raw_probability: np.ndarray, labels: pd.Series) -> tuple[np.ndarray, str | None]:
    if labels.nunique() < 2:
        return raw_probability, "Platt scaling skipped because validation labels have one class."
    model = LogisticRegression(max_iter=1000)
    model.fit(raw_probability.reshape(-1, 1), labels.astype(int))
    return _clip_probability(model.predict_proba(raw_probability.reshape(-1, 1))[:, 1]), None


def _fit_isotonic(raw_probability: np.ndarray, labels: pd.Series) -> tuple[np.ndarray | None, str | None]:
    label_counts = labels.astype(int).value_counts()
    if labels.nunique() < 2:
        return None, "Isotonic calibration skipped because validation labels have one class."
    if len(labels) < 10 or label_counts.min() < 3:
        return None, "Isotonic calibration skipped because validation sample count or class support is too small."
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_probability, labels.astype(int))
    return _clip_probability(model.predict(raw_probability)), None


def _ece(labels: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> float | None:
    if len(labels) == 0:
        return None
    y = labels.astype(int).to_numpy()
    probabilities = _clip_probability(probabilities)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not mask.any():
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(y[mask]))
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return float(ece)


def _score_summary(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float | None]:
    scores = _clip_probability(probabilities) * 100.0
    y = labels.astype(int).to_numpy()
    safe = scores[y == 0]
    risky = scores[y == 1]
    return {
        "mean_thermometer_score_safe": float(np.mean(safe)) if len(safe) else None,
        "mean_thermometer_score_risky": float(np.mean(risky)) if len(risky) else None,
    }


def calibrate_agentdojo_split_risk_estimator(
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    metrics_in_path: str | Path = DEFAULT_METRICS_IN,
    out_path: str | Path = DEFAULT_OUT,
    metrics_out_path: str | Path = DEFAULT_METRICS_OUT,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> dict[str, Any]:
    predictions_path = Path(predictions_path)
    metrics_in_path = Path(metrics_in_path)
    out_path = Path(out_path)
    metrics_out_path = Path(metrics_out_path)

    df = pd.read_csv(predictions_path).dropna(subset=[TARGET_COLUMN]).copy()
    if score_column not in df.columns:
        raise ValueError(f"Missing score column: {score_column}")
    if "oracle_violation" not in df.columns:
        df["oracle_violation"] = 0
    if "hook_type" not in df.columns:
        df["hook_type"] = ""

    prior_metrics = json.loads(metrics_in_path.read_text(encoding="utf-8")) if metrics_in_path.exists() else {}
    labels = df[TARGET_COLUMN].astype(int)
    raw_probability = _score_to_probability(df[score_column])
    warnings: list[str] = []

    platt_probability, warning = _fit_platt(raw_probability, labels)
    if warning:
        warnings.append(warning)

    isotonic_probability, warning = _fit_isotonic(raw_probability, labels)
    if warning:
        warnings.append(warning)

    if labels.nunique() < 2:
        methods_used = ["identity"]
        selected_probability = raw_probability
        selected_calibration_method = "identity"
    else:
        methods_used = ["identity", "platt"]
        selected_probability = platt_probability
        selected_calibration_method = "platt"
    if labels.nunique() >= 2 and isotonic_probability is not None:
        methods_used.append("isotonic")
        selected_probability = isotonic_probability
        selected_calibration_method = "isotonic"

    out = pd.DataFrame({
        "episode_id": df["episode_id"],
        "step_id": df["step_id"],
        "hook_type": df["hook_type"],
        TARGET_COLUMN: labels,
        "oracle_violation": df["oracle_violation"].astype(int),
        "raw_score": raw_probability * 100.0,
        "calibrated_score_platt": platt_probability * 100.0,
        "calibrated_score_isotonic": (
            isotonic_probability * 100.0 if isotonic_probability is not None else np.nan
        ),
        "thermometer_score": selected_probability * 100.0,
    })
    out["policy"] = out["thermometer_score"].map(policy_from_score)

    metrics: dict[str, Any] = {
        "score_column": score_column,
        "calibration_methods_used": methods_used,
        "selected_calibration_method": selected_calibration_method,
        "label_counts": _label_counts(labels),
        "row_count": int(len(out)),
        "brier_score_raw": float(brier_score_loss(labels, raw_probability)) if len(out) else None,
        "brier_score_platt": float(brier_score_loss(labels, platt_probability)) if len(out) else None,
        "brier_score_isotonic": (
            float(brier_score_loss(labels, isotonic_probability)) if isotonic_probability is not None else None
        ),
        "ece_raw": _ece(labels, raw_probability),
        "ece_platt": _ece(labels, platt_probability),
        "ece_isotonic": _ece(labels, isotonic_probability) if isotonic_probability is not None else None,
        **_score_summary(labels, selected_probability),
        "policy_counts": {
            str(key): int(value)
            for key, value in out["policy"].value_counts().sort_index().items()
        },
        "warnings": warnings,
        "test_split_used": False,
        "diagnostic_only": ["oracle_violation"],
        "target": TARGET_COLUMN,
        "input_predictions": str(predictions_path),
        "input_metrics": str(metrics_in_path),
        "prior_leakage_audit_passed": prior_metrics.get("leakage_audit", {}).get("passed"),
        "limitation": "Validation-only calibration sanity check, not a final benchmark result.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    out[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    metrics_out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate AgentDojo split Risk Estimator validation scores.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics-in", default=DEFAULT_METRICS_IN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    args = parser.parse_args()

    metrics = calibrate_agentdojo_split_risk_estimator(
        predictions_path=args.predictions,
        metrics_in_path=args.metrics_in,
        out_path=args.out,
        metrics_out_path=args.metrics_out,
        score_column=args.score_column,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
