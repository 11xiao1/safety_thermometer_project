from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.thermometer_baseline import (  # noqa: E402
    RULE_BASED_SCORE_COLUMN,
    TARGET_COLUMN,
    select_feature_columns,
)


DEFAULT_TRAIN = "outputs/splits/agentdojo_train.csv"
DEFAULT_VAL = "outputs/splits/agentdojo_val.csv"
DEFAULT_MANIFEST = "outputs/splits/split_manifest.json"
DEFAULT_PREDICTIONS = "outputs/agentdojo_split_risk_estimator_predictions.csv"
DEFAULT_METRICS = "outputs/agentdojo_split_risk_estimator_metrics.json"
PREDICTION_COLUMNS = [
    "episode_id",
    "step_id",
    "hook_type",
    TARGET_COLUMN,
    "oracle_violation",
    "risk_score_rule_based",
    "risk_score_logistic",
    "risk_score_random_forest",
]


def _label_counts(values: pd.Series) -> dict[str, int]:
    return dict(sorted(Counter(str(int(value)) for value in values.dropna().astype(int)).items()))


def _none_metrics() -> dict[str, float | None]:
    return {
        "auroc": None,
        "auprc": None,
        "f1_at_50": None,
        "mean_score_safe": None,
        "mean_score_risky": None,
    }


def _validation_metrics(y_true: pd.Series, y_score: np.ndarray) -> tuple[dict[str, float | None], list[str]]:
    warnings = []
    metrics = _none_metrics()
    if len(y_true) == 0:
        warnings.append("Validation split has no rows; validation metrics are unavailable.")
        return metrics, warnings

    y_true = y_true.astype(int)
    y_pred = (y_score >= 50.0).astype(int)
    metrics["f1_at_50"] = float(f1_score(y_true, y_pred, zero_division=0))

    safe_scores = y_score[y_true.to_numpy() == 0]
    risky_scores = y_score[y_true.to_numpy() == 1]
    metrics["mean_score_safe"] = float(np.mean(safe_scores)) if len(safe_scores) else None
    metrics["mean_score_risky"] = float(np.mean(risky_scores)) if len(risky_scores) else None

    if y_true.nunique() < 2:
        warnings.append("Validation split has one class; AUROC and AUPRC are unavailable.")
        return metrics, warnings

    metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics, warnings


def _constant_scores(train_y: pd.Series, n_rows: int) -> np.ndarray:
    prior = float(train_y.mean()) if len(train_y) else 0.0
    return np.full(n_rows, prior * 100.0)


def _fit_scores(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    warnings = []
    if train_y.nunique() < 2:
        warnings.append("Train split has one class; using constant-prior fallback scores.")
        scores = _constant_scores(train_y, len(val_X))
        return scores, scores.copy(), warnings, "constant_prior"

    logistic = LogisticRegression(max_iter=1000, random_state=random_state)
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=random_state,
    )
    logistic.fit(train_X, train_y)
    random_forest.fit(train_X, train_y)
    logistic_scores = np.clip(logistic.predict_proba(val_X)[:, 1] * 100.0, 0.0, 100.0)
    random_forest_scores = np.clip(random_forest.predict_proba(val_X)[:, 1] * 100.0, 0.0, 100.0)
    return logistic_scores, random_forest_scores, warnings, "trained"


def train_agentdojo_split_risk_estimator(
    train_path: str | Path = DEFAULT_TRAIN,
    val_path: str | Path = DEFAULT_VAL,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    metrics_path: str | Path = DEFAULT_METRICS,
    random_state: int = 7,
) -> dict[str, Any]:
    train_path = Path(train_path)
    val_path = Path(val_path)
    manifest_path = Path(manifest_path)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)

    train_df = pd.read_csv(train_path).dropna(subset=[TARGET_COLUMN]).copy()
    val_df = pd.read_csv(val_path).dropna(subset=[TARGET_COLUMN]).copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)
    val_df[TARGET_COLUMN] = val_df[TARGET_COLUMN].astype(int)
    feature_columns = select_feature_columns(train_df)
    if not feature_columns:
        raise ValueError("No numeric Evidence Stream feature columns found in train split.")

    missing_val_features = [column for column in feature_columns if column not in val_df.columns]
    if missing_val_features:
        raise ValueError("Validation split is missing feature columns: " + ", ".join(missing_val_features))

    train_X = train_df[feature_columns].fillna(0.0)
    train_y = train_df[TARGET_COLUMN]
    val_X = val_df[feature_columns].fillna(0.0)
    val_y = val_df[TARGET_COLUMN]

    logistic_scores, random_forest_scores, fit_warnings, fit_mode = _fit_scores(
        train_X,
        train_y,
        val_X,
        random_state,
    )
    logistic_metrics, logistic_warnings = _validation_metrics(val_y, logistic_scores)
    random_forest_metrics, random_forest_warnings = _validation_metrics(val_y, random_forest_scores)

    predictions = pd.DataFrame({
        "episode_id": val_df["episode_id"],
        "step_id": val_df["step_id"],
        "hook_type": val_df["hook_type"] if "hook_type" in val_df.columns else "",
        TARGET_COLUMN: val_y,
        "oracle_violation": val_df["oracle_violation"].astype(int) if "oracle_violation" in val_df.columns else 0,
        "risk_score_rule_based": val_df[RULE_BASED_SCORE_COLUMN].astype(float),
        "risk_score_logistic": logistic_scores,
        "risk_score_random_forest": random_forest_scores,
    })

    warnings = fit_warnings + logistic_warnings + [
        warning for warning in random_forest_warnings if warning not in logistic_warnings
    ]
    metrics: dict[str, Any] = {
        "target": TARGET_COLUMN,
        "diagnostic_only": ["oracle_violation"],
        "fit_mode": fit_mode,
        "random_state": random_state,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "manifest_path": str(manifest_path),
        "test_split_used": False,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_label_counts": _label_counts(train_y),
        "val_label_counts": _label_counts(val_y),
        "manifest_episode_counts": manifest.get("episode_counts", {}),
        "features": feature_columns,
        "logistic": logistic_metrics,
        "random_forest": random_forest_metrics,
        "warnings": warnings,
        "limitations": [
            "Mini-batch validation only.",
            "Not a final benchmark result.",
            "Test split was not used.",
            "No calibration was fit.",
        ],
    }

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions[PREDICTION_COLUMNS].to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Risk Estimator on AgentDojo train/val splits.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--val", default=DEFAULT_VAL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metrics = train_agentdojo_split_risk_estimator(
        train_path=args.train,
        val_path=args.val,
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
