from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "future_risk_label"
RULE_BASED_SCORE_COLUMN = "risk_score"
EXCLUDED_FEATURE_COLUMNS = {
    "episode_id",
    "step_id",
    "hook_type",
    "oracle_rules",
    "oracle_violation",
    "policy",
    TARGET_COLUMN,
    "future_severity",
    "t_risk",
    "lead_time_if_alert_now",
}
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


@dataclass(frozen=True)
class RiskEstimatorResult:
    predictions: pd.DataFrame
    feature_columns: list[str]
    metrics: dict[str, Any]
    warning: str | None = None


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in EXCLUDED_FEATURE_COLUMNS]


def _episode_level_split(df: pd.DataFrame, random_state: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    episode_labels = df.groupby("episode_id")[TARGET_COLUMN].max().astype(int)
    if len(episode_labels) < 4 or episode_labels.nunique() < 2:
        return None, None, "Toy data too small for an episode-level split; fitting and predicting on all rows."

    label_counts = episode_labels.value_counts()
    stratify = episode_labels if label_counts.min() >= 2 else None
    train_episodes, test_episodes = train_test_split(
        episode_labels.index.to_numpy(),
        test_size=0.4,
        random_state=random_state,
        stratify=stratify,
    )
    train_mask = df["episode_id"].isin(train_episodes).to_numpy()
    test_mask = df["episode_id"].isin(test_episodes).to_numpy()

    if df.loc[train_mask, TARGET_COLUMN].nunique() < 2:
        return None, None, "Episode split training rows have one class; fitting and predicting on all rows."
    return train_mask, test_mask, None


def _score_0_to_100(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1] * 100.0, 0.0, 100.0)


def _safe_classification_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    if y_true.nunique() < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def train_risk_estimator(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    random_state: int = 7,
) -> RiskEstimatorResult:
    df = pd.read_csv(csv_path).dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    feature_columns = select_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric Evidence Stream feature columns found.")

    X = df[feature_columns].fillna(0.0)
    y = df[TARGET_COLUMN]
    train_mask, test_mask, warning = _episode_level_split(df, random_state)
    if train_mask is None or test_mask is None:
        train_mask = np.ones(len(df), dtype=bool)
        test_mask = np.ones(len(df), dtype=bool)

    logistic = LogisticRegression(max_iter=1000, random_state=random_state)
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=random_state,
    )
    logistic.fit(X.loc[train_mask], y.loc[train_mask])
    random_forest.fit(X.loc[train_mask], y.loc[train_mask])

    logistic_scores = _score_0_to_100(logistic, X)
    random_forest_scores = _score_0_to_100(random_forest, X)
    predictions = pd.DataFrame({
        "episode_id": df["episode_id"],
        "step_id": df["step_id"],
        "hook_type": df["hook_type"] if "hook_type" in df.columns else "",
        TARGET_COLUMN: y,
        "oracle_violation": df["oracle_violation"].astype(int),
        "risk_score_rule_based": df[RULE_BASED_SCORE_COLUMN].astype(float),
        "risk_score_logistic": logistic_scores,
        "risk_score_random_forest": random_forest_scores,
    })

    y_test = y.loc[test_mask]
    metrics = {
        "target": TARGET_COLUMN,
        "features": feature_columns,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "logistic": _safe_classification_metrics(y_test, logistic_scores[test_mask]),
        "random_forest": _safe_classification_metrics(y_test, random_forest_scores[test_mask]),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)

    return RiskEstimatorResult(
        predictions=predictions[PREDICTION_COLUMNS],
        feature_columns=feature_columns,
        metrics=metrics,
        warning=warning,
    )


def train_baseline(csv_path: str) -> dict[str, Any]:
    result = train_risk_estimator(csv_path)
    return result.metrics
