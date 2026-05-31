from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.oracles.scoring import policy_from_score


TARGET_COLUMN = "future_risk_label"
RAW_SCORE_COLUMNS = {
    "logistic": "risk_score_logistic",
    "random_forest": "risk_score_random_forest",
}
CALIBRATED_SCORE_COLUMNS = {
    "logistic": "calibrated_score_logistic",
    "random_forest": "calibrated_score_random_forest",
}
CALIBRATION_METHODS = Literal["auto", "identity", "platt", "isotonic"]
VALID_POLICIES = {"continue", "watch", "verify", "alert", "block"}


@dataclass(frozen=True)
class CalibrationResult:
    scores: pd.DataFrame
    methods: dict[str, str]


def _clip_0_to_100(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)


def _choose_method(y: pd.Series, n_samples: int, requested: CALIBRATION_METHODS) -> str:
    if requested != "auto":
        return requested
    if y.nunique() < 2 or n_samples < 6:
        return "identity"
    if n_samples >= 30 and y.value_counts().min() >= 3:
        return "isotonic"
    return "platt"


def calibrate_scores(
    raw_scores,
    labels,
    method: CALIBRATION_METHODS = "auto",
) -> tuple[np.ndarray, str]:
    raw = _clip_0_to_100(raw_scores)
    y = pd.Series(labels).astype(int)
    selected_method = _choose_method(y, len(y), method)

    if selected_method == "identity":
        return raw, selected_method

    if y.nunique() < 2:
        return raw, "identity"

    x = (raw / 100.0).reshape(-1, 1)
    if selected_method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(x, y)
        return _clip_0_to_100(model.predict_proba(x)[:, 1] * 100.0), selected_method

    if selected_method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw / 100.0, y)
        return _clip_0_to_100(model.predict(raw / 100.0) * 100.0), selected_method

    raise ValueError(f"Unknown calibration method: {selected_method}")


def build_thermometer_scores(
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    method: CALIBRATION_METHODS = "auto",
) -> CalibrationResult:
    df = pd.read_csv(predictions_path).dropna(subset=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int)
    methods: dict[str, str] = {}

    output_columns = ["episode_id", "step_id"]
    if "hook_type" in df.columns:
        output_columns.append("hook_type")
    output_columns.extend([TARGET_COLUMN, "oracle_violation"])

    out = df[output_columns].copy()
    calibrated_arrays = []
    for model_name, raw_column in RAW_SCORE_COLUMNS.items():
        if raw_column not in df.columns:
            raise ValueError(f"Missing required prediction score column: {raw_column}")
        raw_output_column = f"raw_score_{model_name}"
        calibrated_column = CALIBRATED_SCORE_COLUMNS[model_name]
        out[raw_output_column] = _clip_0_to_100(df[raw_column])
        calibrated, selected_method = calibrate_scores(df[raw_column], y, method)
        out[calibrated_column] = calibrated
        calibrated_arrays.append(calibrated)
        methods[model_name] = selected_method

    out["thermometer_score"] = _clip_0_to_100(np.mean(calibrated_arrays, axis=0))
    out["policy"] = out["thermometer_score"].map(policy_from_score)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

    return CalibrationResult(scores=out, methods=methods)
