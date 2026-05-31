import subprocess
import sys

import pandas as pd

from src.models.calibration import VALID_POLICIES


PREDICTIONS = "outputs/toy_risk_estimator_predictions.csv"
REQUIRED_COLUMNS = [
    "episode_id",
    "step_id",
    "future_risk_label",
    "oracle_violation",
    "raw_score_logistic",
    "raw_score_random_forest",
    "calibrated_score_logistic",
    "calibrated_score_random_forest",
    "thermometer_score",
    "policy",
]
SCORE_COLUMNS = [
    "raw_score_logistic",
    "raw_score_random_forest",
    "calibrated_score_logistic",
    "calibrated_score_random_forest",
    "thermometer_score",
]


def _run_calibration(tmp_path):
    out_path = tmp_path / "toy_thermometer_scores.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_toy_risk_estimator.py",
            "--pred",
            PREDICTIONS,
            "--out",
            str(out_path),
        ],
        check=True,
    )
    return pd.read_csv(out_path)


def test_calibrated_output_has_required_columns(tmp_path):
    df = _run_calibration(tmp_path)

    assert all(column in df.columns for column in REQUIRED_COLUMNS)


def test_calibrated_scores_are_between_0_and_100(tmp_path):
    df = _run_calibration(tmp_path)

    for column in SCORE_COLUMNS:
        assert df[column].between(0, 100).all()


def test_calibrated_policies_are_valid(tmp_path):
    df = _run_calibration(tmp_path)

    assert set(df["policy"]).issubset(VALID_POLICIES)
