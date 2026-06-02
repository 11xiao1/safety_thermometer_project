import subprocess
import sys

import pandas as pd

from src.models.calibration import VALID_POLICIES


TOY_TRACE = "data/samples/toy_episodes.jsonl"
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
    prefix_path = tmp_path / "toy_prefix_dataset.csv"
    predictions_path = tmp_path / "toy_risk_estimator_predictions.csv"
    out_path = tmp_path / "toy_thermometer_scores.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/replay.py",
            "--trace",
            TOY_TRACE,
            "--out",
            str(prefix_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_toy_risk_estimator.py",
            "--data",
            str(prefix_path),
            "--out",
            str(predictions_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_toy_risk_estimator.py",
            "--pred",
            str(predictions_path),
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
