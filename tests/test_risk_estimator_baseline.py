import subprocess
import sys

import pandas as pd

from src.models.thermometer_baseline import (
    EXCLUDED_FEATURE_COLUMNS,
    PREDICTION_COLUMNS,
    TARGET_COLUMN,
    select_feature_columns,
)


TOY_TRACE = "data/samples/toy_episodes.jsonl"
SCORE_COLUMNS = ["risk_score_logistic", "risk_score_random_forest"]


def _build_prefix_dataset(tmp_path):
    prefix_path = tmp_path / "toy_prefix_dataset.csv"
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
    return str(prefix_path)


def test_train_toy_risk_estimator_script_runs_successfully(tmp_path):
    prefix_dataset = _build_prefix_dataset(tmp_path)
    out_path = tmp_path / "toy_risk_estimator_predictions.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_toy_risk_estimator.py",
            "--data",
            prefix_dataset,
            "--out",
            str(out_path),
        ],
        check=True,
    )

    assert out_path.exists()


def test_risk_estimator_predictions_contain_required_columns(tmp_path):
    prefix_dataset = _build_prefix_dataset(tmp_path)
    out_path = tmp_path / "toy_risk_estimator_predictions.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_toy_risk_estimator.py",
            "--data",
            prefix_dataset,
            "--out",
            str(out_path),
        ],
        check=True,
    )

    predictions = pd.read_csv(out_path)
    assert all(column in predictions.columns for column in PREDICTION_COLUMNS)
    assert "hook_type" in predictions.columns


def test_risk_estimator_scores_are_between_0_and_100(tmp_path):
    prefix_dataset = _build_prefix_dataset(tmp_path)
    out_path = tmp_path / "toy_risk_estimator_predictions.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_toy_risk_estimator.py",
            "--data",
            prefix_dataset,
            "--out",
            str(out_path),
        ],
        check=True,
    )

    predictions = pd.read_csv(out_path)
    for column in SCORE_COLUMNS:
        assert predictions[column].between(0, 100).all()


def test_risk_estimator_target_is_future_risk_label_not_oracle_violation(tmp_path):
    prefix_dataset = _build_prefix_dataset(tmp_path)
    df = pd.read_csv(prefix_dataset)
    feature_columns = select_feature_columns(df)

    assert TARGET_COLUMN == "future_risk_label"
    assert "oracle_violation" in EXCLUDED_FEATURE_COLUMNS
    assert "oracle_violation" not in feature_columns
    assert TARGET_COLUMN not in feature_columns


def test_make_tables_reads_risk_estimator_predictions(tmp_path):
    prefix_dataset = _build_prefix_dataset(tmp_path)
    pred_path = tmp_path / "toy_risk_estimator_predictions.csv"
    outdir = tmp_path / "tables"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_toy_risk_estimator.py",
            "--data",
            prefix_dataset,
            "--out",
            str(pred_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/make_tables.py",
            "--pred",
            str(pred_path),
            "--outdir",
            str(outdir),
        ],
        check=True,
    )

    table3 = pd.read_csv(outdir / "table3_toy_risk_estimator.csv")
    table4 = pd.read_csv(outdir / "table4_toy_risk_estimator.csv")
    assert table3["Method"].tolist() == ["rule_based", "logistic", "random_forest"]
    assert all(
        column in table3.columns
        for column in ["AUROC", "AUPRC", "F1@50", "False Alert Rate", "Risky Detected Rate Proxy"]
    )
    assert table4["Method"].tolist() == ["rule_based", "logistic", "random_forest"]
