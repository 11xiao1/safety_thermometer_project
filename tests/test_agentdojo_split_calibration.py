import json
import subprocess
import sys

import pandas as pd

from scripts.calibrate_agentdojo_split_risk_estimator import (
    DEFAULT_SCORE_COLUMN,
    OUTPUT_COLUMNS,
    calibrate_agentdojo_split_risk_estimator,
)


SCRIPT = "scripts/calibrate_agentdojo_split_risk_estimator.py"


def _write_predictions(path, labels=(0, 0, 0, 1, 1, 1, 0, 1, 0, 1)):
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "episode_id": f"episode_{index // 2}",
            "step_id": index + 1,
            "hook_type": "pre_step",
            "future_risk_label": label,
            "oracle_violation": int(label),
            "risk_score_logistic": 20.0 + 50.0 * label,
            "risk_score_random_forest": 25.0 + 45.0 * label,
            "risk_score_hist_gradient_boosting": 10.0 + 75.0 * label,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metrics(path):
    path.write_text(
        json.dumps({
            "test_split_used": False,
            "leakage_audit": {"passed": True},
        }),
        encoding="utf-8",
    )


def test_agentdojo_split_calibration_cli_writes_scores_and_metrics(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    metrics_in = tmp_path / "metrics_in.json"
    out_path = tmp_path / "scores.csv"
    metrics_out = tmp_path / "metrics_out.json"
    _write_predictions(predictions_path)
    _write_metrics(metrics_in)

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--predictions",
            str(predictions_path),
            "--metrics-in",
            str(metrics_in),
            "--out",
            str(out_path),
            "--metrics-out",
            str(metrics_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    scores = pd.read_csv(out_path)
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert scores.columns.tolist() == OUTPUT_COLUMNS
    assert scores["thermometer_score"].between(0, 100).all()
    assert set(scores["policy"]).issubset({"continue", "watch", "verify", "alert", "block"})
    assert metrics["score_column"] == DEFAULT_SCORE_COLUMN
    assert metrics["test_split_used"] is False
    assert metrics["prior_leakage_audit_passed"] is True
    assert "platt" in metrics["calibration_methods_used"]
    assert metrics["row_count"] == len(scores)


def test_agentdojo_split_calibration_supports_explicit_score_column(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    metrics_in = tmp_path / "metrics_in.json"
    out_path = tmp_path / "scores.csv"
    metrics_out = tmp_path / "metrics_out.json"
    _write_predictions(predictions_path)
    _write_metrics(metrics_in)

    metrics = calibrate_agentdojo_split_risk_estimator(
        predictions_path=predictions_path,
        metrics_in_path=metrics_in,
        out_path=out_path,
        metrics_out_path=metrics_out,
        score_column="risk_score_logistic",
    )

    assert metrics["score_column"] == "risk_score_logistic"
    assert out_path.exists()
    scores = pd.read_csv(out_path)
    assert scores["raw_score"].between(0, 100).all()


def test_agentdojo_split_calibration_uses_identity_when_labels_one_class(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    metrics_in = tmp_path / "metrics_in.json"
    out_path = tmp_path / "scores.csv"
    metrics_out = tmp_path / "metrics_out.json"
    _write_predictions(predictions_path, labels=(0, 0, 0, 0))
    _write_metrics(metrics_in)

    metrics = calibrate_agentdojo_split_risk_estimator(
        predictions_path=predictions_path,
        metrics_in_path=metrics_in,
        out_path=out_path,
        metrics_out_path=metrics_out,
    )

    assert metrics["calibration_methods_used"] == ["identity"]
    assert metrics["selected_calibration_method"] == "identity"
    assert "Platt scaling skipped because validation labels have one class." in metrics["warnings"]
    scores = pd.read_csv(out_path)
    assert scores["thermometer_score"].tolist() == scores["raw_score"].tolist()


def test_agentdojo_split_calibration_does_not_read_test_split(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.csv"
    metrics_in = tmp_path / "metrics_in.json"
    out_path = tmp_path / "scores.csv"
    metrics_out = tmp_path / "metrics_out.json"
    test_path = tmp_path / "agentdojo_test.csv"
    _write_predictions(predictions_path)
    _write_metrics(metrics_in)
    test_path.write_text("must,not,be,read\n", encoding="utf-8")
    original_read_csv = pd.read_csv

    def guarded_read_csv(path, *args, **kwargs):
        if str(path) == str(test_path):
            raise AssertionError("test split must not be read")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    metrics = calibrate_agentdojo_split_risk_estimator(
        predictions_path=predictions_path,
        metrics_in_path=metrics_in,
        out_path=out_path,
        metrics_out_path=metrics_out,
    )

    assert metrics["test_split_used"] is False
