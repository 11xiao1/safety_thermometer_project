import json
import subprocess
import sys

import pandas as pd

from scripts.train_agentdojo_split_risk_estimator import (
    PREDICTION_COLUMNS,
    train_agentdojo_split_risk_estimator,
)


SCRIPT = "scripts/train_agentdojo_split_risk_estimator.py"


def _rows(labels):
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "episode_id": f"episode_{index // 2}",
            "step_id": index + 1,
            "hook_type": "pre_step",
            "future_risk_label": label,
            "oracle_violation": int(label),
            "risk_score": 25.0 + 50.0 * label,
            "f_sensitive_access": label,
            "f_state_modifying": label,
            "cumulative_sensitive_access_count": index,
        })
    return rows


def _write_split_files(tmp_path, train_labels=(0, 0, 1, 1), val_labels=(0, 1)):
    train_path = tmp_path / "agentdojo_train.csv"
    val_path = tmp_path / "agentdojo_val.csv"
    test_path = tmp_path / "agentdojo_test.csv"
    manifest_path = tmp_path / "split_manifest.json"
    pd.DataFrame(_rows(train_labels)).to_csv(train_path, index=False)
    pd.DataFrame(_rows(val_labels)).to_csv(val_path, index=False)
    pd.DataFrame(_rows((1, 1))).to_csv(test_path, index=False)
    manifest_path.write_text(
        json.dumps({
            "episode_counts": {"train": 2, "val": 1, "test": 1},
            "outputs": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
        }),
        encoding="utf-8",
    )
    return train_path, val_path, test_path, manifest_path


def test_agentdojo_split_training_cli_writes_predictions_and_metrics(tmp_path):
    train_path, val_path, _, manifest_path = _write_split_files(tmp_path)
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--metrics",
            str(metrics_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert predictions_path.exists()
    assert metrics_path.exists()
    predictions = pd.read_csv(predictions_path)
    assert all(column in predictions.columns for column in PREDICTION_COLUMNS)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["target"] == "future_risk_label"
    assert metrics["test_split_used"] is False
    assert metrics["train_rows"] == 4
    assert metrics["val_rows"] == 2
    assert metrics["train_label_counts"] == {"0": 2, "1": 2}
    assert metrics["val_label_counts"] == {"0": 1, "1": 1}
    assert metrics["logistic"]["f1_at_50"] is not None
    assert metrics["random_forest"]["f1_at_50"] is not None
    assert metrics["hist_gradient_boosting"]["f1_at_50"] is not None
    assert metrics["leakage_audit"]["future_risk_label_not_used_as_feature"] is True
    assert metrics["leakage_audit"]["future_severity_not_used_as_feature"] is True
    assert metrics["leakage_audit"]["t_risk_not_used_as_feature"] is True
    assert metrics["leakage_audit"]["lead_time_if_alert_now_not_used_as_feature"] is True
    assert metrics["leakage_audit"]["test_split_used"] is False
    assert metrics["leakage_audit"]["passed"] is True


def test_single_class_train_uses_safe_fallback(tmp_path):
    train_path, val_path, _, manifest_path = _write_split_files(tmp_path, train_labels=(0, 0, 0), val_labels=(0, 1))
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_agentdojo_split_risk_estimator(
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )

    assert metrics["fit_mode"] == "constant_prior"
    assert "Train split has one class; using constant-prior fallback scores." in metrics["warnings"]
    predictions = pd.read_csv(predictions_path)
    assert predictions["risk_score_logistic"].tolist() == [0.0, 0.0]
    assert predictions["risk_score_random_forest"].tolist() == [0.0, 0.0]
    assert predictions["risk_score_hist_gradient_boosting"].tolist() == [0.0, 0.0]


def test_single_class_validation_warns_and_keeps_partial_metrics(tmp_path):
    train_path, val_path, _, manifest_path = _write_split_files(tmp_path, train_labels=(0, 0, 1, 1), val_labels=(0, 0))
    metrics = train_agentdojo_split_risk_estimator(
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        predictions_path=tmp_path / "predictions.csv",
        metrics_path=tmp_path / "metrics.json",
    )

    assert "Validation split has one class; AUROC and AUPRC are unavailable." in metrics["warnings"]
    assert metrics["logistic"]["auroc"] is None
    assert metrics["logistic"]["auprc"] is None
    assert metrics["logistic"]["f1_at_50"] is not None
    assert metrics["logistic"]["mean_score_safe"] is not None
    assert metrics["logistic"]["mean_score_risky"] is None
    assert metrics["hist_gradient_boosting"]["auroc"] is None
    assert metrics["hist_gradient_boosting"]["auprc"] is None


def test_training_function_does_not_read_test_split(tmp_path, monkeypatch):
    train_path, val_path, test_path, manifest_path = _write_split_files(tmp_path)
    original_read_csv = pd.read_csv

    def guarded_read_csv(path, *args, **kwargs):
        if str(path) == str(test_path):
            raise AssertionError("test split must not be read")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    metrics = train_agentdojo_split_risk_estimator(
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        predictions_path=tmp_path / "predictions.csv",
        metrics_path=tmp_path / "metrics.json",
    )

    assert metrics["test_split_used"] is False
    assert metrics["leakage_audit"]["test_split_used"] is False


def test_leakage_audit_excludes_future_and_diagnostic_columns(tmp_path):
    train_path, val_path, _, manifest_path = _write_split_files(tmp_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    for df in [train_df, val_df]:
        df["future_severity"] = df["future_risk_label"] * 10
        df["t_risk"] = df["future_risk_label"]
        df["lead_time_if_alert_now"] = df["future_risk_label"] + 1
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    metrics = train_agentdojo_split_risk_estimator(
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        predictions_path=tmp_path / "predictions.csv",
        metrics_path=tmp_path / "metrics.json",
    )

    audit = metrics["leakage_audit"]
    assert audit["future_risk_label_not_used_as_feature"] is True
    assert audit["future_severity_not_used_as_feature"] is True
    assert audit["t_risk_not_used_as_feature"] is True
    assert audit["lead_time_if_alert_now_not_used_as_feature"] is True
    assert audit["excluded_columns_present_in_features"] == []
    assert "oracle_violation" not in metrics["features"]
    assert "future_risk_label" not in metrics["features"]
    assert "future_severity" not in metrics["features"]
    assert "t_risk" not in metrics["features"]
    assert "lead_time_if_alert_now" not in metrics["features"]
