import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.make_agentdojo_early_warning_metrics import make_agentdojo_early_warning_metrics


SCRIPT = "scripts/make_agentdojo_early_warning_metrics.py"


def _write_scores(path):
    rows = [
        {
            "episode_id": "risky_early",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 0,
            "thermometer_score": 70.0,
            "policy": "alert",
        },
        {
            "episode_id": "risky_early",
            "step_id": 2,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "thermometer_score": 90.0,
            "policy": "block",
        },
        {
            "episode_id": "risky_missed",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 0,
            "thermometer_score": 20.0,
            "policy": "continue",
        },
        {
            "episode_id": "risky_missed",
            "step_id": 2,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "thermometer_score": 35.0,
            "policy": "watch",
        },
        {
            "episode_id": "risky_jit",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "thermometer_score": 65.0,
            "policy": "alert",
        },
        {
            "episode_id": "safe_false_alert",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 0,
            "oracle_violation": 0,
            "thermometer_score": 85.0,
            "policy": "block",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_val(path):
    rows = [
        {"episode_id": "risky_early", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_early", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_missed", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_missed", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_jit", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 1},
        {"episode_id": "safe_false_alert", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "t_risk": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metrics(path, test_split_used=False):
    path.write_text(json.dumps({"test_split_used": test_split_used, "warnings": []}), encoding="utf-8")


def _write_inputs(tmp_path, test_split_used=False):
    scores = tmp_path / "scores.csv"
    val = tmp_path / "val.csv"
    calibration_metrics = tmp_path / "calibration_metrics.json"
    _write_scores(scores)
    _write_val(val)
    _write_metrics(calibration_metrics, test_split_used=test_split_used)
    return scores, val, calibration_metrics


def test_agentdojo_early_warning_cli_writes_table_and_metrics(tmp_path):
    scores, val, calibration_metrics = _write_inputs(tmp_path)
    table = tmp_path / "table5.csv"
    metrics_path = tmp_path / "metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--scores",
            str(scores),
            "--val",
            str(val),
            "--calibration-metrics",
            str(calibration_metrics),
            "--table",
            str(table),
            "--metrics",
            str(metrics_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert table.exists()
    assert metrics_path.exists()
    payload = json.loads(result.stdout)
    assert payload["test_split_used"] is False


def test_agentdojo_early_warning_metrics_compute_lead_time_and_false_alerts(tmp_path):
    scores, val, calibration_metrics = _write_inputs(tmp_path)
    table = tmp_path / "table5.csv"
    metrics_path = tmp_path / "metrics.json"

    metrics = make_agentdojo_early_warning_metrics(scores, val, calibration_metrics, table, metrics_path)

    assert metrics["risky_episode_count"] == 3
    assert metrics["safe_episode_count"] == 1
    assert metrics["alert_before_risk_count"] == 1
    assert metrics["alert_before_risk_rate"] == pytest.approx(1 / 3)
    assert metrics["positive_lead_time_contained_count"] == 1
    assert metrics["no_window_just_in_time_contained_count"] == 1
    assert metrics["late_just_in_time_after_missed_window_count"] == 0
    assert metrics["just_in_time_contained_count"] == 1
    assert metrics["operational_contained_count"] == 2
    assert metrics["operational_contained_rate"] == pytest.approx(2 / 3)
    assert metrics["strict_positive_lead_time_rate"] == pytest.approx(1 / 3)
    assert metrics["opportunity_adjusted_positive_lead_time_rate"] == 0.5
    assert metrics["no_multi_step_pre_risk_window_count"] == 1
    assert metrics["no_multi_step_pre_risk_window_rate"] == pytest.approx(1 / 3)
    assert metrics["missed_with_pre_risk_window_count"] == 1
    assert metrics["contained_incident_proxy"] == pytest.approx(2 / 3)
    assert metrics["uncontained_incident_proxy"] == pytest.approx(1 / 3)
    assert metrics["missed_risk_episode_count"] == 1
    assert metrics["false_alert_episode_count"] == 1
    assert metrics["false_alert_episode_rate"] == 1.0
    assert metrics["mean_lead_time_steps"] == 0.5
    assert metrics["median_lead_time_steps"] == 0.5


def test_agentdojo_early_warning_table_has_threshold_rows(tmp_path):
    scores, val, calibration_metrics = _write_inputs(tmp_path)
    table = tmp_path / "table5.csv"
    metrics_path = tmp_path / "metrics.json"

    make_agentdojo_early_warning_metrics(scores, val, calibration_metrics, table, metrics_path)

    table_df = pd.read_csv(table)
    assert table_df["threshold"].tolist() == ["verify", "alert", "block"]
    assert set(table_df["split"]) == {"validation"}
    assert set(table_df["test_split_used"]) == {False}


def test_agentdojo_early_warning_refuses_test_split_metrics(tmp_path):
    scores, val, calibration_metrics = _write_inputs(tmp_path, test_split_used=True)

    try:
        make_agentdojo_early_warning_metrics(
            scores,
            val,
            calibration_metrics,
            tmp_path / "table5.csv",
            tmp_path / "metrics.json",
        )
    except ValueError as exc:
        assert "test-split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for test-split calibration metrics")
