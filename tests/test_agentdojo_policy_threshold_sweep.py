import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.sweep_agentdojo_policy_thresholds import (
    _score_setting,
    sweep_agentdojo_policy_thresholds,
)


SCRIPT = "scripts/sweep_agentdojo_policy_thresholds.py"


def _write_scores(path):
    rows = [
        {"episode_id": "risky_early", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "thermometer_score": 45.0},
        {"episode_id": "risky_early", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "thermometer_score": 90.0},
        {"episode_id": "risky_missed", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "thermometer_score": 15.0},
        {"episode_id": "risky_missed", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "thermometer_score": 25.0},
        {"episode_id": "risky_jit", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "thermometer_score": 35.0},
        {"episode_id": "safe_false", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "thermometer_score": 25.0},
        {"episode_id": "safe_ok", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "thermometer_score": 5.0},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_val(path):
    rows = [
        {"episode_id": "risky_early", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_early", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_missed", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_missed", "step_id": 2, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 2},
        {"episode_id": "risky_jit", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1, "t_risk": 1},
        {"episode_id": "safe_false", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "t_risk": ""},
        {"episode_id": "safe_ok", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "t_risk": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_early_warning(path, test_split_used=False):
    path.write_text(json.dumps({"test_split_used": test_split_used, "warnings": []}), encoding="utf-8")


def _write_inputs(tmp_path, test_split_used=False):
    scores = tmp_path / "scores.csv"
    val = tmp_path / "val.csv"
    early_warning = tmp_path / "early_warning.json"
    _write_scores(scores)
    _write_val(val)
    _write_early_warning(early_warning, test_split_used=test_split_used)
    return scores, val, early_warning


def test_policy_threshold_sweep_cli_writes_table_and_metrics(tmp_path):
    scores, val, early_warning = _write_inputs(tmp_path)
    table = tmp_path / "table6.csv"
    metrics_path = tmp_path / "metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--scores",
            str(scores),
            "--val",
            str(val),
            "--early-warning",
            str(early_warning),
            "--table",
            str(table),
            "--metrics",
            str(metrics_path),
            "--verify-thresholds",
            "20,40",
            "--alert-thresholds",
            "40",
            "--block-thresholds",
            "80",
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
    assert payload["recommended_policy"]


def test_threshold_order_validation():
    df = pd.DataFrame([
        {"episode_id": "e", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0, "thermometer_score": 10.0}
    ])

    try:
        _score_setting(df, verify_threshold=50, alert_threshold=40, block_threshold=80)
    except ValueError as exc:
        assert "verify_threshold <= alert_threshold <= block_threshold" in str(exc)
    else:
        raise AssertionError("Expected invalid threshold order to raise ValueError")


def test_threshold_sweep_counts_contained_missed_and_false_alerts(tmp_path):
    scores, val, early_warning = _write_inputs(tmp_path)
    table = tmp_path / "table6.csv"
    metrics_path = tmp_path / "metrics.json"

    sweep_agentdojo_policy_thresholds(
        scores,
        val,
        early_warning,
        table,
        metrics_path,
        verify_thresholds=[30],
        alert_thresholds=[60],
        block_thresholds=[80],
    )

    row = pd.read_csv(table).iloc[0]
    assert int(row["risky_episode_count"]) == 3
    assert int(row["safe_episode_count"]) == 2
    assert int(row["alert_before_risk_count"]) == 1
    assert float(row["alert_before_risk_rate"]) == pytest.approx(1 / 3)
    assert int(row["positive_lead_time_contained_count"]) == 1
    assert int(row["no_window_just_in_time_contained_count"]) == 1
    assert int(row["late_just_in_time_after_missed_window_count"]) == 0
    assert int(row["just_in_time_contained_count"]) == 1
    assert int(row["operational_contained_count"]) == 2
    assert float(row["operational_contained_rate"]) == pytest.approx(2 / 3)
    assert float(row["opportunity_adjusted_positive_lead_time_rate"]) == 0.5
    assert int(row["no_multi_step_pre_risk_window_count"]) == 1
    assert int(row["missed_with_pre_risk_window_count"]) == 1
    assert int(row["missed_risk_episode_count"]) == 1
    assert int(row["false_alert_episode_count"]) == 0
    assert float(row["false_alert_episode_rate"]) == 0.0


def test_threshold_sweep_recommended_policy_respects_false_alert_cap(tmp_path):
    scores, val, early_warning = _write_inputs(tmp_path)

    metrics = sweep_agentdojo_policy_thresholds(
        scores,
        val,
        early_warning,
        tmp_path / "table6.csv",
        tmp_path / "metrics.json",
        verify_thresholds=[20, 30],
        alert_thresholds=[40],
        block_thresholds=[80],
    )

    recommended = metrics["recommended_policy"]
    assert recommended["verify_threshold"] == 30.0
    assert recommended["false_alert_episode_rate"] <= 0.25


def test_threshold_sweep_refuses_test_split_metrics(tmp_path):
    scores, val, early_warning = _write_inputs(tmp_path, test_split_used=True)

    try:
        sweep_agentdojo_policy_thresholds(
            scores,
            val,
            early_warning,
            tmp_path / "table6.csv",
            tmp_path / "metrics.json",
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for test split metrics")
