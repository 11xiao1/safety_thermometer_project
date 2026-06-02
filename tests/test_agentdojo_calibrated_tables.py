import json
import subprocess
import sys

import pandas as pd

from scripts.make_agentdojo_calibrated_tables import make_agentdojo_calibrated_tables


SCRIPT = "scripts/make_agentdojo_calibrated_tables.py"
TABLE3_COLUMNS = [
    "score_name",
    "split",
    "row_count",
    "label_counts",
    "AUROC",
    "AUPRC",
    "F1@50",
    "Brier score",
    "ECE",
    "mean_score_safe",
    "mean_score_risky",
    "calibration_method",
    "test_split_used",
    "limitation",
]
TABLE4_COLUMNS = [
    "split",
    "row_count",
    "continue",
    "watch",
    "verify",
    "alert",
    "block",
    "alert_or_block_count",
    "alert_or_block_rate",
    "risky_prefixes",
    "safe_prefixes",
    "risky_alert_or_block_count",
    "safe_alert_or_block_count",
    "false_alert_proxy",
    "missed_risky_proxy",
    "test_split_used",
    "limitation",
]


def _write_scores(path):
    rows = [
        {
            "episode_id": "episode_0",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 0,
            "oracle_violation": 0,
            "raw_score": 10.0,
            "calibrated_score_platt": 15.0,
            "calibrated_score_isotonic": 5.0,
            "thermometer_score": 5.0,
            "policy": "continue",
        },
        {
            "episode_id": "episode_1",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 0,
            "oracle_violation": 0,
            "raw_score": 35.0,
            "calibrated_score_platt": 25.0,
            "calibrated_score_isotonic": 30.0,
            "thermometer_score": 30.0,
            "policy": "watch",
        },
        {
            "episode_id": "episode_2",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "raw_score": 70.0,
            "calibrated_score_platt": 75.0,
            "calibrated_score_isotonic": 72.0,
            "thermometer_score": 72.0,
            "policy": "alert",
        },
        {
            "episode_id": "episode_3",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "raw_score": 90.0,
            "calibrated_score_platt": 85.0,
            "calibrated_score_isotonic": 95.0,
            "thermometer_score": 95.0,
            "policy": "block",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path, test_split_used=False):
    scores = tmp_path / "scores.csv"
    calibration_metrics = tmp_path / "calibration_metrics.json"
    risk_metrics = tmp_path / "risk_metrics.json"
    _write_scores(scores)
    _write_json(
        calibration_metrics,
        {
            "test_split_used": test_split_used,
            "selected_calibration_method": "isotonic",
            "calibration_methods_used": ["identity", "platt", "isotonic"],
        },
    )
    _write_json(risk_metrics, {"test_split_used": False})
    return scores, calibration_metrics, risk_metrics


def test_agentdojo_calibrated_tables_cli_writes_outputs(tmp_path):
    scores, calibration_metrics, risk_metrics = _write_inputs(tmp_path)
    table3 = tmp_path / "table3.csv"
    table4 = tmp_path / "table4.csv"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--scores",
            str(scores),
            "--calibration-metrics",
            str(calibration_metrics),
            "--risk-metrics",
            str(risk_metrics),
            "--table3",
            str(table3),
            "--table4",
            str(table4),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert table3.exists()
    assert table4.exists()
    payload = json.loads(result.stdout)
    assert payload["test_split_used"] is False


def test_agentdojo_calibrated_tables_have_required_columns_and_rows(tmp_path):
    scores, calibration_metrics, risk_metrics = _write_inputs(tmp_path)
    table3 = tmp_path / "table3.csv"
    table4 = tmp_path / "table4.csv"

    make_agentdojo_calibrated_tables(scores, calibration_metrics, risk_metrics, table3, table4)

    table3_df = pd.read_csv(table3)
    table4_df = pd.read_csv(table4)
    assert table3_df.columns.tolist() == TABLE3_COLUMNS
    assert table4_df.columns.tolist() == TABLE4_COLUMNS
    assert table3_df["score_name"].tolist() == [
        "raw_hist_gradient_boosting",
        "calibrated_platt",
        "calibrated_isotonic",
        "selected_thermometer_score",
    ]
    assert set(table3_df["split"]) == {"validation"}
    assert table4_df.loc[0, "split"] == "validation"


def test_agentdojo_calibrated_table4_policy_counts_and_proxy_rates(tmp_path):
    scores, calibration_metrics, risk_metrics = _write_inputs(tmp_path)
    table3 = tmp_path / "table3.csv"
    table4 = tmp_path / "table4.csv"

    make_agentdojo_calibrated_tables(scores, calibration_metrics, risk_metrics, table3, table4)

    row = pd.read_csv(table4).iloc[0]
    assert int(row["row_count"]) == 4
    assert int(row["continue"]) == 1
    assert int(row["watch"]) == 1
    assert int(row["alert"]) == 1
    assert int(row["block"]) == 1
    assert int(row["alert_or_block_count"]) == 2
    assert float(row["alert_or_block_rate"]) == 0.5
    assert int(row["risky_alert_or_block_count"]) == 2
    assert int(row["safe_alert_or_block_count"]) == 0
    assert float(row["false_alert_proxy"]) == 0.0
    assert float(row["missed_risky_proxy"]) == 0.0


def test_agentdojo_calibrated_tables_refuse_test_split_metrics(tmp_path):
    scores, calibration_metrics, risk_metrics = _write_inputs(tmp_path, test_split_used=True)

    try:
        make_agentdojo_calibrated_tables(
            scores,
            calibration_metrics,
            risk_metrics,
            tmp_path / "table3.csv",
            tmp_path / "table4.csv",
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for test split metrics")
