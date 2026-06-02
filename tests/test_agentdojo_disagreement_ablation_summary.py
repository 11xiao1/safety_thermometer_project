import json

import pandas as pd

from scripts.summarize_agentdojo_disagreement_ablation import (
    SUMMARY_COLUMNS,
    summarize_agentdojo_disagreement_ablation,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(root, score_column="risk_score_logistic", auroc=0.7, operational_rate=0.5):
    _write_json(
        root / "agentdojo_split_risk_estimator_metrics.json",
        {
            "test_split_used": False,
            "logistic": {"auroc": auroc, "auprc": auroc + 0.1, "f1_at_50": 0.6},
            "random_forest": {"auroc": 0.5, "auprc": 0.5, "f1_at_50": 0.5},
            "hist_gradient_boosting": {"auroc": 0.4, "auprc": 0.4, "f1_at_50": 0.4},
        },
    )
    _write_json(
        root / "agentdojo_calibration_metrics_val.json",
        {
            "score_column": score_column,
            "selected_calibration_method": "isotonic",
            "brier_score_isotonic": 0.1,
            "ece_isotonic": 0.02,
            "test_split_used": False,
        },
    )
    _write_json(
        root / "agentdojo_early_warning_metrics_val.json",
        {
            "pre_risk_auroc": 0.8,
            "pre_risk_auprc": 0.75,
            "test_split_used": False,
        },
    )
    _write_json(
        root / "agentdojo_threshold_sweep_val.json",
        {
            "test_split_used": False,
            "recommended_policy": {
                "positive_lead_time_contained_count": 1,
                "strict_positive_lead_time_rate": 0.2,
                "no_window_just_in_time_contained_count": 2,
                "operational_contained_count": 3,
                "operational_contained_rate": operational_rate,
                "opportunity_adjusted_positive_lead_time_rate": 0.33,
                "missed_with_pre_risk_window_count": 2,
                "false_alert_episode_count": 1,
                "false_alert_episode_rate": 0.25,
                "verify_threshold": 30.0,
                "alert_threshold": 60.0,
                "block_threshold": 70.0,
                "test_split_used": False,
            },
        },
    )


def test_ablation_summary_writes_required_outputs_and_deltas(tmp_path):
    without_dir = tmp_path / "without"
    with_dir = tmp_path / "with"
    _write_inputs(without_dir, auroc=0.7, operational_rate=0.5)
    _write_inputs(with_dir, auroc=0.9, operational_rate=0.8)

    result = summarize_agentdojo_disagreement_ablation(
        without_dir=without_dir,
        with_dir=with_dir,
        csv_out=tmp_path / "summary.csv",
        json_out=tmp_path / "summary.json",
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary.columns.tolist() == SUMMARY_COLUMNS
    assert summary["setting"].tolist() == ["without_disagreement", "with_disagreement"]
    assert result["deltas_with_minus_without"]["AUROC"] == 0.2
    assert result["deltas_with_minus_without"]["operational_contained_rate"] == 0.30000000000000004
    assert result["test_split_used"] is False
    assert (tmp_path / "summary.json").exists()
