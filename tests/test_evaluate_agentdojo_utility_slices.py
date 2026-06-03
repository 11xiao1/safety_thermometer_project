import json

import pandas as pd

from scripts.evaluate_agentdojo_utility_slices import evaluate_agentdojo_utility_slices


def _write_inputs(root, setting):
    scores = root / setting / "scores.csv"
    val = root / setting / "val.csv"
    sweep = root / setting / "sweep.json"
    scores.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "episode_id": "ep_true",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 0,
            "thermometer_score": 80.0,
        },
        {
            "episode_id": "ep_false",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 0,
            "oracle_violation": 0,
            "thermometer_score": 20.0,
        },
        {
            "episode_id": "ep_unknown",
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": 1,
            "oracle_violation": 1,
            "thermometer_score": 60.0,
        },
    ]
    pd.DataFrame(rows).assign(policy="watch").to_csv(scores, index=False)
    pd.DataFrame([
        {
            **row,
            "utility": utility,
            "security": security,
            "metadata_status": metadata_status,
            "source_suite": "workspace",
            "source_batch": setting,
            "t_risk": 2 if row["future_risk_label"] else "",
            "lead_time_if_alert_now": 1 if row["future_risk_label"] else "",
            "future_severity": float(row["future_risk_label"]),
        }
        for row, utility, security, metadata_status in [
            (rows[0], True, True, "known_true"),
            (rows[1], False, True, "known_false"),
            (rows[2], "", "", "unknown_no_summary"),
        ]
    ]).to_csv(val, index=False)
    sweep.write_text(
        json.dumps(
            {
                "recommended_policy": {
                    "verify_threshold": 30.0,
                    "alert_threshold": 50.0,
                    "block_threshold": 75.0,
                },
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    return scores, val, sweep


def test_evaluate_agentdojo_utility_slices_writes_all_setting_slice_rows(tmp_path):
    without_scores, without_val, without_sweep = _write_inputs(tmp_path, "without")
    with_scores, with_val, with_sweep = _write_inputs(tmp_path, "with")

    result = evaluate_agentdojo_utility_slices(
        without_scores=without_scores,
        without_val=without_val,
        without_sweep=without_sweep,
        with_scores=with_scores,
        with_val=with_val,
        with_sweep=with_sweep,
        csv_out=tmp_path / "slice_eval.csv",
        json_out=tmp_path / "slice_eval.json",
    )

    assert result["status"] == "ok"
    assert len(result["rows"]) == 8
    assert {(row["setting"], row["slice"]) for row in result["rows"]} == {
        (setting, slice_name)
        for setting in ["without_disagreement", "with_disagreement"]
        for slice_name in ["all", "utility=true", "utility=false", "unknown_no_summary"]
    }
    unknown = next(row for row in result["rows"] if row["setting"] == "without_disagreement" and row["slice"] == "unknown_no_summary")
    assert unknown["episode_count"] == 1
    assert unknown["future_risk_label_counts"] == {"1": 1}
    assert (tmp_path / "slice_eval.csv").exists()
    assert (tmp_path / "slice_eval.json").exists()
