import json

import pandas as pd

from scripts.audit_agentdojo_full_datasets import (
    DISAGREEMENT_COLUMNS,
    audit_agentdojo_full_datasets,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rows(include_disagreement):
    rows = []
    episodes = [
        ("ep_train", "train", 1, True, True),
        ("ep_val", "val", 0, False, True),
        ("ep_test", "test", 1, "", ""),
    ]
    for episode_id, split, label, utility, security in episodes:
        row = {
            "episode_id": episode_id,
            "step_id": 1,
            "hook_type": "pre_step",
            "source_suite": "workspace" if split != "val" else "slack",
            "source_batch": f"{split}_batch",
            "future_risk_label": label,
            "future_severity": float(label),
            "t_risk": 1 if label else "",
            "lead_time_if_alert_now": 0 if label else "",
            "oracle_violation": label,
            "risk_score": 75.0 if label else 0.0,
            "utility": utility,
            "security": security,
            "metadata_status": "unknown_no_summary" if utility == "" else ("known_false" if utility is False else "known_true"),
            "metadata_source_batch": "" if utility == "" else f"{split}_batch",
        }
        if include_disagreement:
            for column in DISAGREEMENT_COLUMNS:
                row[column] = 0
        rows.append(row)
    return rows


def _manifest(path):
    payload = {
        "episode_ids": {
            "train": ["ep_train"],
            "val": ["ep_val"],
            "test": ["ep_test"],
        },
        "outputs": {"manifest": str(path)},
    }
    _write_json(path, payload)


def test_audit_agentdojo_full_datasets_reports_readiness_and_unknown_group(tmp_path):
    combined = tmp_path / "combined.csv"
    disagreement = tmp_path / "disagreement.csv"
    combined_manifest = tmp_path / "combined_manifest.json"
    disagreement_manifest = tmp_path / "disagreement_manifest.json"
    backfill = tmp_path / "backfill.json"
    pd.DataFrame(_rows(include_disagreement=False)).to_csv(combined, index=False)
    pd.DataFrame(_rows(include_disagreement=True)).to_csv(disagreement, index=False)
    _manifest(combined_manifest)
    _manifest(disagreement_manifest)
    _write_json(backfill, {"totals": {"remaining_unknown_episodes": 1}})

    result = audit_agentdojo_full_datasets(
        combined_path=combined,
        disagreement_path=disagreement,
        combined_manifest_path=combined_manifest,
        disagreement_manifest_path=disagreement_manifest,
        backfill_report_path=backfill,
        json_out=tmp_path / "audit.json",
        csv_out=tmp_path / "audit.csv",
    )

    assert result["episode_sets_match"] is True
    assert result["split_assignment_matches"] is True
    assert result["episode_overlap_check"]["passed"] is True
    assert result["semantic_duplicate_rows"]["no_disagreement"] == 0
    assert result["column_checks"]["required_disagreement_columns_exist"] is True
    assert result["column_checks"]["no_disagreement_has_disagreement_columns"] is False
    assert result["splits"]["no_disagreement"]["test"]["utility_counts"]["unknown_no_summary"] == 1
    assert result["utility_subset_distributions"]["no_disagreement"]["unknown_no_summary"]["future_risk_label_counts"] == {"1": 1}
    assert result["leakage_audit"]["no_disagreement"]["utility_security_diagnostic_only"] is True
    assert result["leakage_audit"]["no_disagreement"]["oracle_violation_diagnostic_only"] is True
    assert result["training_readiness"]["unknown_no_summary_blocks_training"] is False
    assert (tmp_path / "audit.csv").exists()
