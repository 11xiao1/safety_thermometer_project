import json

import pandas as pd

from scripts.regenerate_agentdojo_prefix_with_variance_disagreement import (
    VARIANCE_COLUMNS,
    regenerate_agentdojo_prefix_with_variance_disagreement,
)


HEURISTIC_COLUMNS = [
    "f_intent_tool_mismatch",
    "f_plan_action_mismatch",
    "f_sensitive_tool_without_need",
    "f_external_send_without_request",
    "f_observation_error",
    "f_self_check_risk_mismatch",
    "f_fallback_after_error",
    "cumulative_observation_error_count",
    "cumulative_disagreement_count",
    "max_disagreement_score_so_far",
]


def _event(**overrides):
    defaults = {
        "episode_id": "workspace:user_task_0:none:none",
        "step_id": 1,
        "hook_type": "pre_step",
        "user_instruction": "Summarize the public notes.",
        "plan_summary": "Read only and keep it safe.",
        "proposed_tool": "send_email",
        "tool_args": {"to": "external@example.com", "body": "private token"},
        "observation": "Permission denied.",
        "self_check": {"risk": "low"},
        "notes": "external send proposed",
    }
    defaults.update(overrides)
    return defaults


def _write_trace(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _base_row(episode_id, source_batch, label):
    return {
        "source_suite": episode_id.split(":")[0],
        "source_batch": source_batch,
        "episode_id": episode_id,
        "step_id": 1,
        "hook_type": "pre_step",
        "f_confirm_required": 0,
        "f_confirm_seen": 0,
        "f_state_modifying": 0,
        "f_irreversible": 0,
        "f_sensitive_access": 0,
        "f_external_send": 0,
        "f_fallback_count": 0,
        "f_unprotected_fallback": 0,
        "f_plan_action_gap": 0.0,
        "f_self_check_gap": 0.0,
        "f_repeated_failures": 0,
        "f_risk_keyword_count": 0,
        "cumulative_state_modifying_count": 0,
        "cumulative_irreversible_count": 0,
        "cumulative_sensitive_access_count": 0,
        "cumulative_external_send_count": 0,
        "cumulative_fallback_count": 0,
        "confirmation_seen_so_far": 0,
        "max_risk_score_so_far": 0.0,
        "oracle_violation": 0,
        "oracle_rules": "",
        "risk_score": 0.0,
        "policy": "continue",
        "future_risk_label": label,
        "future_severity": 0.0,
        "t_risk": "",
        "lead_time_if_alert_now": "",
        "utility": True,
        "security": True,
    }


def test_regenerate_variance_disagreement_writes_two_aligned_variants(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    second_episode = "workspace:user_task_1:none:none"
    _write_trace(
        trace_path,
        [
            _event(),
            _event(
                hook_type="final",
                step_id=2,
                proposed_tool=None,
                tool_args={},
                self_check={},
                notes="final completed utility false security true",
            ),
            _event(
                episode_id=second_episode,
                proposed_tool="read_file",
                tool_args={"path": "public.md"},
                observation="Read succeeded.",
                notes=None,
            ),
        ],
    )
    canonical = tmp_path / "canonical.csv"
    canonical_df = pd.DataFrame(
        [
            _base_row("workspace:user_task_0:none:none", "workspace_round1", 1),
            _base_row(second_episode, "workspace_round1", 0),
        ]
    )
    canonical_df.to_csv(canonical, index=False)
    heuristic = tmp_path / "heuristic.csv"
    heuristic_df = canonical_df.copy()
    for column in HEURISTIC_COLUMNS:
        heuristic_df[column] = 0
    heuristic_df.to_csv(heuristic, index=False)
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "episode_ids": {
                    "train": ["workspace:user_task_0:none:none"],
                    "val": [second_episode],
                    "test": [],
                }
            }
        ),
        encoding="utf-8",
    )

    result = regenerate_agentdojo_prefix_with_variance_disagreement(
        canonical_path=canonical,
        heuristic_path=heuristic,
        audit_path=tmp_path / "unused_audit.json",
        prior_manifest_path=prior_manifest,
        variance_dir=tmp_path / "variance",
        heuristic_variance_dir=tmp_path / "heuristic_variance",
        trace_sources=[
            {
                "trace": str(trace_path),
                "source_suite": "workspace",
                "source_batch": "workspace_round1",
                "source_group": "unit",
            }
        ],
    )

    variance = pd.read_csv(tmp_path / "variance" / "agentdojo_full_variance_disagreement_prefix_dataset.csv")
    heuristic_variance = pd.read_csv(
        tmp_path / "heuristic_variance" / "agentdojo_full_heuristic_variance_disagreement_prefix_dataset.csv"
    )
    variance_manifest = json.loads((tmp_path / "variance" / "splits" / "split_manifest.json").read_text())
    feature_manifest = json.loads((tmp_path / "heuristic_variance" / "feature_manifest.json").read_text())

    assert result["status"] == "ok"
    assert result["outputs"]["variance_only"]["row_count"] == 2
    assert all(column in variance.columns for column in VARIANCE_COLUMNS)
    assert not any(column in variance.columns for column in HEURISTIC_COLUMNS)
    assert all(column in heuristic_variance.columns for column in HEURISTIC_COLUMNS)
    assert all(column in heuristic_variance.columns for column in VARIANCE_COLUMNS)
    assert variance_manifest["episode_ids"]["train"] == ["workspace:user_task_0:none:none"]
    assert variance_manifest["episode_ids"]["val"] == [second_episode]
    assert feature_manifest["heuristic_features"] == HEURISTIC_COLUMNS
    assert feature_manifest["variance_features"]
    assert feature_manifest["notes_leakage_audit"]["final_event_notes_used_for_stream_q"] is False
    assert feature_manifest["notes_leakage_audit"]["outcome_term_hits"]["utility"] == 1
