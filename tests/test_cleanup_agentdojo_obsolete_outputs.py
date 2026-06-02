import pandas as pd
import pytest

from scripts import cleanup_agentdojo_obsolete_outputs as cleanup


def _row(episode_id, step_id=1):
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "hook_type": "pre_step",
        "future_risk_label": 0,
        "source_batch": "workspace_round1",
        "f_intent_tool_mismatch": 0,
        "f_plan_action_mismatch": 0,
        "f_sensitive_tool_without_need": 0,
        "f_external_send_without_request": 0,
        "f_observation_error": 0,
        "f_self_check_risk_mismatch": 0,
        "f_fallback_after_error": 0,
        "cumulative_observation_error_count": 0,
        "cumulative_disagreement_count": 0,
        "max_disagreement_score_so_far": 0.0,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_canonical_outputs(root):
    combined = root / "outputs/agentdojo_multisuite_combined"
    disagreement = root / "outputs/agentdojo_multisuite_disagreement"
    rows = [_row(f"workspace:user_task_{task_id}:none:none", task_id + 1) for task_id in range(5)]
    _write_csv(combined / "agentdojo_multisuite_combined_prefix_dataset.csv", rows)
    _write_csv(combined / "splits/agentdojo_train.csv", rows[:3])
    _write_csv(combined / "splits/agentdojo_val.csv", rows[3:4])
    _write_csv(combined / "splits/agentdojo_test.csv", rows[4:])
    (combined / "splits/split_manifest.json").write_text("{}", encoding="utf-8")
    _write_csv(disagreement / "agentdojo_multisuite_disagreement_prefix_dataset.csv", rows)
    _write_csv(disagreement / "splits/agentdojo_train.csv", rows[:3])
    _write_csv(disagreement / "splits/agentdojo_val.csv", rows[3:4])
    _write_csv(disagreement / "splits/agentdojo_test.csv", rows[4:])
    (disagreement / "splits/split_manifest.json").write_text("{}", encoding="utf-8")
    _write_csv(root / "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv", rows)


def test_cleanup_obsolete_outputs_dry_run_writes_plan_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    _write_canonical_outputs(tmp_path)
    obsolete = tmp_path / "outputs/agentdojo_smoke_trace.jsonl"
    obsolete.parent.mkdir(parents=True, exist_ok=True)
    obsolete.write_text("stale", encoding="utf-8")

    plan = cleanup.cleanup_agentdojo_obsolete_outputs(apply=False)

    assert plan["validation"]["passed"] is True
    assert str(obsolete) in plan["would_delete"]
    assert obsolete.exists()
    assert (tmp_path / "outputs/agentdojo_cleanup_plan.json").exists()


def test_cleanup_obsolete_outputs_apply_deletes_only_existing_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    _write_canonical_outputs(tmp_path)
    obsolete_dir = tmp_path / "outputs/splits"
    obsolete_dir.mkdir(parents=True)
    protected = tmp_path / "scripts/keep.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("print('keep')", encoding="utf-8")

    plan = cleanup.cleanup_agentdojo_obsolete_outputs(apply=True)

    assert plan["validation"]["passed"] is True
    assert str(obsolete_dir) in plan["deleted"]
    assert not obsolete_dir.exists()
    assert protected.exists()


def test_cleanup_refuses_paths_outside_candidate_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        cleanup._validate_candidate_path(tmp_path / "outputs/not_listed.csv")
