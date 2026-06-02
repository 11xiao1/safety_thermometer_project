import pandas as pd

from scripts import cleanup_agentdojo_repair_artifacts as cleanup


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(episode_id, step_id=1, split_label=0):
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "hook_type": "pre_step",
        "source_batch": "workspace_round1",
        "future_risk_label": split_label,
        "oracle_violation": 0,
        "risk_score": 10.0,
        "max_risk_score_so_far": 10.0,
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


def _write_canonical_outputs(root):
    round1_rows = [_row(f"workspace:user_task_{task_id}:none:none", task_id + 1) for task_id in range(5)]
    _write_csv(root / "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv", round1_rows)
    (root / "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl").write_text("{}", encoding="utf-8")
    combined = root / "outputs/agentdojo_multisuite_disagreement/agentdojo_multisuite_disagreement_prefix_dataset.csv"
    _write_csv(combined, round1_rows)
    split_dir = root / "outputs/agentdojo_multisuite_disagreement/splits"
    _write_csv(split_dir / "agentdojo_train.csv", round1_rows[:3])
    _write_csv(split_dir / "agentdojo_val.csv", round1_rows[3:4])
    _write_csv(split_dir / "agentdojo_test.csv", round1_rows[4:])
    (split_dir / "split_manifest.json").write_text("{}", encoding="utf-8")


def test_cleanup_dry_run_validates_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    _write_canonical_outputs(tmp_path)
    backup = tmp_path / "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.corrupt_backup.jsonl"
    backup.write_text("corrupt", encoding="utf-8")
    clean_dir = tmp_path / "outputs/agentdojo_multisuite_disagreement_clean"
    clean_dir.mkdir(parents=True)

    result = cleanup.cleanup_agentdojo_repair_artifacts(apply=False)

    assert result["validation"]["passed"] is True
    assert result["would_delete"]
    assert backup.exists()
    assert clean_dir.exists()


def test_cleanup_apply_deletes_only_temporary_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    _write_canonical_outputs(tmp_path)
    backup = tmp_path / "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.corrupt_backup.jsonl"
    backup.write_text("corrupt", encoding="utf-8")
    clean_dir = tmp_path / "outputs/agentdojo_multisuite_disagreement_clean"
    clean_dir.mkdir(parents=True)

    result = cleanup.cleanup_agentdojo_repair_artifacts(apply=True)

    assert result["validation"]["passed"] is True
    assert str(backup) in result["deleted"]
    assert str(clean_dir) in result["deleted"]
    assert not backup.exists()
    assert not clean_dir.exists()
    assert (tmp_path / "outputs/agentdojo_multisuite_disagreement/agentdojo_multisuite_disagreement_prefix_dataset.csv").exists()
