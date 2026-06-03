import json

import pandas as pd

from scripts.backfill_agentdojo_metadata import backfill_agentdojo_metadata


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dataset(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"episode_id": "suite:user_task_0:none:none", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0},
        {"episode_id": "suite:user_task_1:none:none", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 1},
        {"episode_id": "suite:user_task_2:none:none", "step_id": 1, "hook_type": "pre_step", "future_risk_label": 0},
    ]).to_csv(path, index=False)


def _write_manifest(path, dataset):
    _write_json(
        path,
        {
            "input": str(dataset),
            "episode_ids": {
                "train": ["suite:user_task_0:none:none"],
                "val": ["suite:user_task_1:none:none"],
                "test": ["suite:user_task_2:none:none"],
            },
            "outputs": {"manifest": str(path)},
            "warnings": [],
        },
    )


def test_backfill_agentdojo_metadata_updates_csvs_and_manifest(tmp_path):
    dataset = tmp_path / "full.csv"
    split_train = tmp_path / "splits/agentdojo_train.csv"
    split_val = tmp_path / "splits/agentdojo_val.csv"
    split_test = tmp_path / "splits/agentdojo_test.csv"
    _write_dataset(dataset)
    split_train.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(dataset).iloc[[0]].to_csv(split_train, index=False)
    pd.read_csv(dataset).iloc[[1]].to_csv(split_val, index=False)
    pd.read_csv(dataset).iloc[[2]].to_csv(split_test, index=False)

    manifest = tmp_path / "splits/split_manifest.json"
    _write_manifest(manifest, dataset)
    summary = tmp_path / "batch/run_summary.json"
    _write_json(
        summary,
        {
            "suite": "suite",
            "task_results": [
                {"task_id": "user_task_0", "status": "ok", "utility": True, "security": True},
                {"task_id": "user_task_1", "status": "ok", "utility": False, "security": True},
                {"task_id": "user_task_2", "status": "stopped"},
            ],
        },
    )
    audit = tmp_path / "audit.json"
    _write_json(audit, {"effective_after_recovery": {"budget_limited_tasks": ["suite:user_task_2"]}})

    result = backfill_agentdojo_metadata(
        dataset_paths=[dataset, split_train, split_val, split_test],
        combined_manifest=manifest,
        disagreement_manifest=tmp_path / "missing_manifest.json",
        audit_path=audit,
        json_report=tmp_path / "report.json",
        csv_report=tmp_path / "report.csv",
        summary_patterns=[str(summary)],
    )

    updated = pd.read_csv(dataset)
    by_episode = updated.set_index("episode_id")
    assert bool(by_episode.loc["suite:user_task_0:none:none", "utility"]) is True
    assert bool(by_episode.loc["suite:user_task_0:none:none", "security"]) is True
    assert by_episode.loc["suite:user_task_0:none:none", "metadata_status"] == "known_true"
    assert bool(by_episode.loc["suite:user_task_1:none:none", "utility"]) is False
    assert by_episode.loc["suite:user_task_1:none:none", "metadata_status"] == "known_false"
    assert by_episode.loc["suite:user_task_2:none:none", "metadata_status"] == "unknown_not_evaluated"
    assert result["totals"]["conflict_episodes"] == 0

    updated_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated_manifest["episode_ids"]["train"] == ["suite:user_task_0:none:none"]
    assert updated_manifest["utility_counts"]["train"]["true"] == 1
    assert updated_manifest["utility_counts"]["val"]["false"] == 1
    assert updated_manifest["utility_counts"]["test"]["unknown_not_evaluated"] == 1
    assert updated_manifest["security_counts"]["test"]["unknown_not_evaluated"] == 1


def test_backfill_agentdojo_metadata_marks_conflicts(tmp_path):
    dataset = tmp_path / "full.csv"
    _write_dataset(dataset)
    summary_a = tmp_path / "a/run_summary.json"
    summary_b = tmp_path / "b/run_summary.json"
    _write_json(summary_a, {"suite": "suite", "task_results": [{"task_id": "user_task_0", "status": "ok", "utility": True, "security": True}]})
    _write_json(summary_b, {"suite": "suite", "task_results": [{"task_id": "user_task_0", "status": "ok", "utility": False, "security": True}]})

    result = backfill_agentdojo_metadata(
        dataset_paths=[dataset],
        combined_manifest=tmp_path / "missing_manifest.json",
        disagreement_manifest=tmp_path / "missing_manifest2.json",
        audit_path=tmp_path / "missing_audit.json",
        json_report=tmp_path / "report.json",
        csv_report=tmp_path / "report.csv",
        summary_patterns=[str(summary_a), str(summary_b)],
    )

    updated = pd.read_csv(dataset)
    assert updated.loc[0, "metadata_status"] == "conflict"
    assert result["totals"]["conflict_episodes"] == 1
