import json

import pandas as pd

from scripts.audit_agentdojo_trace_provenance import (
    audit_agentdojo_trace_provenance,
    repair_round1_merged_trace,
)


def _event(episode_id):
    return {
        "episode_id": episode_id,
        "step_id": 1,
        "hook_type": "pre_step",
        "user_instruction": "Summarize notes.",
    }


def _write_trace(path, episode_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(_event(episode_id)) for episode_id in episode_ids) + "\n",
        encoding="utf-8",
    )


def test_trace_provenance_audit_flags_round1_file_containing_round2_tasks(tmp_path):
    round1_merged = tmp_path / "agentdojo_mini_batch" / "merged" / "workspace_mini_batch_trace.jsonl"
    round2_merged = tmp_path / "agentdojo_mini_batch_round2" / "merged" / "workspace_mini_batch_round2_trace.jsonl"
    round1_dir = tmp_path / "agentdojo_mini_batch" / "traces"
    _write_trace(round1_merged, ["workspace:user_task_5:none:none"])
    _write_trace(round2_merged, ["workspace:user_task_5:none:none"])
    _write_trace(round1_dir / "workspace_user_task_0_trace.jsonl", ["workspace:user_task_0:none:none"])

    result = audit_agentdojo_trace_provenance(
        merged_traces=[round1_merged, round2_merged],
        trace_dirs=[round1_dir],
        json_out=tmp_path / "audit.json",
        csv_out=tmp_path / "audit.csv",
    )

    assert result["workspace_round1_merged_appears_to_contain_round2_tasks"] is True
    assert result["duplicate_episode_count_across_batches"] == 1
    assert result["duplicate_episode_ids_across_batches"][0]["episode_id"] == "workspace:user_task_5:none:none"
    assert any(
        item["source_path"] == str(round1_merged)
        and "contains_task_ids_outside_expected_batch_range" in item["suspicion_reasons"]
        for item in result["suspicious_files"]
    )
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "audit.csv").exists()


def test_repair_round1_merged_trace_backs_up_and_rebuilds_from_trusted_traces(tmp_path):
    merged_trace = tmp_path / "merged" / "workspace_mini_batch_trace.jsonl"
    trace_dir = tmp_path / "traces"
    backup = tmp_path / "merged" / "workspace_mini_batch_trace.corrupt_backup.jsonl"
    prefix = tmp_path / "merged" / "workspace_mini_batch_prefix_dataset.csv"
    _write_trace(merged_trace, ["workspace:user_task_5:none:none"])
    for task_id in range(5):
        _write_trace(
            trace_dir / f"workspace_user_task_{task_id}_trace.jsonl",
            [f"workspace:user_task_{task_id}:none:none"],
        )

    result = repair_round1_merged_trace(
        merged_trace_path=merged_trace,
        trace_dir=trace_dir,
        backup_path=backup,
        prefix_path=prefix,
    )

    assert result["repaired_task_ids"] == [0, 1, 2, 3, 4]
    assert backup.exists()
    assert "workspace:user_task_5:none:none" in backup.read_text(encoding="utf-8")
    assert "workspace:user_task_5:none:none" not in merged_trace.read_text(encoding="utf-8")
    repaired_prefix = pd.read_csv(prefix)
    assert sorted(repaired_prefix["episode_id"].unique()) == [
        f"workspace:user_task_{task_id}:none:none"
        for task_id in range(5)
    ]
