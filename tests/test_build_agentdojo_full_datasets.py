import json

import pandas as pd

from scripts.build_agentdojo_full_datasets import (
    DISAGREEMENT_COLUMNS,
    build_agentdojo_full_datasets,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prefix_rows(episode_id, label=0):
    rows = []
    for step in [1, 2]:
        row = {
            "episode_id": episode_id,
            "step_id": step,
            "hook_type": "pre_step" if step == 1 else "final",
            "future_risk_label": label,
            "oracle_violation": 0,
            "risk_score": float(step),
            "max_risk_score_so_far": float(step),
        }
        for column in DISAGREEMENT_COLUMNS:
            row[column] = 0
        rows.append(row)
    return rows


def _write_prefix(path, episode_id, label=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_prefix_rows(episode_id, label=label)).to_csv(path, index=False)


def _summary(path, suite, source_group, source_batch, selected, completed, stopped, prefix_root):
    results = []
    for task in completed:
        prefix = prefix_root / f"{suite}_{task}_prefix_dataset.csv"
        _write_prefix(prefix, f"{suite}:{task}:none:none", label=int(task.endswith("2")))
        results.append({
            "task_id": task,
            "status": "ok",
            "prefix": str(prefix),
            "trace": str(prefix.with_suffix(".jsonl")),
            "utility": not task.endswith("1"),
            "security": True,
        })
    for task in stopped:
        results.append({
            "task_id": task,
            "status": "stopped",
            "prefix": str(prefix_root / f"{suite}_{task}_prefix_dataset.csv"),
            "trace": str(prefix_root / f"{suite}_{task}.jsonl"),
            "stop_reason": "max_tool_calls_exceeded",
        })
    _write_json(
        path,
        {
            "suite": suite,
            "selected_tasks": selected,
            "completed_tasks": completed,
            "stopped_tasks": stopped,
            "failed_tasks": [],
            "task_results": results,
        },
    )
    return {
        "batch": source_batch,
        "source_group": source_group,
        "summary_path": str(path),
        "suite": suite,
        "selected_tasks": selected,
    }


def test_build_agentdojo_full_datasets_uses_recovery_and_excludes_budget_limited(tmp_path):
    base = tmp_path / "base.csv"
    base_df = pd.DataFrame(
        _prefix_rows("workspace:user_task_0:none:none", label=0)
        + _prefix_rows("slack:user_task_0:none:none", label=1)
    )
    base_df.insert(0, "source_batch", "base_batch")
    base_df.insert(0, "source_suite", "workspace")
    base_df.to_csv(base, index=False)

    expansion_prefix = tmp_path / "expansion_prefix"
    recovery_prefix = tmp_path / "recovery_prefix"
    expansion_row = _summary(
        tmp_path / "expansion/run_summary.json",
        "travel",
        "expansion",
        "travel_expansion_round1",
        ["user_task_1", "user_task_2"],
        ["user_task_1"],
        ["user_task_2"],
        expansion_prefix,
    )
    recovery_row = _summary(
        tmp_path / "recovery/run_summary.json",
        "travel",
        "recovery",
        "travel_recovery_round1",
        ["user_task_2", "user_task_3"],
        ["user_task_2"],
        ["user_task_3"],
        recovery_prefix,
    )
    audit = tmp_path / "audit.json"
    _write_json(
        audit,
        {
            "batches": [expansion_row],
            "recovery_batches": [recovery_row],
            "effective_after_recovery": {
                "budget_limited_tasks": ["travel:user_task_3"],
            },
        },
    )

    result = build_agentdojo_full_datasets(
        base_disagreement_path=base,
        audit_path=audit,
        full_out=tmp_path / "full/agentdojo_full_prefix_dataset.csv",
        full_disagreement_out=tmp_path / "full_disagreement/agentdojo_full_disagreement_prefix_dataset.csv",
        seed=42,
    )

    assert result["status"] == "ok"
    assert result["included_episode_count"] == 4
    assert result["excluded_budget_limited_tasks"] == ["travel:user_task_3"]
    full = pd.read_csv(tmp_path / "full/agentdojo_full_prefix_dataset.csv")
    full_disagreement = pd.read_csv(tmp_path / "full_disagreement/agentdojo_full_disagreement_prefix_dataset.csv")
    assert "travel:user_task_3:none:none" not in set(full["episode_id"])
    assert "travel:user_task_2:none:none" in set(full["episode_id"])
    assert set(DISAGREEMENT_COLUMNS).isdisjoint(full.columns)
    assert set(DISAGREEMENT_COLUMNS).issubset(full_disagreement.columns)
    assert int(full.duplicated(subset=["episode_id", "step_id", "hook_type"], keep=False).sum()) == 0

    no_manifest = json.loads((tmp_path / "full/splits/split_manifest.json").read_text(encoding="utf-8"))
    dis_manifest = json.loads((tmp_path / "full_disagreement/splits/split_manifest.json").read_text(encoding="utf-8"))
    assert no_manifest["episode_ids"] == dis_manifest["episode_ids"]
    assert no_manifest["excluded_budget_limited_tasks"] == ["travel:user_task_3"]
    assert no_manifest["episode_overlap_check"]["passed"] is True
    assert no_manifest["utility_counts"]
    assert no_manifest["security_counts"]
