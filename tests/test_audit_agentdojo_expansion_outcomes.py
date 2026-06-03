import csv
import json

from scripts.audit_agentdojo_expansion_outcomes import audit_agentdojo_expansion_outcomes


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_trace(path, episode_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episode_id": episode_id, "step_id": 1, "hook_type": "pre_step"}) + "\n",
        encoding="utf-8",
    )


def _summary(suite, selected, completed, stopped):
    task_results = []
    for task in completed:
        task_results.append({
            "task_id": task,
            "status": "ok",
            "utility": task.endswith("0"),
            "security": True,
            "max_tool_calls_hit": False,
            "prefix": f"prefix/{suite}_{task}.csv",
            "trace": f"traces/{suite}_{task}.jsonl",
        })
    for task in stopped:
        task_results.append({
            "task_id": task,
            "status": "stopped",
            "stop_reason": "max_tool_calls_exceeded",
            "max_tool_calls_hit": True,
            "prefix": f"prefix/{suite}_{task}.csv",
            "trace": f"traces/{suite}_{task}.jsonl",
        })
    return {
        "status": "partial" if stopped else "ok",
        "suite": suite,
        "selected_tasks": selected,
        "completed_tasks": completed,
        "stopped_tasks": stopped,
        "failed_tasks": [],
        "task_results": task_results,
        "merged_prefix_rows": 3,
        "merged_trace_rows": 4,
    }


def test_audit_agentdojo_expansion_outcomes_writes_audit_and_recovery_commands(tmp_path):
    root = tmp_path / "outputs/agentdojo_expansion"
    _write_json(
        root / "travel_expansion_round1/run_summary.json",
        _summary(
            "travel",
            ["user_task_0", "user_task_1", "user_task_2"],
            ["user_task_0", "user_task_1"],
            ["user_task_2"],
        ),
    )
    _write_json(
        root / "banking_expansion_round1/run_summary.json",
        _summary("banking", ["user_task_0"], ["user_task_0"], []),
    )
    _write_trace(root / "travel_expansion_round1/traces/travel_user_task_2_trace.jsonl", "travel:user_task_2:none:none")
    plan = tmp_path / "outputs/plan.json"
    _write_json(plan, {"status": "ok"})

    result = audit_agentdojo_expansion_outcomes(
        expansion_root=root,
        recovery_root=tmp_path / "outputs/no_recovery_yet",
        plan_path=plan,
        json_out=tmp_path / "outputs/audit.json",
        csv_out=tmp_path / "outputs/audit.csv",
        commands_out=tmp_path / "outputs/recovery_commands.txt",
    )

    assert result["status"] == "ok"
    assert result["totals"]["completed_task_count"] == 3
    assert result["totals"]["stopped_task_count"] == 1
    assert result["totals"]["utility_true_count"] == 2
    assert result["totals"]["utility_false_count"] == 1
    assert result["stopped_tasks_by_suite"] == {"travel": ["user_task_2"]}
    assert result["merge_guardrails"]["recovery_duplicate_episode_ids_against_expansion"] == [
        "travel:user_task_2:none:none"
    ]

    commands = (tmp_path / "outputs/recovery_commands.txt").read_text(encoding="utf-8")
    assert "outputs/agentdojo_expansion_recovery/travel_recovery_round1" in commands
    assert "--tasks user_task_2" in commands
    assert "--max-steps 10" in commands
    assert "--max-tool-calls 10" in commands
    assert "--dry-run" in commands
    assert "--allow-real-run --allow-provider-call" in commands
    assert "user_task_0,user_task_1" not in commands

    with (tmp_path / "outputs/audit.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    travel = next(row for row in rows if row["batch"] == "travel_expansion_round1")
    assert travel["utility_true_count"] == "1"
    assert travel["utility_false_count"] == "1"
    assert json.loads(travel["stopped_tasks"]) == ["user_task_2"]


def test_audit_agentdojo_expansion_outcomes_uses_stopped_tasks_only(tmp_path):
    root = tmp_path / "outputs/agentdojo_expansion"
    _write_json(
        root / "slack_expansion_round1/run_summary.json",
        _summary(
            "slack",
            ["user_task_10", "user_task_11", "user_task_12"],
            ["user_task_11"],
            ["user_task_10", "user_task_12"],
        ),
    )
    plan = tmp_path / "outputs/plan.json"
    _write_json(plan, {"status": "ok"})

    result = audit_agentdojo_expansion_outcomes(
        expansion_root=root,
        recovery_root=tmp_path / "outputs/no_recovery_yet",
        plan_path=plan,
        json_out=tmp_path / "audit.json",
        csv_out=tmp_path / "audit.csv",
        commands_out=tmp_path / "commands.txt",
    )

    assert result["recovery_commands"][0]["tasks"] == ["user_task_10", "user_task_12"]
    assert result["recovery_commands"][0]["task_count"] == 2
    commands = (tmp_path / "commands.txt").read_text(encoding="utf-8")
    assert "--limit 2" in commands
    assert "--tasks user_task_10,user_task_12" in commands
    assert "user_task_11" not in commands


def test_audit_agentdojo_expansion_outcomes_includes_recovery_effective_status(tmp_path):
    expansion = tmp_path / "outputs/agentdojo_expansion"
    recovery = tmp_path / "outputs/agentdojo_expansion_recovery"
    _write_json(
        expansion / "travel_expansion_round1/run_summary.json",
        _summary(
            "travel",
            ["user_task_0", "user_task_1", "user_task_2"],
            ["user_task_0"],
            ["user_task_1", "user_task_2"],
        ),
    )
    _write_json(
        recovery / "travel_recovery_round1/run_summary.json",
        _summary(
            "travel",
            ["user_task_1", "user_task_2"],
            ["user_task_1"],
            ["user_task_2"],
        ),
    )
    plan = tmp_path / "outputs/plan.json"
    _write_json(plan, {"status": "ok"})

    result = audit_agentdojo_expansion_outcomes(
        expansion_root=expansion,
        recovery_root=recovery,
        plan_path=plan,
        json_out=tmp_path / "audit.json",
        csv_out=tmp_path / "audit.csv",
        commands_out=tmp_path / "commands.txt",
    )

    effective = result["effective_after_recovery"]
    assert effective["completed_task_count"] == 2
    assert effective["stopped_task_count"] == 1
    assert effective["recovered_tasks"] == ["travel:user_task_1"]
    assert effective["still_stopped_tasks"] == ["travel:user_task_2"]
    assert result["stopped_tasks_by_suite"] == {"travel": ["user_task_2"]}
    assert result["next_recovery_round"] == 2

    commands = (tmp_path / "commands.txt").read_text(encoding="utf-8")
    assert "travel_recovery_round2" in commands
    assert "--tasks user_task_2" in commands
    assert "user_task_1,user_task_2" not in commands
