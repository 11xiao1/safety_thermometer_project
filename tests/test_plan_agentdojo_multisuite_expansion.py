import json

from scripts import plan_agentdojo_multisuite_expansion as planner


def _write_trace(path, episode_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "step_id": 1,
                "hook_type": "pre_step",
                "user_instruction": "Summarize.",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_plan_agentdojo_multisuite_expansion_generates_missing_suite_batches(tmp_path, monkeypatch):
    available = {
        "workspace": [f"user_task_{i}" for i in range(3)],
        "slack": [f"user_task_{i}" for i in range(4)],
        "travel": [f"user_task_{i}" for i in range(3)],
        "banking": [f"user_task_{i}" for i in range(2)],
    }

    def fake_list_tasks(package_name, benchmark_version, suite):
        return {"status": "ok", "user_tasks": available[suite], "injection_tasks": []}

    workspace_dir = tmp_path / "workspace_traces"
    slack_dir = tmp_path / "slack_traces"
    _write_trace(workspace_dir / "workspace_user_task_0_trace.jsonl", "workspace:user_task_0:none:none")
    _write_trace(workspace_dir / "workspace_user_task_1_trace.jsonl", "workspace:user_task_1:none:none")
    _write_trace(workspace_dir / "workspace_user_task_2_trace.jsonl", "workspace:user_task_2:none:none")
    _write_trace(slack_dir / "slack_user_task_0_trace.jsonl", "slack:user_task_0:none:none")
    monkeypatch.setattr(planner, "list_tasks", fake_list_tasks)
    monkeypatch.setattr(
        planner,
        "TRACE_DIR_BATCHES",
        {
            "workspace": [("workspace_round1", workspace_dir)],
            "slack": [("slack_round1", slack_dir)],
            "travel": [],
            "banking": [],
        },
    )
    monkeypatch.setattr(planner, "DEFAULT_OUTPUTS_ROOT", tmp_path / "outputs")

    plan = planner.plan_agentdojo_multisuite_expansion(
        plan_out=tmp_path / "plan.json",
        commands_out=tmp_path / "commands.txt",
        batch_size=2,
    )

    assert plan["suite_plans"]["workspace"]["missing_count"] == 0
    assert plan["suite_plans"]["slack"]["missing_tasks"] == ["user_task_1", "user_task_2", "user_task_3"]
    assert plan["suite_plans"]["travel"]["missing_count"] == 3
    assert plan["suite_plans"]["banking"]["missing_count"] == 2
    assert not any(batch["suite"] == "workspace" for batch in plan["proposed_batches"])
    commands = (tmp_path / "commands.txt").read_text(encoding="utf-8")
    assert "--dry-run" in commands
    assert "--allow-real-run --allow-provider-call" in commands
    assert "--max-steps 6" in commands
    assert "--max-tool-calls 6" in commands
    assert (tmp_path / "plan.json").exists()


def test_plan_agentdojo_multisuite_expansion_records_no_provider_calls(tmp_path, monkeypatch):
    def fake_list_tasks(package_name, benchmark_version, suite):
        return {"status": "ok", "user_tasks": ["user_task_0"], "injection_tasks": []}

    monkeypatch.setattr(planner, "list_tasks", fake_list_tasks)
    monkeypatch.setattr(planner, "TRACE_DIR_BATCHES", {suite: [] for suite in planner.SUITES})
    monkeypatch.setattr(planner, "DEFAULT_OUTPUTS_ROOT", tmp_path / "outputs")

    plan = planner.plan_agentdojo_multisuite_expansion(
        plan_out=tmp_path / "plan.json",
        commands_out=tmp_path / "commands.txt",
    )

    assert plan["will_call_provider"] is False
    assert plan["will_run_agentdojo"] is False
