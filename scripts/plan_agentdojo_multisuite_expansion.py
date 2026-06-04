from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_agentdojo_smoke_trace import list_tasks  # noqa: E402
from src.monitor.logger import load_trace_events  # noqa: E402


DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_PLAN_OUT = "outputs/agentdojo_multisuite_expansion_plan.json"
DEFAULT_COMMANDS_OUT = "outputs/agentdojo_multisuite_expansion_commands.txt"
SUITES = ["workspace", "slack", "travel", "banking"]
TRACE_DIR_BATCHES = {
    "workspace": [
        ("workspace_round1", Path("outputs/agentdojo_mini_batch/traces")),
        ("workspace_round2", Path("outputs/agentdojo_mini_batch_round2/traces")),
        ("workspace_round3", Path("outputs/agentdojo_mini_batch_round3/traces")),
        ("workspace_round4", Path("outputs/agentdojo_mini_batch_round4/traces")),
    ],
    "slack": [
        ("slack_round1", Path("outputs/agentdojo_mini_batch_slack_round1/traces")),
        ("slack_recovery1", Path("outputs/agentdojo_mini_batch_slack_recovery1/traces")),
    ],
    "travel": [],
    "banking": [],
}
RUN_SETTINGS = {
    "provider": "openai-compatible",
    "model": "gpt-3.5-turbo",
    "max_steps": 6,
    "max_tool_calls": 6,
    "max_output_tokens": 512,
    "temperature": 0,
    "cost_guard": True,
}
DEFAULT_BATCH_SIZE = 10


def _task_sort_key(task_id: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)(\d+)", task_id)
    if match:
        return match.group(1), int(match.group(2)), task_id
    return task_id, -1, task_id


def _task_number(task_id: str) -> int | None:
    match = re.search(r"user_task_(\d+)", task_id)
    return int(match.group(1)) if match else None


def _episode_task_id(episode_id: str) -> str | None:
    task_number = _task_number(episode_id)
    return f"user_task_{task_number}" if task_number is not None else None


def _available_user_tasks(suite: str, benchmark_version: str) -> list[str]:
    payload = list_tasks("agentdojo", benchmark_version, suite)
    if payload.get("status") != "ok":
        raise ValueError(f"Could not list tasks for suite {suite}: {payload.get('status')}")
    return sorted(payload["user_tasks"], key=_task_sort_key)


def _scan_trace_dir(path: Path) -> dict[str, Any]:
    task_ids: set[str] = set()
    episode_ids: set[str] = set()
    event_count = 0
    files = sorted(path.glob("*.jsonl")) if path.exists() else []
    for trace_file in files:
        events = load_trace_events(trace_file)
        event_count += len(events)
        for event in events:
            episode_ids.add(event.episode_id)
            task_id = _episode_task_id(event.episode_id)
            if task_id is not None:
                task_ids.add(task_id)
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "event_count": event_count,
        "episode_ids": sorted(episode_ids),
        "task_ids": sorted(task_ids, key=_task_sort_key),
    }


def _suite_existing_summary(suite: str) -> dict[str, Any]:
    batch_summaries = []
    completed: set[str] = set()
    for batch, trace_dir in TRACE_DIR_BATCHES.get(suite, []):
        scanned = _scan_trace_dir(trace_dir)
        scanned["source_batch"] = batch
        batch_summaries.append(scanned)
        completed.update(scanned["task_ids"])
    run_summaries = []
    for summary_path in sorted(DEFAULT_OUTPUTS_ROOT.glob("agentdojo_mini_batch*/run_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("suite") != suite:
            continue
        run_summaries.append({
            "path": str(summary_path),
            "status": payload.get("status"),
            "selected_tasks": payload.get("selected_tasks", []),
            "completed_tasks": payload.get("completed_tasks", []),
            "stopped_tasks": payload.get("stopped_tasks", []),
            "failed_tasks": payload.get("failed_tasks", []),
        })
    stopped = sorted(
        {
            task
            for summary in run_summaries
            for task in summary.get("stopped_tasks", [])
        },
        key=_task_sort_key,
    )
    return {
        "completed_tasks": sorted(completed, key=_task_sort_key),
        "completed_count": len(completed),
        "stopped_tasks_from_run_summaries": stopped,
        "stopped_count_from_run_summaries": len(stopped),
        "trace_batches": batch_summaries,
        "run_summaries": run_summaries,
    }


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _batch_name(suite: str, index: int) -> str:
    return f"{suite}_expansion_round{index}"


def _run_command(suite: str, batch_name: str, tasks: list[str], dry_run: bool) -> str:
    task_arg = ",".join(tasks)
    base_out = f"outputs/agentdojo_expansion/{batch_name}"
    command = [
        sys.executable,
        "scripts/run_agentdojo_mini_batch.py",
        "--suite", suite,
        "--limit", str(len(tasks)),
        "--tasks", task_arg,
        "--trace-dir", f"{base_out}/traces",
        "--prefix-dir", f"{base_out}/prefix",
        "--merged-out", f"{base_out}/merged/{batch_name}_prefix_dataset.csv",
        "--summary", f"{base_out}/run_summary.json",
        "--provider", RUN_SETTINGS["provider"],
        "--model", RUN_SETTINGS["model"],
        "--max-steps", str(RUN_SETTINGS["max_steps"]),
        "--max-tool-calls", str(RUN_SETTINGS["max_tool_calls"]),
        "--max-output-tokens", str(RUN_SETTINGS["max_output_tokens"]),
        "--temperature", str(RUN_SETTINGS["temperature"]),
        "--cost-guard",
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        command.extend(["--allow-real-run", "--allow-provider-call"])
    return " ".join(command)


def plan_agentdojo_multisuite_expansion(
    plan_out: str | Path = DEFAULT_PLAN_OUT,
    commands_out: str | Path = DEFAULT_COMMANDS_OUT,
    benchmark_version: str = "v1.2.2",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    suite_plans: dict[str, Any] = {}
    proposed_batches = []
    commands = []
    total_available = 0
    total_completed = 0
    total_missing = 0
    for suite in SUITES:
        available = _available_user_tasks(suite, benchmark_version)
        existing = _suite_existing_summary(suite)
        completed = set(existing["completed_tasks"])
        missing = [task for task in available if task not in completed]
        total_available += len(available)
        total_completed += len(completed)
        total_missing += len(missing)
        suite_plans[suite] = {
            "available_tasks": available,
            "available_count": len(available),
            "completed_tasks": existing["completed_tasks"],
            "completed_count": len(completed),
            "stopped_tasks": existing["stopped_tasks_from_run_summaries"],
            "stopped_count": existing["stopped_count_from_run_summaries"],
            "missing_tasks": missing,
            "missing_count": len(missing),
            "trace_batches": existing["trace_batches"],
            "run_summaries": existing["run_summaries"],
        }
        if suite == "workspace":
            continue
        for batch_index, tasks in enumerate(_chunks(missing, batch_size), start=1):
            batch_name = _batch_name(suite, batch_index)
            batch = {
                "suite": suite,
                "batch_name": batch_name,
                "tasks": tasks,
                "task_count": len(tasks),
                "output_dir": f"outputs/agentdojo_expansion/{batch_name}",
                "dry_run_command": _run_command(suite, batch_name, tasks, dry_run=True),
                "real_run_command": _run_command(suite, batch_name, tasks, dry_run=False),
            }
            proposed_batches.append(batch)
            commands.append(f"# {batch_name} ({suite}, {len(tasks)} tasks)")
            commands.append("# dry-run")
            commands.append(batch["dry_run_command"])
            commands.append("# real-run")
            commands.append(batch["real_run_command"])
            commands.append("")

    projected_unique_user_episodes = total_completed + total_missing
    plan = {
        "status": "ok",
        "benchmark_version": benchmark_version,
        "will_call_provider": False,
        "will_run_agentdojo": False,
        "run_settings": RUN_SETTINGS,
        "target_note": (
            "Full available AgentDojo user tasks total 97, so 100+ unique user-task episodes is not reachable "
            "without adding injection tasks or repeated recovery/rerun episodes."
        ),
        "total_available_user_tasks": total_available,
        "current_completed_unique_user_tasks": total_completed,
        "remaining_missing_user_tasks": total_missing,
        "projected_unique_user_tasks_after_plan": projected_unique_user_episodes,
        "suite_plans": suite_plans,
        "proposed_batches": proposed_batches,
        "outputs": {
            "plan": str(plan_out),
            "commands": str(commands_out),
        },
    }
    plan_out = Path(plan_out)
    commands_out = Path(commands_out)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    commands_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    commands_out.write_text("\n".join(commands).rstrip() + "\n", encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan AgentDojo multi-suite expansion commands without running tasks.")
    parser.add_argument("--plan-out", default=DEFAULT_PLAN_OUT)
    parser.add_argument("--commands-out", default=DEFAULT_COMMANDS_OUT)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    plan = plan_agentdojo_multisuite_expansion(
        plan_out=args.plan_out,
        commands_out=args.commands_out,
        benchmark_version=args.benchmark_version,
        batch_size=args.batch_size,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
