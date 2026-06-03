from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXPANSION_ROOT = "outputs/agentdojo_expansion"
DEFAULT_RECOVERY_ROOT = "outputs/agentdojo_expansion_recovery"
DEFAULT_PLAN = "outputs/agentdojo_multisuite_expansion_plan.json"
DEFAULT_JSON_OUT = "outputs/agentdojo_expansion_outcome_audit.json"
DEFAULT_CSV_OUT = "outputs/agentdojo_expansion_outcome_audit.csv"
DEFAULT_COMMANDS_OUT = "outputs/agentdojo_expansion_recovery_commands.txt"
DEFAULT_UTILITY_AUDIT_JSON_OUT = "outputs/agentdojo_expansion_utility_false_trace_quality_audit.json"
DEFAULT_UTILITY_AUDIT_CSV_OUT = "outputs/agentdojo_expansion_utility_false_trace_quality_audit.csv"
PYTHON_EXE = r"F:\Anaconda_envs\envs\safetythermo\python.exe"
RECOVERY_ROOT = "outputs/agentdojo_expansion_recovery"
CSV_COLUMNS = [
    "batch",
    "suite",
    "selected_tasks",
    "completed_tasks",
    "stopped_tasks",
    "failed_tasks",
    "stop_reason_counts",
    "utility_true_count",
    "utility_false_count",
    "utility_unknown_count",
    "security_true_count",
    "security_false_count",
    "security_unknown_count",
    "max_tool_calls_hit_count",
    "merged_prefix_rows",
    "merged_trace_rows",
    "missing_prefix_for_stopped_tasks",
    "warnings",
]
UTILITY_AUDIT_COLUMNS = [
    "suite",
    "task_id",
    "episode_id",
    "source_group",
    "batch",
    "status",
    "utility",
    "security",
    "trace",
    "trace_exists",
    "trace_rows",
    "final_event_exists",
    "tool_call_count",
    "valid_tool_calls",
    "prefix",
    "prefix_exists",
    "prefix_rows",
    "future_risk_label_counts",
    "oracle_violation_counts",
    "risk_score_min",
    "risk_score_max",
    "risk_score_mean",
    "quality_status",
    "interpretation",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(values: list[Any]) -> str:
    return json.dumps(values, sort_keys=True)


def _json_dict(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True)


def _task_sort_key(task_id: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)(\d+)", task_id)
    if match:
        return match.group(1), int(match.group(2)), task_id
    return task_id, -1, task_id


def _bool_counts(results: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = {"true": 0, "false": 0, "unknown": 0}
    for result in results:
        value = result.get(field)
        if value is True:
            counts["true"] += 1
        elif value is False:
            counts["false"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _read_episode_ids_from_trace(path: str | Path) -> set[str]:
    path = Path(path)
    episode_ids: set[str] = set()
    if not path.exists():
        return episode_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_id = payload.get("episode_id")
            if episode_id:
                episode_ids.add(str(episode_id))
    return episode_ids


def _read_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_decode_error": True})
    return rows


def _read_prefix_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_values(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            value = "missing"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _numeric_stats(rows: list[dict[str, Any]], column: str) -> dict[str, float | None]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(column, "")))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _trace_episode_inventory(expansion_root: str | Path) -> dict[str, list[str]]:
    inventory: dict[str, set[str]] = defaultdict(set)
    root = Path(expansion_root)
    if not root.exists():
        return {}
    for trace_path in root.glob("*/traces/*.jsonl"):
        batch = trace_path.parents[1].name
        inventory[batch].update(_read_episode_ids_from_trace(trace_path))
    return {batch: sorted(values) for batch, values in sorted(inventory.items())}


def _batch_name_from_summary(path: Path) -> str:
    return path.parent.name


def _audit_one_summary(path: Path, source_group: str) -> dict[str, Any]:
    summary = _load_json(path)
    batch = _batch_name_from_summary(path)
    selected_tasks = [str(task) for task in summary.get("selected_tasks", [])]
    completed_tasks = [str(task) for task in summary.get("completed_tasks", [])]
    stopped_tasks = [str(task) for task in summary.get("stopped_tasks", [])]
    failed_tasks = [str(task) for task in summary.get("failed_tasks", [])]
    task_results = list(summary.get("task_results", []))
    completed_results = [result for result in task_results if result.get("status") == "ok"]
    stop_reason_counts = Counter(
        str(result.get("stop_reason") or "none")
        for result in task_results
        if result.get("status") in {"stopped", "failed"} or result.get("stop_reason")
    )
    utility_counts = _bool_counts(completed_results, "utility")
    security_counts = _bool_counts(completed_results, "security")
    max_tool_calls_hit_count = sum(1 for result in task_results if result.get("max_tool_calls_hit") is True)
    missing_prefix_for_stopped = [
        str(result.get("task_id"))
        for result in task_results
        if result.get("status") == "stopped" and not Path(str(result.get("prefix", ""))).exists()
    ]

    warnings: list[str] = []
    if stopped_tasks:
        warnings.append("Stopped tasks require recovery before final merge.")
    if utility_counts["false"]:
        warnings.append("Completed tasks include utility=false; audit before training or reporting.")
    if failed_tasks:
        warnings.append("Failed tasks require manual inspection.")
    if missing_prefix_for_stopped:
        warnings.append("Some stopped tasks have no prefix CSV; this is expected for max-tool-call stops.")

    return {
        "batch": batch,
        "source_group": source_group,
        "summary_path": str(path),
        "suite": str(summary.get("suite", "")),
        "status": summary.get("status"),
        "selected_tasks": selected_tasks,
        "completed_tasks": completed_tasks,
        "stopped_tasks": stopped_tasks,
        "failed_tasks": failed_tasks,
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "utility_counts_completed": utility_counts,
        "security_counts_completed": security_counts,
        "max_tool_calls_hit_count": int(max_tool_calls_hit_count),
        "merged_prefix": summary.get("merged_prefix"),
        "merged_prefix_rows": int(summary.get("merged_prefix_rows") or 0),
        "merged_trace": summary.get("merged_trace"),
        "merged_trace_rows": int(summary.get("merged_trace_rows") or 0),
        "missing_prefix_for_stopped_tasks": missing_prefix_for_stopped,
        "warnings": warnings,
    }


def _build_recovery_command(suite: str, tasks: list[str], dry_run: bool, recovery_round: int) -> str:
    batch = f"{suite}_recovery_round{recovery_round}"
    out_dir = f"{RECOVERY_ROOT}/{batch}"
    pieces = [
        PYTHON_EXE,
        "scripts\\run_agentdojo_mini_batch.py",
        "--suite",
        suite,
        "--limit",
        str(len(tasks)),
        "--tasks",
        ",".join(tasks),
        "--trace-dir",
        f"{out_dir}\\traces",
        "--prefix-dir",
        f"{out_dir}\\prefix",
        "--merged-out",
        f"{out_dir}\\merged\\{batch}_prefix_dataset.csv",
        "--summary",
        f"{out_dir}\\run_summary.json",
        "--provider",
        "openai-compatible",
        "--model",
        "gpt-3.5-turbo",
        "--max-steps",
        "10",
        "--max-tool-calls",
        "10",
        "--max-output-tokens",
        "512",
        "--temperature",
        "0",
        "--cost-guard",
    ]
    if dry_run:
        pieces.append("--dry-run")
    else:
        pieces.extend(["--allow-real-run", "--allow-provider-call"])
    return " ".join(pieces)


def _write_recovery_commands(
    path: str | Path,
    stopped_by_suite: dict[str, list[str]],
    recovery_round: int,
) -> list[dict[str, Any]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command_entries: list[dict[str, Any]] = []
    lines = [
        "# AgentDojo expansion recovery commands",
        "# Generated for stopped tasks only.",
        "# Dry-run commands do not call providers.",
        "# Real-run commands must be run manually one suite at a time.",
        "# Recovery outputs are intentionally separate; do not append original stopped traces and recovery traces together in final merges.",
        "",
    ]
    for suite in sorted(stopped_by_suite):
        tasks = sorted(set(stopped_by_suite[suite]), key=_task_sort_key)
        if not tasks:
            continue
        batch = f"{suite}_recovery_round{recovery_round}"
        dry_run_command = _build_recovery_command(suite, tasks, dry_run=True, recovery_round=recovery_round)
        real_run_command = _build_recovery_command(suite, tasks, dry_run=False, recovery_round=recovery_round)
        lines.extend([
            f"# {batch} ({suite}, {len(tasks)} stopped tasks)",
            "# dry-run",
            dry_run_command,
            "# real-run",
            real_run_command,
            "",
        ])
        command_entries.append({
            "suite": suite,
            "batch": batch,
            "tasks": tasks,
            "task_count": len(tasks),
            "output_dir": f"{RECOVERY_ROOT}/{batch}",
            "dry_run_command": dry_run_command,
            "real_run_command": real_run_command,
        })
    path.write_text("\n".join(lines), encoding="utf-8")
    return command_entries


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            utility = row["utility_counts_completed"]
            security = row["security_counts_completed"]
            writer.writerow({
                "batch": row["batch"],
                "suite": row["suite"],
                "selected_tasks": _json_list(row["selected_tasks"]),
                "completed_tasks": _json_list(row["completed_tasks"]),
                "stopped_tasks": _json_list(row["stopped_tasks"]),
                "failed_tasks": _json_list(row["failed_tasks"]),
                "stop_reason_counts": _json_dict(row["stop_reason_counts"]),
                "utility_true_count": utility["true"],
                "utility_false_count": utility["false"],
                "utility_unknown_count": utility["unknown"],
                "security_true_count": security["true"],
                "security_false_count": security["false"],
                "security_unknown_count": security["unknown"],
                "max_tool_calls_hit_count": row["max_tool_calls_hit_count"],
                "merged_prefix_rows": row["merged_prefix_rows"],
                "merged_trace_rows": row["merged_trace_rows"],
                "missing_prefix_for_stopped_tasks": _json_list(row["missing_prefix_for_stopped_tasks"]),
                "warnings": _json_list(row["warnings"]),
            })


def _task_result_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        suite = row["suite"]
        by_task = {
            str(result.get("task_id")): result
            for result in _load_json(row["summary_path"]).get("task_results", [])
            if result.get("task_id")
        }
        for task_id in row["selected_tasks"]:
            result = by_task.get(task_id, {"task_id": task_id, "status": "missing"})
            enriched = dict(result)
            enriched["suite"] = suite
            enriched["batch"] = row["batch"]
            enriched["source_group"] = row["source_group"]
            results[(suite, task_id)] = enriched
    return results


def _effective_task_results(
    primary_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    primary_results = _task_result_map(primary_rows)
    recovery_results = _task_result_map(recovery_rows)
    effective: dict[tuple[str, str], dict[str, Any]] = {}

    for key, result in primary_results.items():
        if result.get("status") == "stopped" and key in recovery_results:
            effective[key] = recovery_results[key]
        else:
            effective[key] = result
    return effective


def _effective_after_recovery(
    primary_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    budget_limited_tasks: set[str] | None = None,
) -> dict[str, Any]:
    effective = _effective_task_results(primary_rows, recovery_rows)
    recovery_results = _task_result_map(recovery_rows)
    budget_limited_tasks = budget_limited_tasks or set()
    recovered_tasks: list[str] = []
    still_stopped_tasks: list[str] = []
    budget_limited_found: list[str] = []

    for key, result in effective.items():
        full_task = f"{key[0]}:{key[1]}"
        if full_task in budget_limited_tasks and result.get("status") == "stopped":
            budget_limited_found.append(full_task)
        if key in recovery_results and recovery_results[key].get("status") == "ok":
            recovered_tasks.append(full_task)

    completed = [result for result in effective.values() if result.get("status") == "ok"]
    stopped = [
        result
        for result in effective.values()
        if result.get("status") == "stopped"
        and f"{result['suite']}:{result['task_id']}" not in budget_limited_tasks
    ]
    failed = [result for result in effective.values() if result.get("status") == "failed"]
    missing = [result for result in effective.values() if result.get("status") == "missing"]
    utility_counts = _bool_counts(completed, "utility")
    security_counts = _bool_counts(completed, "security")
    for result in stopped:
        still_stopped_tasks.append(f"{result['suite']}:{result['task_id']}")

    completed_count = len(completed)
    utility_false_ratio = (
        utility_counts["false"] / completed_count
        if completed_count
        else 0.0
    )
    return {
        "selected_task_count": len(effective),
        "completed_task_count": completed_count,
        "stopped_task_count": len(stopped),
        "budget_limited_task_count": len(budget_limited_found),
        "budget_limited_tasks": sorted(budget_limited_found),
        "failed_task_count": len(failed),
        "missing_task_count": len(missing),
        "recovered_task_count": len(recovered_tasks),
        "recovered_tasks": sorted(recovered_tasks),
        "still_stopped_tasks": sorted(still_stopped_tasks),
        "still_stopped_tasks_by_suite": _tasks_by_suite(still_stopped_tasks),
        "utility_counts_completed": utility_counts,
        "security_counts_completed": security_counts,
        "utility_false_ratio_completed": utility_false_ratio,
        "utility_false_ratio_completed_pct": round(utility_false_ratio * 100.0, 2),
        "utility_false_ratio_note": "Computed among effective completed tasks after recovery supersedes original stopped results.",
    }


def _utility_false_trace_quality_audit(
    primary_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    effective = _effective_task_results(primary_rows, recovery_rows)
    rows: list[dict[str, Any]] = []
    for (suite, task_id), result in sorted(effective.items()):
        if result.get("status") != "ok" or result.get("utility") is not False:
            continue
        trace_path = Path(str(result.get("trace", "")))
        prefix_path = Path(str(result.get("prefix", "")))
        trace_rows = _read_trace_rows(trace_path)
        prefix_rows = _read_prefix_rows(prefix_path)
        hook_counts = Counter(str(row.get("hook_type", "missing")) for row in trace_rows)
        episode_ids = sorted({str(row.get("episode_id")) for row in trace_rows if row.get("episode_id")})
        if not episode_ids:
            episode_ids = sorted({str(row.get("episode_id")) for row in prefix_rows if row.get("episode_id")})
        final_event_exists = hook_counts.get("final", 0) > 0
        tool_call_count = int(result.get("tool_call_count") or hook_counts.get("pre_step", 0))
        valid_tool_calls = tool_call_count > 0
        stats = _numeric_stats(prefix_rows, "risk_score")
        trace_quality_ok = bool(
            trace_path.exists()
            and prefix_path.exists()
            and trace_rows
            and prefix_rows
            and final_event_exists
            and valid_tool_calls
            and result.get("status") == "ok"
        )
        interpretation = (
            "task_result_not_trace_quality"
            if trace_quality_ok and result.get("security") is True
            else "trace_quality_or_metadata_issue"
        )
        rows.append({
            "suite": suite,
            "task_id": task_id,
            "episode_id": episode_ids[0] if episode_ids else "",
            "source_group": result.get("source_group", ""),
            "batch": result.get("batch", ""),
            "status": result.get("status"),
            "utility": result.get("utility"),
            "security": result.get("security"),
            "trace": str(trace_path),
            "trace_exists": trace_path.exists(),
            "trace_rows": len(trace_rows),
            "trace_hook_counts": dict(sorted(hook_counts.items())),
            "final_event_exists": final_event_exists,
            "tool_call_count": tool_call_count,
            "valid_tool_calls": valid_tool_calls,
            "prefix": str(prefix_path),
            "prefix_exists": prefix_path.exists(),
            "prefix_rows": len(prefix_rows),
            "future_risk_label_counts": _count_values(prefix_rows, "future_risk_label"),
            "oracle_violation_counts": _count_values(prefix_rows, "oracle_violation"),
            "risk_score": stats,
            "quality_status": "usable_trace" if trace_quality_ok else "quality_issue",
            "interpretation": interpretation,
        })
    quality_counts = Counter(row["quality_status"] for row in rows)
    interpretation_counts = Counter(row["interpretation"] for row in rows)
    return {
        "status": "ok",
        "utility_false_completed_count": len(rows),
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "interpretation_counts": dict(sorted(interpretation_counts.items())),
        "rows": rows,
        "recommendation": (
            "Keep utility=false completed episodes as a separately reported subset if quality_status is usable_trace; "
            "do not use quality_issue rows in the main training dataset without manual repair."
        ),
    }


def _write_utility_audit_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UTILITY_AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            risk_score = row["risk_score"]
            writer.writerow({
                "suite": row["suite"],
                "task_id": row["task_id"],
                "episode_id": row["episode_id"],
                "source_group": row["source_group"],
                "batch": row["batch"],
                "status": row["status"],
                "utility": row["utility"],
                "security": row["security"],
                "trace": row["trace"],
                "trace_exists": row["trace_exists"],
                "trace_rows": row["trace_rows"],
                "final_event_exists": row["final_event_exists"],
                "tool_call_count": row["tool_call_count"],
                "valid_tool_calls": row["valid_tool_calls"],
                "prefix": row["prefix"],
                "prefix_exists": row["prefix_exists"],
                "prefix_rows": row["prefix_rows"],
                "future_risk_label_counts": _json_dict(row["future_risk_label_counts"]),
                "oracle_violation_counts": _json_dict(row["oracle_violation_counts"]),
                "risk_score_min": risk_score["min"],
                "risk_score_max": risk_score["max"],
                "risk_score_mean": risk_score["mean"],
                "quality_status": row["quality_status"],
                "interpretation": row["interpretation"],
            })


def _tasks_by_suite(items: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if ":" in item:
            suite, task_id = item.split(":", 1)
        else:
            suite, task_id = "unknown", item
        grouped[suite].append(task_id)
    return {
        suite: sorted(set(tasks), key=_task_sort_key)
        for suite, tasks in sorted(grouped.items())
        if tasks
    }


def _next_recovery_round(recovery_rows: list[dict[str, Any]]) -> int:
    max_round = 0
    for row in recovery_rows:
        match = re.search(r"_recovery_round(\d+)$", row["batch"])
        if match:
            max_round = max(max_round, int(match.group(1)))
    return max_round + 1 if max_round else 1


def audit_agentdojo_expansion_outcomes(
    expansion_root: str | Path = DEFAULT_EXPANSION_ROOT,
    recovery_root: str | Path = DEFAULT_RECOVERY_ROOT,
    plan_path: str | Path = DEFAULT_PLAN,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    commands_out: str | Path = DEFAULT_COMMANDS_OUT,
    utility_audit_json_out: str | Path = DEFAULT_UTILITY_AUDIT_JSON_OUT,
    utility_audit_csv_out: str | Path = DEFAULT_UTILITY_AUDIT_CSV_OUT,
    budget_limited_tasks: list[str] | None = None,
) -> dict[str, Any]:
    expansion_root = Path(expansion_root)
    recovery_root = Path(recovery_root)
    primary_summaries = sorted(expansion_root.glob("*/run_summary.json"))
    recovery_summaries = sorted(recovery_root.glob("*/run_summary.json")) if recovery_root.exists() else []
    if not primary_summaries:
        raise FileNotFoundError(f"No run_summary.json files found under {expansion_root}")

    primary_rows = [_audit_one_summary(path, "expansion") for path in primary_summaries]
    recovery_rows = [_audit_one_summary(path, "recovery") for path in recovery_summaries]
    batch_rows = primary_rows + recovery_rows
    budget_limited_set = set(budget_limited_tasks or [])
    effective = _effective_after_recovery(primary_rows, recovery_rows, budget_limited_set)
    stopped_by_suite: dict[str, list[str]] = defaultdict(list)
    for suite, tasks in effective["still_stopped_tasks_by_suite"].items():
        stopped_by_suite[suite].extend(tasks)

    next_recovery_round = _next_recovery_round(recovery_rows)
    recovery_commands = _write_recovery_commands(commands_out, stopped_by_suite, recovery_round=next_recovery_round)
    trace_inventory = _trace_episode_inventory(expansion_root)
    recovery_episode_ids = {
        entry["suite"]: [f"{entry['suite']}:{task_id}:none:none" for task_id in entry["tasks"]]
        for entry in recovery_commands
    }
    existing_episode_ids = {
        episode_id
        for episode_ids in trace_inventory.values()
        for episode_id in episode_ids
    }
    duplicate_episode_ids = sorted(
        episode_id
        for episode_ids in recovery_episode_ids.values()
        for episode_id in episode_ids
        if episode_id in existing_episode_ids
    )

    totals = {
        "batch_count": len(batch_rows),
        "selected_task_count": sum(len(row["selected_tasks"]) for row in batch_rows),
        "completed_task_count": sum(len(row["completed_tasks"]) for row in batch_rows),
        "stopped_task_count": sum(len(row["stopped_tasks"]) for row in batch_rows),
        "failed_task_count": sum(len(row["failed_tasks"]) for row in batch_rows),
        "utility_true_count": sum(row["utility_counts_completed"]["true"] for row in batch_rows),
        "utility_false_count": sum(row["utility_counts_completed"]["false"] for row in batch_rows),
        "utility_unknown_count": sum(row["utility_counts_completed"]["unknown"] for row in batch_rows),
        "security_true_count": sum(row["security_counts_completed"]["true"] for row in batch_rows),
        "security_false_count": sum(row["security_counts_completed"]["false"] for row in batch_rows),
        "security_unknown_count": sum(row["security_counts_completed"]["unknown"] for row in batch_rows),
        "max_tool_calls_hit_count": sum(row["max_tool_calls_hit_count"] for row in batch_rows),
        "merged_prefix_rows": sum(row["merged_prefix_rows"] for row in batch_rows),
        "merged_trace_rows": sum(row["merged_trace_rows"] for row in batch_rows),
    }
    plan = _load_json(plan_path) if Path(plan_path).exists() else {}
    payload = {
        "status": "ok",
        "inputs": {
            "expansion_root": str(expansion_root),
            "recovery_root": str(recovery_root),
            "plan": str(plan_path),
        },
        "outputs": {
            "json": str(json_out),
            "csv": str(csv_out),
            "recovery_commands": str(commands_out),
            "utility_false_trace_quality_json": str(utility_audit_json_out),
            "utility_false_trace_quality_csv": str(utility_audit_csv_out),
        },
        "plan_status": plan.get("status"),
        "batches": primary_rows,
        "recovery_batches": recovery_rows,
        "totals": totals,
        "effective_after_recovery": effective,
        "stopped_tasks_by_suite": {
            suite: sorted(set(tasks), key=_task_sort_key)
            for suite, tasks in sorted(stopped_by_suite.items())
            if tasks
        },
        "next_recovery_round": next_recovery_round,
        "recovery_commands": recovery_commands,
        "merge_guardrails": {
            "recovery_outputs_are_separate": True,
            "do_not_merge_original_stopped_partial_traces_with_recovery_traces": True,
            "dedupe_key": ["episode_id", "step_id", "hook_type"],
            "recovery_duplicate_episode_ids_against_expansion": duplicate_episode_ids,
            "recovery_duplicate_episode_count_against_expansion": len(duplicate_episode_ids),
            "note": (
                "Recovery reruns reuse the same suite/task episode ids as stopped originals. "
                "When building a final dataset, prefer successful recovery outputs for those episodes "
                "and exclude original stopped partial traces to avoid duplicate prefix rows."
            ),
        },
        "rules": {
            "will_call_provider": False,
            "will_run_agentdojo": False,
            "will_train_risk_estimator": False,
            "will_fit_calibration": False,
            "test_split_used": False,
        },
    }
    utility_audit = _utility_false_trace_quality_audit(primary_rows, recovery_rows)
    payload["utility_false_trace_quality_audit"] = {
        "json": str(utility_audit_json_out),
        "csv": str(utility_audit_csv_out),
        "utility_false_completed_count": utility_audit["utility_false_completed_count"],
        "quality_status_counts": utility_audit["quality_status_counts"],
        "interpretation_counts": utility_audit["interpretation_counts"],
        "recommendation": utility_audit["recommendation"],
    }

    json_out = Path(json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_out, batch_rows)
    utility_audit_json_out = Path(utility_audit_json_out)
    utility_audit_json_out.parent.mkdir(parents=True, exist_ok=True)
    utility_audit_json_out.write_text(json.dumps(utility_audit, indent=2, sort_keys=True), encoding="utf-8")
    _write_utility_audit_csv(utility_audit_csv_out, utility_audit["rows"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AgentDojo expansion outcomes and write recovery commands.")
    parser.add_argument("--expansion-root", default=DEFAULT_EXPANSION_ROOT)
    parser.add_argument("--recovery-root", default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--commands-out", default=DEFAULT_COMMANDS_OUT)
    parser.add_argument("--utility-audit-json-out", default=DEFAULT_UTILITY_AUDIT_JSON_OUT)
    parser.add_argument("--utility-audit-csv-out", default=DEFAULT_UTILITY_AUDIT_CSV_OUT)
    parser.add_argument("--budget-limited-task", action="append", default=[])
    args = parser.parse_args()

    result = audit_agentdojo_expansion_outcomes(
        expansion_root=args.expansion_root,
        recovery_root=args.recovery_root,
        plan_path=args.plan,
        json_out=args.json_out,
        csv_out=args.csv_out,
        commands_out=args.commands_out,
        utility_audit_json_out=args.utility_audit_json_out,
        utility_audit_csv_out=args.utility_audit_csv_out,
        budget_limited_tasks=args.budget_limited_task,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
