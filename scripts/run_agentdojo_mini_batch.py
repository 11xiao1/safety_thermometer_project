from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_agentdojo_smoke_trace import (  # noqa: E402
    _cost_guard_status,
    _provider_status,
    _run_real_smoke,
    list_tasks,
)
from src.adapters.agentdojo_pipeline_builder import inspect_agentdojo_pipeline  # noqa: E402
from src.adapters.agentdojo_tools_wrapper import MAX_TOOL_CALLS_EXCEEDED_REASON, MaxToolCallsExceeded  # noqa: E402
from src.monitor.logger import load_trace_events  # noqa: E402
from src.monitor.replay import make_prefix_dataset  # noqa: E402


DEFAULT_TRACE_DIR = "outputs/agentdojo_mini_batch/traces"
DEFAULT_PREFIX_DIR = "outputs/agentdojo_mini_batch/prefixes"
DEFAULT_MERGED_OUT = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv"
DEFAULT_SUMMARY = "outputs/agentdojo_mini_batch/run_summary.json"


def _blocked(message: dict[str, Any], exit_code: int = 2) -> None:
    print(json.dumps(message, indent=2), file=sys.stderr)
    raise SystemExit(exit_code)


def _task_sort_key(task_id: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)(\d+)", task_id)
    if match:
        return match.group(1), int(match.group(2)), task_id
    return task_id, -1, task_id


def _parse_tasks(tasks: str | None) -> list[str] | None:
    if tasks is None:
        return None
    selected = [task.strip() for task in tasks.split(",") if task.strip()]
    return selected


def _safe_task_name(suite: str, task_id: str) -> str:
    safe_suite = re.sub(r"[^A-Za-z0-9_.-]+", "_", suite)
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    return f"{safe_suite}_{safe_task}"


def select_tasks(
    available_tasks: list[str],
    limit: int,
    task_start: str | None = None,
    tasks: str | None = None,
) -> list[str]:
    if limit <= 0:
        raise ValueError("--limit must be positive.")
    if limit > 10:
        raise ValueError("--limit must not exceed 10 for guarded mini-batch runs.")

    available_set = set(available_tasks)
    explicit_tasks = _parse_tasks(tasks)
    if explicit_tasks is not None:
        missing = [task for task in explicit_tasks if task not in available_set]
        if missing:
            raise ValueError("Unknown task id(s): " + ", ".join(missing))
        return explicit_tasks[:limit]

    ordered = sorted(available_tasks, key=_task_sort_key)
    if task_start is not None:
        if task_start not in available_set:
            raise ValueError(f"Unknown --task-start: {task_start}")
        start_index = ordered.index(task_start)
        ordered = ordered[start_index:]
    return ordered[:limit]


def output_paths_for_task(suite: str, task_id: str, trace_dir: str | Path, prefix_dir: str | Path) -> dict[str, str]:
    base = _safe_task_name(suite, task_id)
    return {
        "trace": str(Path(trace_dir) / f"{base}_trace.jsonl"),
        "prefix": str(Path(prefix_dir) / f"{base}_prefix_dataset.csv"),
    }


def derive_merged_trace_path(merged_out: str | Path) -> str:
    merged_out = Path(merged_out)
    filename = merged_out.name
    suffix = "_prefix_dataset.csv"
    if filename.endswith(suffix):
        trace_name = filename[: -len(suffix)] + "_trace.jsonl"
    else:
        trace_name = merged_out.stem + "_trace.jsonl"
    return str(merged_out.with_name(trace_name))


def _available_user_tasks(args: argparse.Namespace) -> list[str]:
    payload = list_tasks(args.agentdojo_package, args.benchmark_version, args.suite)
    if payload.get("status") != "ok":
        raise ValueError(f"Could not list AgentDojo tasks: {payload.get('status')}")
    return list(payload["user_tasks"])


def build_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    available_tasks = _available_user_tasks(args)
    selected = select_tasks(available_tasks, args.limit, args.task_start, args.tasks)
    return {
        "mode": "dry_run",
        "status": "ok",
        "suite": args.suite,
        "selected_tasks": selected,
        "task_count": len(selected),
        "outputs": {
            task_id: output_paths_for_task(args.suite, task_id, args.trace_dir, args.prefix_dir)
            for task_id in selected
        },
        "merged_out": args.merged_out,
        "merged_trace": derive_merged_trace_path(args.merged_out),
        "summary": args.summary,
        "provider": _provider_status(args),
        "cost_guard": _cost_guard_status(args),
        "will_call_provider": False,
        "will_run_full_benchmark": False,
        "will_modify_agentdojo_source": False,
        "will_monkey_patch": False,
    }


def _merge_trace_files(trace_paths: list[str], merged_trace: str | Path) -> int:
    merged_trace = Path(merged_trace)
    merged_trace.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with merged_trace.open("w", encoding="utf-8", newline="\n") as out_f:
        for trace_path in trace_paths:
            text = Path(trace_path).read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                out_f.write(line)
                out_f.write("\n")
                line_count += 1
    return line_count


def _merge_prefix_files(prefix_paths: list[str], merged_out: str | Path) -> dict[str, int]:
    merged_out = Path(merged_out)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    column_count = 0
    header: list[str] | None = None
    with merged_out.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        for prefix_path in prefix_paths:
            with Path(prefix_path).open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    current_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = current_header
                    column_count = len(header)
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                elif current_header != header:
                    raise ValueError(f"Prefix columns do not match for {prefix_path}")
                assert writer is not None
                for row in reader:
                    if len(row) != column_count:
                        raise ValueError(f"Prefix row has unexpected column count in {prefix_path}")
                    writer.writerow(row)
                    row_count += 1
    return {"rows": row_count, "columns": column_count}


def _run_one_task(args: argparse.Namespace, task_id: str, trace_path: str, prefix_path: str, plan: Any) -> dict[str, Any]:
    task_args = argparse.Namespace(**vars(args))
    task_args.user_task = task_id
    task_args.trace = trace_path
    result = _run_real_smoke(task_args, plan)

    events = load_trace_events(trace_path)
    tool_call_count = sum(1 for event in events if event.hook_type == "pre_step")
    prefix_df = make_prefix_dataset(trace_path)
    Path(prefix_path).parent.mkdir(parents=True, exist_ok=True)
    prefix_df.to_csv(prefix_path, index=False)

    return {
        "task_id": task_id,
        "status": "ok",
        "trace": trace_path,
        "prefix": prefix_path,
        "trace_event_count": len(events),
        "tool_call_count": tool_call_count,
        "max_tool_calls": args.max_tool_calls,
        "max_tool_calls_hit": args.max_tool_calls is not None and tool_call_count >= args.max_tool_calls,
        "stop_reason": None,
        "prefix_rows": int(len(prefix_df)),
        "prefix_columns": int(len(prefix_df.columns)),
        "utility": result.get("utility"),
        "security": result.get("security"),
    }


def _tool_call_count_from_trace(trace_path: str) -> int:
    path = Path(trace_path)
    if not path.exists():
        return 0
    try:
        events = load_trace_events(path)
    except Exception:
        return 0
    return sum(1 for event in events if event.hook_type == "pre_step")


def run_mini_batch(args: argparse.Namespace) -> dict[str, Any]:
    plan = inspect_agentdojo_pipeline(package_name=args.agentdojo_package)
    available_tasks = _available_user_tasks(args)
    selected = select_tasks(available_tasks, args.limit, args.task_start, args.tasks)

    if not args.allow_real_run:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": "Mini-batch runs require --allow-real-run.",
            "selected_tasks": selected,
            "provider": _provider_status(args),
            "cost_guard": _cost_guard_status(args),
        })
    if not args.allow_provider_call:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": "Mini-batch provider calls require --allow-provider-call.",
            "selected_tasks": selected,
            "provider": _provider_status(args),
            "cost_guard": _cost_guard_status(args),
        })
    if args.provider == "openai-compatible":
        missing = [
            name for name in ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
            if not os.getenv(name)
        ]
        if missing:
            _blocked({
                "mode": "run",
                "status": "blocked",
                "reason": "Missing required OpenAI-compatible provider environment variables.",
                "missing_env": missing,
                "provider": _provider_status(args),
            })

    merged_trace = derive_merged_trace_path(args.merged_out)
    task_results = []
    completed_trace_paths = []
    completed_prefix_paths = []
    for task_id in selected:
        paths = output_paths_for_task(args.suite, task_id, args.trace_dir, args.prefix_dir)
        Path(paths["trace"]).parent.mkdir(parents=True, exist_ok=True)
        Path(paths["prefix"]).parent.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_one_task(args, task_id, paths["trace"], paths["prefix"], plan)
        except MaxToolCallsExceeded as exc:
            tool_call_count = _tool_call_count_from_trace(paths["trace"])
            result = {
                "task_id": task_id,
                "status": "stopped",
                "stop_reason": MAX_TOOL_CALLS_EXCEEDED_REASON,
                "trace": paths["trace"],
                "prefix": paths["prefix"],
                "error": str(exc),
                "tool_call_count": tool_call_count,
                "max_tool_calls": args.max_tool_calls,
                "max_tool_calls_hit": True,
            }
            task_results.append(result)
            if Path(paths["trace"]).exists():
                completed_trace_paths.append(paths["trace"])
            break
        except Exception as exc:
            tool_call_count = _tool_call_count_from_trace(paths["trace"])
            result = {
                "task_id": task_id,
                "status": "failed",
                "stop_reason": "error",
                "trace": paths["trace"],
                "prefix": paths["prefix"],
                "error": f"{type(exc).__name__}: {exc}",
                "tool_call_count": tool_call_count,
                "max_tool_calls": args.max_tool_calls,
                "max_tool_calls_hit": args.max_tool_calls is not None and tool_call_count >= args.max_tool_calls,
            }
            task_results.append(result)
            break
        task_results.append(result)
        completed_trace_paths.append(paths["trace"])
        completed_prefix_paths.append(paths["prefix"])

    merged_trace_rows = _merge_trace_files(completed_trace_paths, merged_trace) if completed_trace_paths else 0
    merged_prefix_shape = (
        _merge_prefix_files(completed_prefix_paths, args.merged_out)
        if completed_prefix_paths
        else {"rows": 0, "columns": 0}
    )

    summary = {
        "mode": "run",
        "status": "ok" if all(result["status"] == "ok" for result in task_results) else "stopped",
        "suite": args.suite,
        "selected_tasks": selected,
        "completed_tasks": [result["task_id"] for result in task_results if result["status"] == "ok"],
        "failed_tasks": [result["task_id"] for result in task_results if result["status"] != "ok"],
        "task_results": task_results,
        "merged_trace": merged_trace,
        "merged_trace_rows": merged_trace_rows,
        "merged_prefix": args.merged_out,
        "merged_prefix_rows": merged_prefix_shape["rows"],
        "merged_prefix_columns": merged_prefix_shape["columns"],
        "provider": _provider_status(args),
        "cost_guard": _cost_guard_status(args),
        "agentdojo": plan.to_dict(),
        "secrets_recorded": False,
        "will_run_full_benchmark": False,
        "will_modify_agentdojo_source": False,
        "will_monkey_patch": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a guarded traced AgentDojo mini batch.")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--task-start", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs.")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--prefix-dir", default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--merged-out", default=DEFAULT_MERGED_OUT)
    parser.add_argument("--merged-trace", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["local", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-real-run", action="store_true")
    parser.add_argument("--allow-provider-call", action="store_true")
    parser.add_argument("--cost-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agentdojo-package", default="agentdojo", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.dry_run:
            print(json.dumps(build_dry_run_report(args), indent=2, sort_keys=True))
            return
        print(json.dumps(run_mini_batch(args), indent=2, sort_keys=True))
    except SystemExit:
        raise
    except Exception as exc:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": f"{type(exc).__name__}: {exc}",
            "provider": _provider_status(args),
            "cost_guard": _cost_guard_status(args),
        })


if __name__ == "__main__":
    main()
