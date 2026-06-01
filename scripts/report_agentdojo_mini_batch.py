from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = "outputs/agentdojo_mini_batch/run_summary.json"
DEFAULT_TRACE = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl"
DEFAULT_PREFIX = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv"
DEFAULT_REPORT = "reports/agentdojo_mini_batch_report.md"


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _load_summary(path: str | Path) -> dict[str, Any]:
    path = _require_file(path, "run summary")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace_stats(path: str | Path) -> dict[str, Any]:
    path = _require_file(path, "merged trace JSONL")
    hook_counts: Counter[str] = Counter()
    tools: list[str] = []
    row_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in merged trace at line {line_number}: {exc}") from exc
        row_count += 1
        hook_type = str(event.get("hook_type", ""))
        if hook_type:
            hook_counts[hook_type] += 1
        if hook_type == "pre_step" and event.get("proposed_tool"):
            tools.append(str(event["proposed_tool"]))
    return {
        "trace_event_rows": row_count,
        "hook_type_counts": dict(sorted(hook_counts.items())),
        "tool_call_count": len(tools),
        "unique_tools": sorted(set(tools)),
    }


def _column_distribution(rows: list[dict[str, str]], column: str) -> dict[str, int] | None:
    if not rows or column not in rows[0]:
        return None
    counts = Counter(row.get(column, "") for row in rows)
    return dict(sorted(counts.items()))


def _load_prefix_stats(path: str | Path) -> dict[str, Any]:
    path = _require_file(path, "merged prefix CSV")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    return {
        "prefix_rows": len(rows),
        "prefix_columns": len(columns),
        "future_risk_label_distribution": _column_distribution(rows, "future_risk_label"),
        "oracle_violation_distribution": _column_distribution(rows, "oracle_violation"),
    }


def _task_status_lines(task_results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for result in task_results:
        task_id = result.get("task_id", "unknown")
        status = result.get("status", "unknown")
        details = []
        if "utility" in result:
            details.append(f"utility={result.get('utility')}")
        if "security" in result:
            details.append(f"security={result.get('security')}")
        if result.get("error"):
            details.append(f"error={result['error']}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- `{task_id}`: `{status}`{suffix}")
    return lines


def _format_distribution(distribution: dict[str, int] | None) -> str:
    if distribution is None:
        return "not present"
    if not distribution:
        return "empty"
    return ", ".join(f"`{key}`: {value}" for key, value in distribution.items())


def build_report(
    summary_path: str | Path = DEFAULT_SUMMARY,
    trace_path: str | Path = DEFAULT_TRACE,
    prefix_path: str | Path = DEFAULT_PREFIX,
) -> str:
    summary = _load_summary(summary_path)
    trace_stats = _load_trace_stats(trace_path)
    prefix_stats = _load_prefix_stats(prefix_path)

    selected_tasks = list(summary.get("selected_tasks", []))
    completed_tasks = list(summary.get("completed_tasks", []))
    failed_tasks = list(summary.get("failed_tasks", []))
    skipped_tasks = list(summary.get("skipped_tasks", []))
    task_results = list(summary.get("task_results", []))
    cost_guard = summary.get("cost_guard", {}) or {}

    failure_lines = []
    if summary.get("status") not in (None, "ok"):
        failure_lines.append(f"- Run status: `{summary.get('status')}`")
    for result in task_results:
        if result.get("status") != "ok":
            failure_lines.append(
                f"- `{result.get('task_id', 'unknown')}`: {result.get('error', 'failed without error detail')}"
            )
    if not failure_lines:
        failure_lines.append("- None recorded.")

    utility_values = [
        result.get("utility")
        for result in task_results
        if "utility" in result and result.get("utility") is not None
    ]
    security_values = [
        result.get("security")
        for result in task_results
        if "security" in result and result.get("security") is not None
    ]

    lines = [
        "# AgentDojo Mini-Batch Report",
        "",
        "This is a mini-batch smoke report, not a benchmark-scale result.",
        "",
        "## Inputs",
        "",
        f"- Run summary: `{summary_path}`",
        f"- Merged trace: `{trace_path}`",
        f"- Merged prefix dataset: `{prefix_path}`",
        "",
        "## Task Summary",
        "",
        f"- Selected task count: {len(selected_tasks)}",
        f"- Completed task count: {len(completed_tasks)}",
        f"- Failed task count: {len(failed_tasks)}",
        f"- Skipped task count: {len(skipped_tasks)}",
        f"- Selected task IDs: {', '.join(f'`{task}`' for task in selected_tasks) if selected_tasks else 'none'}",
        "",
        "## Per-Task Status",
        "",
        *(_task_status_lines(task_results) or ["- No per-task status available."]),
        "",
        "## Trace Summary",
        "",
        f"- TraceEvent rows: {trace_stats['trace_event_rows']}",
        f"- Hook type counts: {_format_distribution(trace_stats['hook_type_counts'])}",
        f"- Tool call count: {trace_stats['tool_call_count']}",
        f"- Unique tools called: {', '.join(f'`{tool}`' for tool in trace_stats['unique_tools']) if trace_stats['unique_tools'] else 'none'}",
        "",
        "## Prefix Dataset Summary",
        "",
        f"- Merged prefix rows: {prefix_stats['prefix_rows']}",
        f"- Merged prefix columns: {prefix_stats['prefix_columns']}",
        f"- `future_risk_label` distribution: {_format_distribution(prefix_stats['future_risk_label_distribution'])}",
        f"- `oracle_violation` distribution: {_format_distribution(prefix_stats['oracle_violation_distribution'])}",
        "",
        "## Utility/Security Diagnostics",
        "",
        f"- Utility values: {', '.join(f'`{value}`' for value in utility_values) if utility_values else 'not available'}",
        f"- Security values: {', '.join(f'`{value}`' for value in security_values) if security_values else 'not available'}",
        "",
        "## Cost Guard Settings",
        "",
        f"- `max_steps`: {cost_guard.get('max_steps', 'not available')}",
        f"- `max_tool_calls`: {cost_guard.get('max_tool_calls', 'not available')}",
        f"- `max_output_tokens`: {cost_guard.get('max_output_tokens', 'not available')}",
        f"- `temperature`: {cost_guard.get('temperature', 'not available')}",
        "",
        "## Stop Or Failure Summary",
        "",
        *failure_lines,
        "",
        "## Limitations",
        "",
        "- Mini batch only.",
        "- No benchmark-scale claim.",
        "- No calibration claim on AgentDojo yet.",
        "- Results depend on selected tasks and model behavior.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    summary_path: str | Path = DEFAULT_SUMMARY,
    trace_path: str | Path = DEFAULT_TRACE,
    prefix_path: str | Path = DEFAULT_PREFIX,
    out_path: str | Path = DEFAULT_REPORT,
) -> str:
    report = build_report(summary_path, trace_path, prefix_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline AgentDojo mini-batch report.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out", default=DEFAULT_REPORT)
    args = parser.parse_args()

    try:
        out_path = write_report(args.summary, args.trace, args.prefix, args.out)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "will_call_provider": False,
                    "will_run_agentdojo": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(json.dumps({"status": "ok", "report": out_path}, indent=2))


if __name__ == "__main__":
    main()
