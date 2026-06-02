from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitor.logger import load_trace_events  # noqa: E402
from src.monitor.replay import make_prefix_dataset  # noqa: E402


DEFAULT_MERGED_TRACES = [
    "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl",
    "outputs/agentdojo_mini_batch_round2/merged/workspace_mini_batch_round2_trace.jsonl",
    "outputs/agentdojo_mini_batch_round3/merged/workspace_mini_batch_round3_trace.jsonl",
    "outputs/agentdojo_mini_batch_round4/merged/workspace_mini_batch_round4_trace.jsonl",
    "outputs/agentdojo_mini_batch_slack_round1/merged/slack_mini_batch_round1_trace.jsonl",
    "outputs/agentdojo_mini_batch_slack_recovery1/merged/slack_recovery1_trace.jsonl",
]
DEFAULT_TRACE_DIRS = [
    "outputs/agentdojo_mini_batch/traces",
    "outputs/agentdojo_mini_batch_round2/traces",
    "outputs/agentdojo_mini_batch_round3/traces",
    "outputs/agentdojo_mini_batch_round4/traces",
    "outputs/agentdojo_mini_batch_slack_round1/traces",
    "outputs/agentdojo_mini_batch_slack_recovery1/traces",
]
DEFAULT_JSON_OUT = "outputs/agentdojo_trace_provenance_audit.json"
DEFAULT_CSV_OUT = "outputs/agentdojo_trace_provenance_audit.csv"
DEFAULT_ROUND1_MERGED_TRACE = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl"
DEFAULT_ROUND1_TRACE_DIR = "outputs/agentdojo_mini_batch/traces"
DEFAULT_ROUND1_BACKUP = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.corrupt_backup.jsonl"
DEFAULT_ROUND1_PREFIX = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv"
EXPECTED_TASK_RANGES = {
    "workspace_round1": set(range(0, 5)),
    "workspace_round2": set(range(5, 14)),
    "workspace_round3": set(range(14, 20)),
    "workspace_round4": set(range(20, 40)),
    "slack_round1": set(range(0, 10)),
    "slack_recovery1": {1, 5, 6, 8, 9},
}
EXPECTED_ROUND1_EPISODE_IDS = {
    f"workspace:user_task_{task_id}:none:none"
    for task_id in EXPECTED_TASK_RANGES["workspace_round1"]
}


def _source_batch_from_path(path: str | Path) -> str:
    text = str(path).lower()
    if "slack_recovery1" in text:
        return "slack_recovery1"
    if "slack_round1" in text:
        return "slack_round1"
    if "round4" in text:
        return "workspace_round4"
    if "round3" in text:
        return "workspace_round3"
    if "round2" in text:
        return "workspace_round2"
    if "agentdojo_mini_batch" in text and "slack" not in text:
        return "workspace_round1"
    return "unknown"


def _task_id_from_episode(episode_id: str) -> int | None:
    match = re.search(r"user_task_(\d+)", episode_id)
    if not match:
        return None
    return int(match.group(1))


def _suite_from_episode(episode_id: str) -> str:
    return episode_id.split(":", 1)[0] if ":" in episode_id else "unknown"


def _summarize_trace_file(path: str | Path, source_kind: str, expected_source_batch: str | None = None) -> dict[str, Any]:
    path = Path(path)
    expected_source_batch = expected_source_batch or _source_batch_from_path(path)
    if not path.exists():
        return {
            "source_path": str(path),
            "source_kind": source_kind,
            "expected_source_batch": expected_source_batch,
            "exists": False,
            "event_count": 0,
            "episode_count": 0,
            "episode_ids": [],
            "task_ids": [],
            "task_ids_match_expected_batch_range": False,
            "suspicious": True,
            "suspicion_reasons": ["missing_path"],
        }

    events = load_trace_events(path)
    episode_ids = sorted({event.episode_id for event in events})
    task_ids = sorted({task_id for episode_id in episode_ids if (task_id := _task_id_from_episode(episode_id)) is not None})
    expected_range = EXPECTED_TASK_RANGES.get(expected_source_batch, set())
    out_of_range_task_ids = sorted(set(task_ids) - expected_range) if expected_range else []
    missing_expected_task_ids = sorted(expected_range - set(task_ids)) if expected_range else []
    suites = sorted({_suite_from_episode(episode_id) for episode_id in episode_ids})
    reasons = []
    if out_of_range_task_ids:
        reasons.append("contains_task_ids_outside_expected_batch_range")
    if expected_range and source_kind == "trace_dir" and missing_expected_task_ids:
        reasons.append("trace_dir_missing_expected_task_ids")
    if expected_source_batch.startswith("workspace") and suites and suites != ["workspace"]:
        reasons.append("workspace_batch_contains_non_workspace_episode")
    if expected_source_batch.startswith("slack") and suites and suites != ["slack"]:
        reasons.append("slack_batch_contains_non_slack_episode")
    return {
        "source_path": str(path),
        "source_kind": source_kind,
        "expected_source_batch": expected_source_batch,
        "exists": True,
        "event_count": len(events),
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "task_ids": task_ids,
        "task_ids_match_expected_batch_range": bool(expected_range) and not out_of_range_task_ids,
        "out_of_range_task_ids": out_of_range_task_ids,
        "missing_expected_task_ids": missing_expected_task_ids,
        "suites": suites,
        "suspicious": bool(reasons),
        "suspicion_reasons": reasons,
    }


def _summarize_trace_dir(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    expected_source_batch = _source_batch_from_path(path)
    if not path.exists():
        return _summarize_trace_file(path, "trace_dir", expected_source_batch)

    jsonl_files = sorted(path.glob("*.jsonl"))
    events = []
    for trace_file in jsonl_files:
        events.extend(load_trace_events(trace_file))
    tmp_summary = _summarize_events(
        events=events,
        path=path,
        source_kind="trace_dir",
        expected_source_batch=expected_source_batch,
    )
    tmp_summary["file_count"] = len(jsonl_files)
    tmp_summary["trace_files"] = [str(trace_file) for trace_file in jsonl_files]
    return tmp_summary


def _trace_files_by_task_id(trace_dir: Path) -> dict[int, Path]:
    trace_files_by_task_id: dict[int, Path] = {}
    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        events = load_trace_events(trace_file)
        episode_ids = {event.episode_id for event in events}
        task_ids = {
            task_id
            for episode_id in episode_ids
            if (task_id := _task_id_from_episode(episode_id)) is not None
        }
        if len(task_ids) != 1:
            raise ValueError(f"{trace_file} must contain exactly one task id; found {sorted(task_ids)}.")
        task_id = next(iter(task_ids))
        if task_id in trace_files_by_task_id:
            raise ValueError(f"Duplicate per-task trace for workspace user_task_{task_id}.")
        trace_files_by_task_id[task_id] = trace_file
    return trace_files_by_task_id


def repair_round1_merged_trace(
    merged_trace_path: str | Path = DEFAULT_ROUND1_MERGED_TRACE,
    trace_dir: str | Path = DEFAULT_ROUND1_TRACE_DIR,
    backup_path: str | Path = DEFAULT_ROUND1_BACKUP,
    prefix_path: str | Path = DEFAULT_ROUND1_PREFIX,
) -> dict[str, Any]:
    merged_trace_path = Path(merged_trace_path)
    trace_dir = Path(trace_dir)
    backup_path = Path(backup_path)
    prefix_path = Path(prefix_path)
    if not merged_trace_path.exists():
        raise FileNotFoundError(f"Missing round1 merged trace to repair: {merged_trace_path}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trusted round1 per-task trace directory: {trace_dir}")

    trace_files_by_task_id = _trace_files_by_task_id(trace_dir)
    actual_task_ids = set(trace_files_by_task_id)
    expected_task_ids = EXPECTED_TASK_RANGES["workspace_round1"]
    if actual_task_ids != expected_task_ids:
        raise ValueError(
            "Trusted round1 per-task traces do not match expected tasks; "
            f"expected {sorted(expected_task_ids)}, found {sorted(actual_task_ids)}."
        )

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(merged_trace_path, backup_path)
    merged_trace_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_trace_path.open("w", encoding="utf-8") as out_file:
        for task_id in sorted(expected_task_ids):
            text = trace_files_by_task_id[task_id].read_text(encoding="utf-8").strip()
            if text:
                out_file.write(text)
                out_file.write("\n")

    repaired_events = load_trace_events(merged_trace_path)
    repaired_episode_ids = sorted({event.episode_id for event in repaired_events})
    if set(repaired_episode_ids) != EXPECTED_ROUND1_EPISODE_IDS:
        raise ValueError(
            "Repaired round1 merged trace contains unexpected episodes: "
            f"{repaired_episode_ids}."
        )

    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_df = make_prefix_dataset(str(merged_trace_path))
    prefix_df.to_csv(prefix_path, index=False)
    return {
        "status": "repaired",
        "merged_trace": str(merged_trace_path),
        "backup": str(backup_path),
        "trusted_trace_dir": str(trace_dir),
        "trusted_trace_files": [str(trace_files_by_task_id[task_id]) for task_id in sorted(expected_task_ids)],
        "repaired_task_ids": sorted(expected_task_ids),
        "repaired_episode_ids": repaired_episode_ids,
        "event_count": len(repaired_events),
        "prefix_output": str(prefix_path),
        "prefix_rows": int(len(prefix_df)),
        "prefix_episodes": int(prefix_df["episode_id"].nunique()),
    }


def _summarize_events(
    events,
    path: str | Path,
    source_kind: str,
    expected_source_batch: str,
) -> dict[str, Any]:
    episode_ids = sorted({event.episode_id for event in events})
    task_ids = sorted({task_id for episode_id in episode_ids if (task_id := _task_id_from_episode(episode_id)) is not None})
    expected_range = EXPECTED_TASK_RANGES.get(expected_source_batch, set())
    out_of_range_task_ids = sorted(set(task_ids) - expected_range) if expected_range else []
    missing_expected_task_ids = sorted(expected_range - set(task_ids)) if expected_range else []
    suites = sorted({_suite_from_episode(episode_id) for episode_id in episode_ids})
    reasons = []
    if out_of_range_task_ids:
        reasons.append("contains_task_ids_outside_expected_batch_range")
    if expected_range and source_kind == "trace_dir" and missing_expected_task_ids:
        reasons.append("trace_dir_missing_expected_task_ids")
    if expected_source_batch.startswith("workspace") and suites and suites != ["workspace"]:
        reasons.append("workspace_batch_contains_non_workspace_episode")
    if expected_source_batch.startswith("slack") and suites and suites != ["slack"]:
        reasons.append("slack_batch_contains_non_slack_episode")
    return {
        "source_path": str(path),
        "source_kind": source_kind,
        "expected_source_batch": expected_source_batch,
        "exists": True,
        "event_count": len(events),
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "task_ids": task_ids,
        "task_ids_match_expected_batch_range": bool(expected_range) and not out_of_range_task_ids,
        "out_of_range_task_ids": out_of_range_task_ids,
        "missing_expected_task_ids": missing_expected_task_ids,
        "suites": suites,
        "suspicious": bool(reasons),
        "suspicion_reasons": reasons,
    }


def _duplicate_episodes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        if record.get("source_kind") != "merged_trace" or not record.get("exists"):
            continue
        for episode_id in record.get("episode_ids", []):
            locations[episode_id].append({
                "source_batch": record["expected_source_batch"],
                "source_path": record["source_path"],
            })
    duplicates = []
    for episode_id, refs in sorted(locations.items()):
        batches = sorted({ref["source_batch"] for ref in refs})
        if len(batches) > 1:
            allowed_stopped_recovery = set(batches) == {"slack_recovery1", "slack_round1"}
            duplicates.append({
                "episode_id": episode_id,
                "source_batches": batches,
                "locations": refs,
                "allowed_stopped_recovery_duplicate": allowed_stopped_recovery,
            })
    return duplicates


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "source_path": record["source_path"],
            "source_kind": record["source_kind"],
            "expected_source_batch": record["expected_source_batch"],
            "exists": record["exists"],
            "event_count": record["event_count"],
            "episode_count": record["episode_count"],
            "task_ids": ",".join(str(task_id) for task_id in record.get("task_ids", [])),
            "task_ids_match_expected_batch_range": record.get("task_ids_match_expected_batch_range", False),
            "out_of_range_task_ids": ",".join(str(task_id) for task_id in record.get("out_of_range_task_ids", [])),
            "missing_expected_task_ids": ",".join(str(task_id) for task_id in record.get("missing_expected_task_ids", [])),
            "suspicious": record["suspicious"],
            "suspicion_reasons": ";".join(record.get("suspicion_reasons", [])),
        })
    return rows


def audit_agentdojo_trace_provenance(
    merged_traces: list[str | Path] | None = None,
    trace_dirs: list[str | Path] | None = None,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
) -> dict[str, Any]:
    merged_traces = list(merged_traces or DEFAULT_MERGED_TRACES)
    trace_dirs = list(trace_dirs or DEFAULT_TRACE_DIRS)
    records = [
        _summarize_trace_file(path, "merged_trace")
        for path in merged_traces
    ] + [
        _summarize_trace_dir(path)
        for path in trace_dirs
    ]
    duplicate_episode_ids = _duplicate_episodes(records)
    suspicious_records = [record for record in records if record.get("suspicious")]
    workspace_round1_record = next(
        (
            record
            for record in records
            if record.get("source_kind") == "merged_trace"
            and record.get("expected_source_batch") == "workspace_round1"
        ),
        {},
    )
    workspace_round1_appears_to_contain_round2_tasks = any(
        task_id in EXPECTED_TASK_RANGES["workspace_round2"]
        for task_id in workspace_round1_record.get("task_ids", [])
    )
    result = {
        "records": records,
        "duplicate_episode_ids_across_batches": duplicate_episode_ids,
        "duplicate_episode_count_across_batches": len(duplicate_episode_ids),
        "suspicious_files": [
            {
                "source_path": record["source_path"],
                "expected_source_batch": record["expected_source_batch"],
                "source_kind": record["source_kind"],
                "suspicion_reasons": record.get("suspicion_reasons", []),
                "task_ids": record.get("task_ids", []),
                "out_of_range_task_ids": record.get("out_of_range_task_ids", []),
            }
            for record in suspicious_records
        ],
        "workspace_round1_merged_appears_to_contain_round2_tasks": workspace_round1_appears_to_contain_round2_tasks,
        "outputs": {
            "json": str(json_out),
            "csv": str(csv_out),
        },
    }
    json_out = Path(json_out)
    csv_out = Path(csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(_csv_rows(records)).to_csv(csv_out, index=False)
    return result


def audit_and_repair_round1_trace(
    merged_traces: list[str | Path] | None = None,
    trace_dirs: list[str | Path] | None = None,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    merged_trace_path: str | Path = DEFAULT_ROUND1_MERGED_TRACE,
    trace_dir: str | Path = DEFAULT_ROUND1_TRACE_DIR,
    backup_path: str | Path = DEFAULT_ROUND1_BACKUP,
    prefix_path: str | Path = DEFAULT_ROUND1_PREFIX,
) -> dict[str, Any]:
    before = audit_agentdojo_trace_provenance(
        merged_traces=merged_traces,
        trace_dirs=trace_dirs,
        json_out=json_out,
        csv_out=csv_out,
    )
    repair = repair_round1_merged_trace(
        merged_trace_path=merged_trace_path,
        trace_dir=trace_dir,
        backup_path=backup_path,
        prefix_path=prefix_path,
    )
    after = audit_agentdojo_trace_provenance(
        merged_traces=merged_traces,
        trace_dirs=trace_dirs,
        json_out=json_out,
        csv_out=csv_out,
    )
    result = {
        **after,
        "repair": repair,
        "before_repair": {
            "duplicate_episode_count_across_batches": before["duplicate_episode_count_across_batches"],
            "duplicate_episode_ids_across_batches": before["duplicate_episode_ids_across_batches"],
            "suspicious_files": before["suspicious_files"],
            "workspace_round1_merged_appears_to_contain_round2_tasks": before[
                "workspace_round1_merged_appears_to_contain_round2_tasks"
            ],
        },
        "after_repair": {
            "duplicate_episode_count_across_batches": after["duplicate_episode_count_across_batches"],
            "duplicate_episode_ids_across_batches": after["duplicate_episode_ids_across_batches"],
            "suspicious_files": after["suspicious_files"],
            "workspace_round1_merged_appears_to_contain_round2_tasks": after[
                "workspace_round1_merged_appears_to_contain_round2_tasks"
            ],
        },
    }
    Path(json_out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AgentDojo trace provenance and batch/task consistency.")
    parser.add_argument("--merged-trace", action="append", dest="merged_traces", default=None)
    parser.add_argument("--trace-dir", action="append", dest="trace_dirs", default=None)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--repair-round1", action="store_true")
    parser.add_argument("--round1-merged-trace", default=DEFAULT_ROUND1_MERGED_TRACE)
    parser.add_argument("--round1-trace-dir", default=DEFAULT_ROUND1_TRACE_DIR)
    parser.add_argument("--round1-backup", default=DEFAULT_ROUND1_BACKUP)
    parser.add_argument("--round1-prefix", default=DEFAULT_ROUND1_PREFIX)
    args = parser.parse_args()

    if args.repair_round1:
        result = audit_and_repair_round1_trace(
            merged_traces=args.merged_traces,
            trace_dirs=args.trace_dirs,
            json_out=args.json_out,
            csv_out=args.csv_out,
            merged_trace_path=args.round1_merged_trace,
            trace_dir=args.round1_trace_dir,
            backup_path=args.round1_backup,
            prefix_path=args.round1_prefix,
        )
    else:
        result = audit_agentdojo_trace_provenance(
            merged_traces=args.merged_traces,
            trace_dirs=args.trace_dirs,
            json_out=args.json_out,
            csv_out=args.csv_out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
