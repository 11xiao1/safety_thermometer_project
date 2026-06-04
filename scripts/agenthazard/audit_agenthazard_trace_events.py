from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitor.schema import TraceEvent  # noqa: E402


DEFAULT_TRACE_EVENTS = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_events.jsonl"
DEFAULT_CONVERSION_SUMMARY = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_conversion_summary.json"
DEFAULT_METADATA = "external/AgentHazard/data/dataset.json"
DEFAULT_AUDIT_JSON = "outputs/agenthazard/agenthazard_trace_event_audit.json"
DEFAULT_AUDIT_CSV = "outputs/agenthazard/agenthazard_trace_event_audit.csv"
LARGE_FIELD_WARNING_BYTES = 32_000


def _parse_notes(raw_notes: str | None) -> dict[str, Any]:
    if not raw_notes:
        return {}
    try:
        payload = json.loads(raw_notes)
    except json.JSONDecodeError:
        return {"_notes_parse_error": True}
    return payload if isinstance(payload, dict) else {}


def _task_id_from_event(event: TraceEvent, notes: dict[str, Any]) -> str | None:
    task_id = notes.get("task_id") or notes.get("id") or notes.get("original_id")
    if task_id is not None:
        return str(task_id)
    parts = event.episode_id.split(":")
    return parts[-1] if parts else None


def _load_agenthazard_metadata(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata_by_task: dict[str, dict[str, Any]] = {}
    metadata_by_task_category: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: Counter[str] = Counter()
    fields: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        fields.update(str(key) for key in row.keys())
        keys = []
        for key in [row.get("id"), row.get("instance_id"), row.get("original_id")]:
            if key is not None and key not in keys:
                keys.append(key)
        for key in keys:
            text_key = str(key)
            category = row.get("category") or row.get("risk_category")
            if category is not None:
                metadata_by_task_category.setdefault((text_key, str(category)), row)
            if text_key in metadata_by_task:
                duplicate_keys[text_key] += 1
            metadata_by_task.setdefault(text_key, row)
    return metadata_by_task, metadata_by_task_category, {
        "metadata_record_count": len(rows) if isinstance(rows, list) else 0,
        "metadata_fields_available": sorted(fields),
        "duplicate_metadata_keys": dict(sorted(duplicate_keys.items())),
    }


def _metadata_match(task_id: str | None, risk_category: str | None, by_task: dict[str, dict[str, Any]], by_task_category: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    if task_id is None:
        return None
    if risk_category is not None:
        match = by_task_category.get((str(task_id), str(risk_category)))
        if match is not None:
            return match
    return by_task.get(str(task_id))


def _write_audit_csv(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in audit["scalar_counts"].items():
        rows.append({"metric": key, "key": "", "value": value})
    for section in [
        "events_by_agent_family",
        "episodes_by_agent_family",
        "events_by_source_archive",
        "trajectory_file_count_by_source_archive",
        "hook_type_counts",
        "metadata_join_coverage",
    ]:
        for key, value in audit.get(section, {}).items():
            rows.append({"metric": section, "key": key, "value": value})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)


def audit_agenthazard_trace_events(
    trace_events_jsonl: str | Path = DEFAULT_TRACE_EVENTS,
    conversion_summary_json: str | Path = DEFAULT_CONVERSION_SUMMARY,
    metadata_json: str | Path = DEFAULT_METADATA,
    audit_json: str | Path = DEFAULT_AUDIT_JSON,
    audit_csv: str | Path = DEFAULT_AUDIT_CSV,
) -> dict[str, Any]:
    metadata_by_task, metadata_by_task_category, metadata_report = _load_agenthazard_metadata(metadata_json)
    summary = json.loads(Path(conversion_summary_json).read_text(encoding="utf-8"))

    scalar = Counter()
    malformed_examples = []
    large_field_warnings = []
    hook_type_counts: Counter[str] = Counter()
    events_by_agent_family: Counter[str] = Counter()
    events_by_source_archive: Counter[str] = Counter()
    trajectory_files_by_archive: dict[str, set[str]] = {}
    episode_ids: set[str] = set()
    task_ids: set[str] = set()
    episodes_by_family: dict[str, set[str]] = {}
    matched_tasks: set[str] = set()
    unmatched_tasks: set[str] = set()
    semantic_keys: set[tuple[str, str, str, str, int, str]] = set()
    duplicate_examples = []

    with Path(trace_events_jsonl).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = TraceEvent.model_validate(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - keep auditing after malformed lines.
                scalar["malformed_jsonl_line_count"] += 1
                if len(malformed_examples) < 10:
                    malformed_examples.append({"line_no": line_no, "error": f"{type(exc).__name__}: {exc}"})
                continue

            scalar["total_event_count"] += 1
            notes = _parse_notes(event.notes)
            family = str(notes.get("agent_family") or notes.get("source_agent_family") or "unknown")
            source_archive = str(notes.get("source_batch") or "unknown")
            source_trace_path = str(notes.get("source_trace_path") or "")
            task_id = _task_id_from_event(event, notes)
            risk_category = notes.get("risk_category")
            benchmark = str(notes.get("benchmark_name") or "agenthazard")

            if not event.episode_id:
                scalar["missing_episode_id_count"] += 1
            else:
                episode_ids.add(event.episode_id)
                episodes_by_family.setdefault(family, set()).add(event.episode_id)
            if task_id is None:
                scalar["missing_task_id_count"] += 1
            else:
                task_ids.add(task_id)
                if _metadata_match(task_id, risk_category, metadata_by_task, metadata_by_task_category):
                    matched_tasks.add(task_id)
                else:
                    unmatched_tasks.add(task_id)
            if not event.user_instruction:
                scalar["missing_user_instruction_count"] += 1
            if not event.proposed_tool:
                scalar["missing_proposed_tool_action_count"] += 1
            if not event.observation:
                scalar["missing_observation_count"] += 1
            if event.hook_type == "final":
                scalar["final_event_count"] += 1
            if source_trace_path:
                scalar["source_trace_path_present_count"] += 1
                trajectory_files_by_archive.setdefault(source_archive, set()).add(source_trace_path)
            else:
                scalar["missing_source_trace_path_count"] += 1
            if event.step_id is None:
                scalar["missing_step_id_count"] += 1

            for field_name, field_value in [
                ("user_instruction", event.user_instruction),
                ("plan_summary", event.plan_summary),
                ("observation", event.observation),
                ("notes", event.notes),
            ]:
                if field_value and len(str(field_value).encode("utf-8")) > LARGE_FIELD_WARNING_BYTES:
                    scalar["large_field_warning_count"] += 1
                    if len(large_field_warnings) < 20:
                        large_field_warnings.append(
                            {
                                "line_no": line_no,
                                "episode_id": event.episode_id,
                                "field": field_name,
                                "byte_size": len(str(field_value).encode("utf-8")),
                            }
                        )

            semantic_key = (benchmark, family, str(task_id), event.episode_id, event.step_id, event.hook_type)
            if semantic_key in semantic_keys:
                scalar["duplicate_semantic_event_count"] += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append(
                        {
                            "benchmark_name": benchmark,
                            "source_agent_family": family,
                            "task_id": task_id,
                            "episode_id": event.episode_id,
                            "step_id": event.step_id,
                            "hook_type": event.hook_type,
                        }
                    )
            else:
                semantic_keys.add(semantic_key)

            hook_type_counts[event.hook_type] += 1
            events_by_agent_family[family] += 1
            events_by_source_archive[source_archive] += 1

    scalar["total_episode_count"] = len(episode_ids)
    scalar["total_task_count"] = len(task_ids)
    scalar["step_id_present_count"] = scalar["total_event_count"] - scalar["missing_step_id_count"]
    scalar["source_trace_path_coverage_count"] = scalar["source_trace_path_present_count"]
    scalar["converted_event_count_matches_summary"] = int(scalar["total_event_count"] == int(summary.get("event_count", -1)))
    for required_zeroable_count in [
        "duplicate_semantic_event_count",
        "malformed_jsonl_line_count",
        "missing_episode_id_count",
        "missing_task_id_count",
        "missing_user_instruction_count",
        "missing_source_trace_path_count",
        "missing_step_id_count",
        "large_field_warning_count",
        "missing_proposed_tool_action_count",
        "missing_observation_count",
        "final_event_count",
    ]:
        scalar.setdefault(required_zeroable_count, 0)

    join_coverage = {
        "matched_tasks": len(matched_tasks),
        "unmatched_tasks": len(unmatched_tasks),
        "duplicate_metadata_key_count": len(metadata_report["duplicate_metadata_keys"]),
        "metadata_record_count": metadata_report["metadata_record_count"],
    }
    audit = {
        "status": "ok",
        "trace_events_jsonl": str(trace_events_jsonl),
        "conversion_summary_json": str(conversion_summary_json),
        "metadata_json": str(metadata_json),
        "scalar_counts": dict(sorted(scalar.items())),
        "events_by_agent_family": dict(sorted(events_by_agent_family.items())),
        "episodes_by_agent_family": {key: len(value) for key, value in sorted(episodes_by_family.items())},
        "events_by_source_archive": dict(sorted(events_by_source_archive.items())),
        "trajectory_file_count_by_source_archive": {key: len(value) for key, value in sorted(trajectory_files_by_archive.items())},
        "hook_type_counts": dict(sorted(hook_type_counts.items())),
        "metadata_join_coverage": join_coverage,
        "metadata_fields_available": metadata_report["metadata_fields_available"],
        "duplicate_metadata_keys": metadata_report["duplicate_metadata_keys"],
        "malformed_jsonl_examples": malformed_examples,
        "duplicate_semantic_event_examples": duplicate_examples,
        "large_field_warnings": large_field_warnings,
        "labels": {
            "label_status": "missing_pending_judge",
            "future_risk_label_available": False,
            "future_severity_available": False,
        },
        "conversion_summary_expected": {
            "archive_count": summary.get("archive_count"),
            "trajectory_file_count": summary.get("trajectory_file_count"),
            "event_count": summary.get("event_count"),
            "included_families": summary.get("included_families"),
            "excluded_families": summary.get("excluded_families"),
        },
        "outputs": {
            "audit_json": str(audit_json),
            "audit_csv": str(audit_csv),
        },
        "rules": {
            "will_call_provider": False,
            "will_run_agenthazard": False,
            "will_run_agentdojo": False,
            "will_train": False,
            "will_calibrate": False,
            "test_split_used": False,
        },
    }
    audit_json = Path(audit_json)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_audit_csv(Path(audit_csv), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-audit converted AgentHazard TraceEvent JSONL and metadata join coverage.")
    parser.add_argument("--trace-events", default=DEFAULT_TRACE_EVENTS)
    parser.add_argument("--conversion-summary", default=DEFAULT_CONVERSION_SUMMARY)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--audit-json", default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV)
    args = parser.parse_args()
    result = audit_agenthazard_trace_events(
        trace_events_jsonl=args.trace_events,
        conversion_summary_json=args.conversion_summary,
        metadata_json=args.metadata,
        audit_json=args.audit_json,
        audit_csv=args.audit_csv,
    )
    print(json.dumps({
        "status": result["status"],
        "scalar_counts": result["scalar_counts"],
        "metadata_join_coverage": result["metadata_join_coverage"],
        "outputs": result["outputs"],
        "rules": result["rules"],
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
