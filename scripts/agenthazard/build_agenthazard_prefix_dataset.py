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

from src.features.disagreement import stream_q_features  # noqa: E402
from src.features.extractor import extract_step_features  # noqa: E402
from src.monitor.schema import TraceEvent  # noqa: E402


DEFAULT_TRACE_EVENTS = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_events.jsonl"
DEFAULT_METADATA = "external/AgentHazard/data/dataset.json"
DEFAULT_PREFIX_CSV = "outputs/agenthazard/agenthazard_prefix_dataset_preview.csv"
DEFAULT_MANIFEST_JSON = "outputs/agenthazard/agenthazard_prefix_dataset_preview_manifest.json"
DEFAULT_MAX_ROWS = 5000


def _parse_notes(raw_notes: str | None) -> dict[str, Any]:
    if not raw_notes:
        return {}
    try:
        payload = json.loads(raw_notes)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_id_from_event(event: TraceEvent, notes: dict[str, Any]) -> str | None:
    task_id = notes.get("task_id") or notes.get("id") or notes.get("original_id")
    if task_id is not None:
        return str(task_id)
    parts = event.episode_id.split(":")
    return parts[-1] if parts else None


def _load_metadata(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], int]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, Any]] = {}
    metadata_by_task_category: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        for key in [row.get("id"), row.get("instance_id"), row.get("original_id")]:
            if key is not None:
                text_key = str(key)
                category = row.get("category") or row.get("risk_category")
                if category is not None:
                    metadata_by_task_category.setdefault((text_key, str(category)), row)
                metadata.setdefault(text_key, row)
    return metadata, metadata_by_task_category, len(rows) if isinstance(rows, list) else 0


def _metadata_match(
    task_id: str | None,
    risk_category: str | None,
    by_task: dict[str, dict[str, Any]],
    by_task_category: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    if task_id is None:
        return None
    if risk_category is not None:
        match = by_task_category.get((str(task_id), str(risk_category)))
        if match is not None:
            return match
    return by_task.get(str(task_id))


def _fieldnames() -> list[str]:
    columns = [
        "benchmark_name",
        "benchmark_role",
        "source_agent_family",
        "source_archive",
        "source_trace_path",
        "episode_id",
        "task_id",
        "step_id",
        "hook_type",
        "risk_category",
        "attack_strategy",
        "environment",
        "instruction",
        "metadata_match",
        "label_status",
        "future_risk_label",
        "future_severity",
        "expected_harmfulness",
        "benchmark_severity",
        "benchmark_outcome",
        "proposed_tool",
        "has_observation",
        "observation_chars",
        *list(extract_step_features(TraceEvent(episode_id="preview", step_id=0, hook_type="checkpoint")).keys()),
        *list(stream_q_features(TraceEvent(episode_id="preview", step_id=0, hook_type="checkpoint")).keys()),
    ]
    return list(dict.fromkeys(columns))


def _metadata_value(metadata: dict[str, Any] | None, *keys: str) -> Any:
    if not metadata:
        return ""
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return ""


def _preview_row(
    event: TraceEvent,
    metadata_by_task: dict[str, dict[str, Any]],
    metadata_by_task_category: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    notes = _parse_notes(event.notes)
    task_id = _task_id_from_event(event, notes)
    note_risk_category = notes.get("risk_category")
    metadata = _metadata_match(task_id, note_risk_category, metadata_by_task, metadata_by_task_category)
    risk_category = (metadata or {}).get("category") or note_risk_category
    attack_strategy = (metadata or {}).get("jailbreak_method") or notes.get("attack_strategy")
    instruction = (metadata or {}).get("query") or event.user_instruction
    features = extract_step_features(event)
    q_features = stream_q_features(event)
    row = {
        "benchmark_name": "agenthazard",
        "benchmark_role": "primary",
        "source_agent_family": notes.get("agent_family") or "",
        "source_archive": notes.get("source_batch") or "",
        "source_trace_path": notes.get("source_trace_path") or "",
        "episode_id": event.episode_id,
        "task_id": task_id or "",
        "step_id": event.step_id,
        "hook_type": event.hook_type,
        "risk_category": risk_category or "",
        "attack_strategy": attack_strategy or "",
        "environment": (metadata or {}).get("environment") or notes.get("environment") or "",
        "instruction": instruction or "",
        "metadata_match": int(metadata is not None),
        "label_status": "missing_pending_judge",
        "future_risk_label": "",
        "future_severity": "",
        "expected_harmfulness": _metadata_value(metadata, "harmful", "label"),
        "benchmark_severity": _metadata_value(metadata, "severity", "score"),
        "benchmark_outcome": _metadata_value(metadata, "final_outcome", "outcome"),
        "proposed_tool": event.proposed_tool or "",
        "has_observation": int(bool(event.observation)),
        "observation_chars": len(event.observation or ""),
        **features,
        **q_features,
    }
    return row


def build_agenthazard_prefix_dataset_preview(
    trace_events_jsonl: str | Path = DEFAULT_TRACE_EVENTS,
    metadata_json: str | Path = DEFAULT_METADATA,
    prefix_csv: str | Path = DEFAULT_PREFIX_CSV,
    manifest_json: str | Path = DEFAULT_MANIFEST_JSON,
    max_rows: int = DEFAULT_MAX_ROWS,
    balanced_families: tuple[str, ...] = ("claudecode", "iflow"),
) -> dict[str, Any]:
    metadata_by_task, metadata_by_task_category, metadata_record_count = _load_metadata(metadata_json)
    prefix_csv = Path(prefix_csv)
    manifest_json = Path(manifest_json)
    prefix_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames()
    counts = Counter()
    family_counts: Counter[str] = Counter()
    family_quota = {family: max_rows // len(balanced_families) for family in balanced_families} if balanced_families else {}
    matched_tasks: set[str] = set()
    unmatched_tasks: set[str] = set()

    with prefix_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        with Path(trace_events_jsonl).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if counts["prefix_rows"] >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    event = TraceEvent.model_validate(json.loads(line))
                except Exception:  # noqa: BLE001 - preview skips malformed rows and records count.
                    counts["malformed_jsonl_line_count"] += 1
                    continue
                row = _preview_row(event, metadata_by_task, metadata_by_task_category)
                row_family = str(row["source_agent_family"] or "unknown")
                if row_family in family_quota and family_counts[row_family] >= family_quota[row_family]:
                    continue
                writer.writerow(row)
                counts["prefix_rows"] += 1
                counts["events_read"] = line_no
                family_counts[row_family] += 1
                if row["metadata_match"]:
                    matched_tasks.add(str(row["task_id"]))
                elif row["task_id"]:
                    unmatched_tasks.add(str(row["task_id"]))

    manifest = {
        "status": "ok",
        "trace_events_jsonl": str(trace_events_jsonl),
        "metadata_json": str(metadata_json),
        "prefix_csv": str(prefix_csv),
        "max_rows": max_rows,
        "balanced_families": list(balanced_families),
        "family_quota": family_quota,
        "prefix_rows": counts["prefix_rows"],
        "events_read": counts["events_read"],
        "malformed_jsonl_line_count": counts["malformed_jsonl_line_count"],
        "source_agent_family_counts": dict(sorted(family_counts.items())),
        "metadata_join_coverage": {
            "matched_tasks_in_preview": len(matched_tasks),
            "unmatched_tasks_in_preview": len(unmatched_tasks),
            "metadata_record_count": metadata_record_count,
        },
        "labels": {
            "label_status": "missing_pending_judge",
            "future_risk_label_available": False,
            "future_severity_available": False,
        },
        "outputs": {
            "prefix_csv": str(prefix_csv),
            "manifest_json": str(manifest_json),
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
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a preview AgentHazard prefix dataset from converted TraceEvent JSONL.")
    parser.add_argument("--trace-events", default=DEFAULT_TRACE_EVENTS)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--prefix-csv", default=DEFAULT_PREFIX_CSV)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args()
    result = build_agenthazard_prefix_dataset_preview(
        trace_events_jsonl=args.trace_events,
        metadata_json=args.metadata,
        prefix_csv=args.prefix_csv,
        manifest_json=args.manifest_json,
        max_rows=args.max_rows,
    )
    print(json.dumps({
        "status": result["status"],
        "prefix_rows": result["prefix_rows"],
        "metadata_join_coverage": result["metadata_join_coverage"],
        "labels": result["labels"],
        "outputs": result["outputs"],
        "rules": result["rules"],
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
