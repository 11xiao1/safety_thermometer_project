from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.agenthazard_adapter import AgentHazardAdapter, SUPPORTED_TRAJECTORY_FAMILIES  # noqa: E402


DEFAULT_REPO = "external/AgentHazard"
DEFAULT_EVENTS_JSONL = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_events.jsonl"
DEFAULT_SUMMARY_JSON = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_conversion_summary.json"
DEFAULT_SUMMARY_CSV = "outputs/agenthazard/agenthazard_claudecode_iflow_trace_conversion_summary.csv"


def _event_to_dict(event) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event.dict()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "archive_path",
        "agent_family",
        "trajectory_file_count",
        "event_count",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def convert_agenthazard_traces(
    repo_path: str | Path = DEFAULT_REPO,
    events_jsonl: str | Path = DEFAULT_EVENTS_JSONL,
    summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    summary_csv: str | Path = DEFAULT_SUMMARY_CSV,
    max_archives: int | None = None,
    max_trajectories_per_archive: int | None = None,
) -> dict[str, Any]:
    adapter = AgentHazardAdapter(repo_path)
    repo = Path(repo_path)
    archive_paths = [
        path
        for family in sorted(SUPPORTED_TRAJECTORY_FAMILIES)
        for path in sorted((repo / "traces" / family).glob("*.zip"))
    ]
    if max_archives is not None:
        archive_paths = archive_paths[:max_archives]

    events_jsonl = Path(events_jsonl)
    summary_json = Path(summary_json)
    summary_csv = Path(summary_csv)
    events_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_events = 0
    total_trajectory_files = 0
    with events_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for archive_path in archive_paths:
            family = archive_path.parent.name
            row = {
                "archive_path": str(archive_path.relative_to(repo)).replace("\\", "/"),
                "agent_family": family,
                "trajectory_file_count": 0,
                "event_count": 0,
                "status": "ok",
                "error": "",
            }
            try:
                events = adapter.convert_supported_trace_archive(
                    archive_path,
                    max_trajectories=max_trajectories_per_archive,
                )
                row["event_count"] = len(events)
                row["trajectory_file_count"] = len({json.loads(event.notes or "{}").get("source_trace_path") for event in events if event.notes})
                for event in events:
                    handle.write(json.dumps(_event_to_dict(event), ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:  # noqa: BLE001 - report archive-level conversion failures without stopping the batch.
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            total_events += int(row["event_count"] or 0)
            total_trajectory_files += int(row["trajectory_file_count"] or 0)
            rows.append(row)

    summary = {
        "status": "ok",
        "repo_path": str(repo),
        "included_families": sorted(SUPPORTED_TRAJECTORY_FAMILIES),
        "excluded_families": ["openclaw"],
        "archive_count": len(rows),
        "converted_archive_count": sum(1 for row in rows if row["status"] == "ok"),
        "trajectory_file_count": total_trajectory_files,
        "event_count": total_events,
        "rows": rows,
        "outputs": {
            "events_jsonl": str(events_jsonl),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
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
    _write_json(summary_json, summary)
    _write_summary_csv(summary_csv, rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AgentHazard claudecode/iflow trajectory_*.jsonl traces to TraceEvent JSONL.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--events-jsonl", default=DEFAULT_EVENTS_JSONL)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--max-trajectories-per-archive", type=int, default=None)
    args = parser.parse_args()
    result = convert_agenthazard_traces(
        repo_path=args.repo,
        events_jsonl=args.events_jsonl,
        summary_json=args.summary_json,
        summary_csv=args.summary_csv,
        max_archives=args.max_archives,
        max_trajectories_per_archive=args.max_trajectories_per_archive,
    )
    print(json.dumps({key: result[key] for key in ["status", "archive_count", "trajectory_file_count", "event_count", "outputs", "rules"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
