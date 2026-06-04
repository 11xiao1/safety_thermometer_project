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

from src.adapters.agenthazard_adapter import AgentHazardAdapter  # noqa: E402


DEFAULT_REPO = "external/AgentHazard"
DEFAULT_SCHEMA_JSON = "outputs/agenthazard/agenthazard_trace_archive_schema_report.json"
DEFAULT_SCHEMA_CSV = "outputs/agenthazard/agenthazard_trace_archive_schema_report.csv"
DEFAULT_PLAN_JSON = "outputs/agenthazard/agenthazard_trace_conversion_plan.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_schema_csv(path: Path, archives: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "archive_path",
        "agent_family",
        "compressed_size",
        "uncompressed_size",
        "internal_file_count",
        "schema_signature",
        "candidate_trajectory_file_count",
        "candidate_metadata_file_count",
        "candidate_result_outcome_file_count",
        "sampled_file_count",
        "has_task_id",
        "has_step_id",
        "has_action",
        "has_command",
        "has_tool_call",
        "has_observation",
        "has_final_outcome",
        "has_harmful_safe",
        "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for archive in archives:
            capabilities = archive.get("detected_fields", {}).get("detected_capabilities", {})
            writer.writerow(
                {
                    "archive_path": archive.get("archive_path"),
                    "agent_family": archive.get("agent_family"),
                    "compressed_size": archive.get("compressed_size"),
                    "uncompressed_size": archive.get("uncompressed_size"),
                    "internal_file_count": archive.get("internal_file_count"),
                    "schema_signature": archive.get("schema_signature"),
                    "candidate_trajectory_file_count": len(archive.get("candidate_trajectory_files", [])),
                    "candidate_metadata_file_count": len(archive.get("candidate_metadata_files", [])),
                    "candidate_result_outcome_file_count": len(archive.get("candidate_result_outcome_files", [])),
                    "sampled_file_count": len(archive.get("sampled_files", [])),
                    "has_task_id": capabilities.get("task_id", False),
                    "has_step_id": capabilities.get("step_id", False),
                    "has_action": capabilities.get("action", False),
                    "has_command": capabilities.get("command", False),
                    "has_tool_call": capabilities.get("tool_call", False),
                    "has_observation": capabilities.get("observation", False),
                    "has_final_outcome": capabilities.get("final_outcome", False),
                    "has_harmful_safe": capabilities.get("harmful_safe", False),
                    "blockers": "; ".join(archive.get("blockers", [])),
                }
            )


def inspect_agenthazard_trace_archives(
    repo_path: str | Path = DEFAULT_REPO,
    schema_json: str | Path = DEFAULT_SCHEMA_JSON,
    schema_csv: str | Path = DEFAULT_SCHEMA_CSV,
    plan_json: str | Path = DEFAULT_PLAN_JSON,
    max_archives: int | None = None,
    max_text_samples: int = 8,
) -> dict[str, Any]:
    adapter = AgentHazardAdapter(repo_path)
    repo = Path(repo_path)
    archive_paths = sorted(repo.glob("traces/**/*.zip"))
    if max_archives is not None:
        archive_paths = archive_paths[:max_archives]

    archive_reports = [
        adapter.inspect_trace_archive(path, max_text_samples=max_text_samples)
        for path in archive_paths
    ]
    plan = adapter.plan_trace_conversion(archive_reports)
    schema_report = {
        "status": "ok",
        "repo_path": str(repo),
        "archive_count": len(archive_reports),
        "archives": archive_reports,
        "schema_family_count": plan["schema_family_count"],
        "schema_families": plan["schema_families"],
        "conversion_feasible": plan["conversion_feasible"],
        "blockers": plan["blockers"],
        "rules": plan["rules"],
    }

    schema_json = Path(schema_json)
    schema_csv = Path(schema_csv)
    plan_json = Path(plan_json)
    _write_json(schema_json, schema_report)
    _write_json(plan_json, plan)
    _write_schema_csv(schema_csv, archive_reports)

    return {
        "status": "ok",
        "archive_count": len(archive_reports),
        "schema_family_count": plan["schema_family_count"],
        "conversion_feasible": plan["conversion_feasible"],
        "blockers": plan["blockers"],
        "outputs": {
            "schema_report_json": str(schema_json),
            "schema_report_csv": str(schema_csv),
            "conversion_plan_json": str(plan_json),
        },
        "rules": plan["rules"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AgentHazard trace zip schemas without extraction or execution.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--schema-json", default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--schema-csv", default=DEFAULT_SCHEMA_CSV)
    parser.add_argument("--plan-json", default=DEFAULT_PLAN_JSON)
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--max-text-samples", type=int, default=8)
    args = parser.parse_args()
    result = inspect_agenthazard_trace_archives(
        repo_path=args.repo,
        schema_json=args.schema_json,
        schema_csv=args.schema_csv,
        plan_json=args.plan_json,
        max_archives=args.max_archives,
        max_text_samples=args.max_text_samples,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
