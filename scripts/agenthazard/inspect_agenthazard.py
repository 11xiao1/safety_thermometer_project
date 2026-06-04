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
DEFAULT_REPORT_JSON = "outputs/agenthazard/agenthazard_availability_report.json"
DEFAULT_REPORT_CSV = "outputs/agenthazard/agenthazard_availability_report.csv"
DEFAULT_MAPPING_PREVIEW = "outputs/agenthazard/agenthazard_trace_mapping_preview.json"
DEFAULT_PROTOCOL_JSON = "outputs/final_benchmark_protocol_manifest.json"
DEFAULT_PROTOCOL_CSV = "outputs/final_benchmark_protocol_manifest.csv"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kind", "path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _availability_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key in [
        "detected_readme_files",
        "detected_dataset_files",
        "detected_trajectory_files",
        "detected_label_files",
        "detected_runner_files",
        "detected_environment_files",
    ]:
        for path in report.get(key, []):
            rows.append({"kind": key, "path": path})
    if not rows:
        rows.append({"kind": "blocker", "path": "; ".join(report.get("blockers", []))})
    return rows


def _agentdojo_status() -> dict[str, Any]:
    full_prefix = Path("outputs/agentdojo_full_combined/agentdojo_full_prefix_dataset.csv")
    manifest = Path("outputs/agentdojo_full_combined/splits/split_manifest.json")
    return {
        "benchmark_name": "agentdojo",
        "benchmark_role": "secondary",
        "availability_status": "available" if full_prefix.exists() and manifest.exists() else "missing_outputs",
        "full_prefix_dataset": str(full_prefix),
        "split_manifest": str(manifest),
    }


def _protocol_manifest(agenthazard_report: dict[str, Any]) -> dict[str, Any]:
    trace_schema_status = "ready_for_mapping_preview" if agenthazard_report.get("availability_status") == "available" else "blocked"
    judge_schema = Path("outputs/labels/judge_label_schema_preview.json")
    return {
        "status": "ok",
        "benchmarks": [
            {
                "benchmark_name": "agenthazard",
                "benchmark_role": "primary",
                "availability_status": agenthazard_report.get("availability_status"),
                "repo_path": agenthazard_report.get("repo_path"),
                "adapter_status": agenthazard_report.get("adapter_status"),
                "blockers": agenthazard_report.get("blockers", []),
            },
            _agentdojo_status(),
        ],
        "unified_trace_event_schema_status": trace_schema_status,
        "judge_label_protocol_status": "available" if judge_schema.exists() else "missing_schema_preview",
        "split_protocol_status": "episode_level_required; test split must not be used before method freeze",
        "rules": {
            "will_call_provider": False,
            "will_run_agenthazard": False,
            "will_run_agentdojo": False,
            "will_train": False,
            "will_calibrate": False,
            "test_split_used": False,
        },
    }


def _write_protocol_csv(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark_name", "benchmark_role", "availability_status", "adapter_status"])
        writer.writeheader()
        for row in manifest["benchmarks"]:
            writer.writerow(
                {
                    "benchmark_name": row.get("benchmark_name"),
                    "benchmark_role": row.get("benchmark_role"),
                    "availability_status": row.get("availability_status"),
                    "adapter_status": row.get("adapter_status", ""),
                }
            )


def inspect_agenthazard(
    repo_path: str | Path = DEFAULT_REPO,
    report_json: str | Path = DEFAULT_REPORT_JSON,
    report_csv: str | Path = DEFAULT_REPORT_CSV,
    mapping_preview_json: str | Path = DEFAULT_MAPPING_PREVIEW,
    protocol_json: str | Path = DEFAULT_PROTOCOL_JSON,
    protocol_csv: str | Path = DEFAULT_PROTOCOL_CSV,
    max_depth: int = 4,
) -> dict[str, Any]:
    adapter = AgentHazardAdapter(repo_path)
    report = adapter.inspect_repository(max_depth=max_depth)
    mapping_preview = {
        "status": "ok" if report.get("sample_schema_preview") else "blocked",
        "mapping_assumptions": {
            "benchmark_name": "agenthazard",
            "benchmark_role": "primary",
            "suite": "risk_category if available",
            "task_id": "id, instance_id, or original_id",
            "user_instruction": "query, goal, objective, instruction, or target",
            "proposed_tool_tool_args": "trajectory action or environment operation when archives are parsed",
            "observation_state_delta": "environment feedback when archives are parsed",
            "final_outcome": "harmful/final outcome if available",
            "utility_security": "unknown unless AgentHazard exposes direct utility/security fields",
        },
        "sample_detected_mapping": report.get("sample_schema_preview", {}).get("detected_field_mapping"),
        "trace_event_conversion_status": "instance_start_event_supported; full trajectory conversion pending schema inspection",
    }
    protocol = _protocol_manifest(report)

    report_json = Path(report_json)
    report_csv = Path(report_csv)
    mapping_preview_json = Path(mapping_preview_json)
    protocol_json = Path(protocol_json)
    protocol_csv = Path(protocol_csv)
    for path, payload in [
        (report_json, report),
        (mapping_preview_json, mapping_preview),
        (protocol_json, protocol),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_csv(report_csv, _availability_rows(report))
    _write_protocol_csv(protocol_csv, protocol)

    return {
        "status": "ok",
        "outputs": {
            "availability_report_json": str(report_json),
            "availability_report_csv": str(report_csv),
            "mapping_preview_json": str(mapping_preview_json),
            "protocol_manifest_json": str(protocol_json),
            "protocol_manifest_csv": str(protocol_csv),
        },
        "agenthazard_availability_status": report.get("availability_status"),
        "detected_dataset_file_count": len(report.get("detected_dataset_files", [])),
        "detected_trajectory_file_count": len(report.get("detected_trajectory_files", [])),
        "detected_runner_file_count": len(report.get("detected_runner_files", [])),
        "blockers": report.get("blockers", []),
        "rules": protocol["rules"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local AgentHazard repository without executing it.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--mapping-preview-json", default=DEFAULT_MAPPING_PREVIEW)
    parser.add_argument("--protocol-json", default=DEFAULT_PROTOCOL_JSON)
    parser.add_argument("--protocol-csv", default=DEFAULT_PROTOCOL_CSV)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    result = inspect_agenthazard(
        repo_path=args.repo,
        report_json=args.report_json,
        report_csv=args.report_csv,
        mapping_preview_json=args.mapping_preview_json,
        protocol_json=args.protocol_json,
        protocol_csv=args.protocol_csv,
        max_depth=args.max_depth,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
