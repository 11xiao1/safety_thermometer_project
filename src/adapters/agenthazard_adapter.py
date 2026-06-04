from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.monitor.schema import TraceEvent


DATASET_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml"}
README_NAMES = {"readme", "readme.md", "readme.txt"}
ENVIRONMENT_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "dockerfile",
    "docker-compose.yml",
}
RUNNER_KEYWORDS = {"run", "runner", "main", "evaluate", "eval", "benchmark", "collect", "judge"}
TRAJECTORY_KEYWORDS = {"trace", "traces", "trajectory", "session", "history"}
LABEL_KEYWORDS = {"label", "annotation", "annotations", "evaluation", "result", "score"}
TEXT_SAMPLE_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".yml", ".log"}
MAX_INTERNAL_SAMPLE_BYTES = 64_000
MAX_INTERNAL_TEXT_SAMPLES = 8
ARCHIVE_TRAJECTORY_KEYWORDS = {
    "action",
    "command",
    "history",
    "message",
    "session",
    "step",
    "trace",
    "trajectory",
}
ARCHIVE_METADATA_KEYWORDS = {"config", "dataset", "env", "metadata", "prompt", "readme", "task"}
ARCHIVE_RESULT_KEYWORDS = {"eval", "evaluation", "judge", "label", "outcome", "result", "score", "success"}

TRACE_EVENT_TARGET_FIELDS = [
    "benchmark_name",
    "benchmark_role",
    "suite",
    "task_id",
    "episode_id",
    "step_id",
    "hook_type",
    "user_instruction",
    "plan_summary",
    "proposed_tool",
    "tool_args",
    "observation",
    "state_delta",
    "self_check",
    "final_outcome",
    "utility",
    "security",
    "source_trace_path",
    "source_batch",
]

SUPPORTED_TRAJECTORY_FAMILIES = {"claudecode", "iflow"}
AGENTHAZARD_STEP_SCALE = 10_000
AGENTHAZARD_PRE_TOOL_OFFSET = -1_000
AGENTHAZARD_POST_TOOL_OFFSET = 1_000

FIELD_SYNONYMS = {
    "task_id": {"id", "task_id", "instance_id", "original_id", "case_id", "sample_id"},
    "step_id": {"step", "step_id", "step_index", "turn", "turn_id", "turn_idx", "sequence", "round", "index"},
    "user_instruction": {"query", "goal", "objective", "instruction", "user_instruction", "prompt", "target"},
    "action": {"action", "operation", "tool", "tool_name", "tool_call", "command", "cmd"},
    "command": {"command", "cmd", "shell_command"},
    "tool_call": {"tool_call", "tool_calls", "function_call", "function_calls", "tool_name", "tool"},
    "tool_args": {"args", "arguments", "tool_args", "tool_input", "parameters", "params", "input"},
    "observation": {"observation", "result", "output", "stdout", "stderr", "response", "environment_response"},
    "state_delta": {"state_delta", "state", "diff", "changes", "filesystem_delta"},
    "model_response": {"assistant", "assistant_message", "model_response", "response", "completion", "content"},
    "final_outcome": {"final_outcome", "outcome", "result", "judgement", "judgment", "verdict"},
    "success_failure": {"success", "failed", "failure", "done", "status"},
    "harmful_safe": {"harmful", "safe", "security", "violation", "risk", "label"},
    "first_harmful_step": {"first_harmful_step", "first_risk_step", "violation_step"},
    "timestamp": {"timestamp", "time", "created_at", "datetime"},
}


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


class AgentHazardAdapter:
    """Adapter skeleton for mapping AgentHazard records to Safety Thermometer TraceEvent records.

    The exact AgentHazard trajectory archive schema is not yet frozen in this project, so the
    conversion methods expose conservative interfaces and avoid executing AgentHazard code.
    """

    def __init__(self, repo_path: str | Path = "external/AgentHazard"):
        self.repo_path = Path(repo_path)

    def inspect_repository(self, max_depth: int = 4) -> dict[str, Any]:
        exists = self.repo_path.exists()
        if not exists:
            return {
                "availability_status": "missing",
                "repo_path": str(self.repo_path),
                "blockers": [f"AgentHazard repository not found: {self.repo_path}"],
            }
        files = self._safe_file_walk(max_depth=max_depth)
        dataset_files = self.list_dataset_files(files=files)
        trajectory_files = self.list_trajectory_files(files=files)
        sample = self.infer_schema_from_sample(dataset_files[:5])
        return {
            "availability_status": "available",
            "repo_path": str(self.repo_path),
            "detected_readme_files": self._detect_readmes(files),
            "detected_dataset_files": dataset_files,
            "detected_trajectory_files": trajectory_files,
            "detected_label_files": self._detect_label_files(files),
            "detected_runner_files": self._detect_runner_files(files),
            "detected_environment_files": self._detect_environment_files(files),
            "sample_schema_preview": sample,
            "task_count_if_known": sample.get("task_count_if_known"),
            "risk_categories_if_known": sample.get("risk_categories_if_known", []),
            "attack_strategies_if_known": sample.get("attack_strategies_if_known", []),
            "adapter_status": "skeleton_ready",
            "trace_schema_status": "mapping_preview_ready",
            "blockers": sample.get("blockers", []),
            "next_required_action": "Open trajectory archives and implement exact AgentHazard trajectory conversion without executing benchmark code.",
        }

    def _safe_file_walk(self, max_depth: int = 4) -> list[Path]:
        files: list[Path] = []
        for path in self.repo_path.rglob("*"):
            if path.is_dir():
                continue
            try:
                relative = path.relative_to(self.repo_path)
            except ValueError:
                continue
            if len(relative.parts) > max_depth:
                continue
            files.append(path)
        return sorted(files)

    def list_dataset_files(self, files: list[Path] | None = None) -> list[str]:
        files = files or self._safe_file_walk()
        return [
            _rel(path, self.repo_path)
            for path in files
            if path.suffix.lower() in DATASET_SUFFIXES
            and any(part.lower() in {"data", "dataset", "datasets", "config"} for part in path.relative_to(self.repo_path).parts)
        ]

    def list_trajectory_files(self, files: list[Path] | None = None) -> list[str]:
        files = files or self._safe_file_walk()
        detected = []
        for path in files:
            rel_text = _rel(path, self.repo_path).lower()
            if any(keyword in rel_text for keyword in TRAJECTORY_KEYWORDS) or path.suffix.lower() == ".zip":
                detected.append(_rel(path, self.repo_path))
        return detected

    def infer_schema_from_sample(self, candidate_files: list[str] | None = None) -> dict[str, Any]:
        candidate_files = candidate_files or self.list_dataset_files()
        blockers = []
        for rel_path in candidate_files:
            path = self.repo_path / rel_path
            if path.name.lower() == "dataset.json":
                return self._schema_from_dataset_json(path)
            if path.suffix.lower() == ".csv":
                preview = self._schema_from_csv(path)
                if preview:
                    return preview
        blockers.append("No small readable dataset sample found among detected dataset files.")
        return {"blockers": blockers}

    def _schema_from_dataset_json(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = payload[0] if isinstance(payload, list) and payload else payload
        keys = sorted(sample.keys()) if isinstance(sample, dict) else []
        categories = sorted({str(row.get("category")) for row in payload if isinstance(row, dict) and row.get("category")}) if isinstance(payload, list) else []
        attacks = sorted({str(row.get("jailbreak_method")) for row in payload if isinstance(row, dict) and row.get("jailbreak_method")}) if isinstance(payload, list) else []
        return {
            "sample_path": _rel(path, self.repo_path),
            "format": "json",
            "top_level_type": type(payload).__name__,
            "sample_keys": keys,
            "sample_record": sample if isinstance(sample, dict) else None,
            "task_count_if_known": len(payload) if isinstance(payload, list) else None,
            "risk_categories_if_known": categories,
            "attack_strategies_if_known": attacks,
            "detected_field_mapping": self.mapping_preview_for_record(sample if isinstance(sample, dict) else {}),
            "blockers": [] if keys else ["dataset sample is not an object record"],
        }

    def _schema_from_csv(self, path: Path) -> dict[str, Any] | None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
            if row is None:
                return None
            return {
                "sample_path": _rel(path, self.repo_path),
                "format": "csv",
                "sample_keys": list(row.keys()),
                "sample_record": row,
                "detected_field_mapping": self.mapping_preview_for_record(row),
                "blockers": [],
            }

    def mapping_preview_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        task_id = record.get("id") or record.get("instance_id") or record.get("original_id")
        instruction = record.get("query") or record.get("goal") or record.get("objective") or record.get("instruction") or record.get("target")
        risk_category = record.get("category") or record.get("risk_category")
        return {
            "benchmark_name": "agenthazard",
            "benchmark_role": "primary",
            "suite": risk_category,
            "task_id": task_id,
            "episode_id": f"agenthazard:{risk_category}:{task_id}" if task_id is not None else None,
            "user_instruction": instruction,
            "attack_strategy": record.get("jailbreak_method") or record.get("attack_strategy"),
            "final_outcome": record.get("final_outcome") or record.get("harmful"),
            "severity_or_harm_score": record.get("severity") or record.get("score") or record.get("harm_score"),
            "first_harmful_step": record.get("first_harmful_step") or record.get("first_risk_step"),
        }

    def convert_instance_to_trace_events(self, record: dict[str, Any], source_trace_path: str | None = None, source_batch: str | None = None) -> list[TraceEvent]:
        mapping = self.mapping_preview_for_record(record)
        if mapping["task_id"] is None or mapping["user_instruction"] is None:
            raise NotImplementedError("AgentHazard instance schema needs explicit task_id and instruction mapping.")
        notes = {
            "benchmark_name": "agenthazard",
            "benchmark_role": "primary",
            "risk_category": mapping["suite"],
            "attack_strategy": mapping["attack_strategy"],
            "final_outcome": mapping["final_outcome"],
            "source_trace_path": source_trace_path,
            "source_batch": source_batch,
            "utility": None,
            "security": None,
        }
        return [
            TraceEvent(
                episode_id=str(mapping["episode_id"]),
                step_id=0,
                hook_type="pre_step",
                user_instruction=str(mapping["user_instruction"]),
                plan_summary=None,
                proposed_tool=None,
                tool_args={},
                observation=None,
                state_delta={},
                self_check={},
                notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
            )
        ]

    def convert_trajectory_to_trace_events(
        self,
        trajectory: Any,
        source_trace_path: str | None = None,
        source_batch: str | None = None,
        agent_family: str | None = None,
    ) -> list[TraceEvent]:
        if not isinstance(trajectory, dict):
            raise NotImplementedError("AgentHazard trajectory conversion expects one JSON object record.")
        if agent_family is not None and agent_family not in SUPPORTED_TRAJECTORY_FAMILIES:
            raise NotImplementedError(f"AgentHazard family is not enabled for main conversion: {agent_family}")

        task_id = trajectory.get("id") or trajectory.get("instance_id") or trajectory.get("original_id")
        instruction = trajectory.get("query") or trajectory.get("goal") or trajectory.get("objective") or trajectory.get("instruction")
        category = trajectory.get("category") or trajectory.get("risk_category")
        if task_id is None or not instruction:
            raise NotImplementedError("AgentHazard trajectory needs task_id/id and query/instruction fields.")

        inferred_family = agent_family or self._family_from_source_batch(source_batch) or "unknown"
        if inferred_family not in SUPPORTED_TRAJECTORY_FAMILIES:
            raise NotImplementedError(f"AgentHazard family is not enabled for main conversion: {inferred_family}")

        episode_id = self._trajectory_episode_id(trajectory, inferred_family, source_batch)
        base_notes = {
            "benchmark_name": "agenthazard",
            "benchmark_role": "primary",
            "agent_family": inferred_family,
            "agent_name": trajectory.get("agent_name"),
            "task_id": task_id,
            "risk_category": category,
            "attack_strategy": trajectory.get("jailbreak_method") or trajectory.get("attack_strategy"),
            "original_id": trajectory.get("original_id"),
            "sandbox_id": trajectory.get("sandbox_id"),
            "source": trajectory.get("source"),
            "source_trace_path": source_trace_path,
            "source_batch": source_batch,
            "utility": None,
            "security": None,
        }

        events: list[TraceEvent] = []
        events.extend(self._turn_events(trajectory, episode_id, str(instruction), base_notes))
        events.extend(self._tool_events(trajectory, episode_id, str(instruction), base_notes))
        final_step = max((event.step_id for event in events), default=0) + AGENTHAZARD_STEP_SCALE
        final_notes = dict(base_notes)
        final_notes["event_kind"] = "trajectory_final"
        final_notes["total_turns"] = trajectory.get("total_turns")
        events.append(
            TraceEvent(
                episode_id=episode_id,
                step_id=final_step,
                hook_type="final",
                user_instruction=str(instruction),
                observation=self._stringify_observation(
                    {
                        "total_turns": trajectory.get("total_turns"),
                        "turn_count": len(trajectory.get("turns") or []),
                        "tool_event_count": len(events) - len(trajectory.get("turns") or []),
                    }
                ),
                notes=json.dumps(final_notes, ensure_ascii=False, sort_keys=True),
            )
        )
        return sorted(events, key=lambda event: (event.step_id, {"pre_step": 0, "checkpoint": 1, "post_step": 2, "final": 3}.get(event.hook_type, 9)))

    def convert_supported_trace_archive(
        self,
        archive_path: str | Path,
        max_trajectories: int | None = None,
    ) -> list[TraceEvent]:
        path = Path(archive_path)
        if not path.is_absolute() and not path.exists():
            path = self.repo_path / path
        family = self._agent_family_for_archive(path)
        if family not in SUPPORTED_TRAJECTORY_FAMILIES:
            raise NotImplementedError(f"AgentHazard family is not enabled for main conversion: {family}")

        events: list[TraceEvent] = []
        with zipfile.ZipFile(path, "r") as archive:
            names = [
                name
                for name in archive.namelist()
                if not self._is_ignorable_internal_path(name)
                and Path(name).name.startswith("trajectory_")
                and Path(name).suffix.lower() == ".jsonl"
            ]
            for name in sorted(names)[:max_trajectories]:
                text = archive.read(name).decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    events.extend(
                        self.convert_trajectory_to_trace_events(
                            record,
                            source_trace_path=name,
                            source_batch=_rel(path, self.repo_path) if self.repo_path in path.parents else str(path),
                            agent_family=family,
                        )
                    )
        return events

    def inspect_trace_archive(
        self,
        archive_path: str | Path,
        max_text_samples: int = MAX_INTERNAL_TEXT_SAMPLES,
        max_sample_bytes: int = MAX_INTERNAL_SAMPLE_BYTES,
    ) -> dict[str, Any]:
        path = Path(archive_path)
        if not path.is_absolute() and not path.exists():
            path = self.repo_path / path
        report: dict[str, Any] = {
            "archive_path": _rel(path, self.repo_path) if path.exists() and self.repo_path in path.parents else str(path),
            "agent_family": self._agent_family_for_archive(path),
            "compressed_size": path.stat().st_size if path.exists() else None,
            "uncompressed_size": None,
            "internal_file_count": 0,
            "top_level_internal_paths": [],
            "candidate_trajectory_files": [],
            "candidate_metadata_files": [],
            "candidate_result_outcome_files": [],
            "sampled_files": [],
            "detected_fields": {},
            "schema_signature": "unreadable",
            "blockers": [],
        }
        if not path.exists():
            report["blockers"].append(f"Archive not found: {path}")
            return report
        try:
            with zipfile.ZipFile(path, "r") as archive:
                infos = [info for info in archive.infolist() if not info.is_dir() and not self._is_ignorable_internal_path(info.filename)]
                report["internal_file_count"] = len(infos)
                report["uncompressed_size"] = sum(info.file_size for info in infos)
                names = [info.filename for info in infos]
                report["top_level_internal_paths"] = self._top_level_internal_paths(names)
                report["candidate_trajectory_files"] = self._candidate_internal_files(names, ARCHIVE_TRAJECTORY_KEYWORDS)
                report["candidate_metadata_files"] = self._candidate_internal_files(names, ARCHIVE_METADATA_KEYWORDS)
                report["candidate_result_outcome_files"] = self._candidate_internal_files(names, ARCHIVE_RESULT_KEYWORDS)
                sample_names = self._sample_internal_names(
                    names,
                    report["candidate_trajectory_files"],
                    report["candidate_result_outcome_files"],
                    report["candidate_metadata_files"],
                    max_text_samples,
                )
                report["sampled_files"] = [
                    self._sample_internal_file(archive, name, max_sample_bytes=max_sample_bytes)
                    for name in sample_names
                ]
        except zipfile.BadZipFile:
            report["blockers"].append("BadZipFile: archive cannot be read as a zip file.")
            return report

        report["detected_fields"] = self.infer_archive_schema(report)
        report["schema_signature"] = self._schema_signature(report)
        if not report["candidate_trajectory_files"]:
            report["blockers"].append("No likely trajectory step files detected by name.")
        if not report["sampled_files"]:
            report["blockers"].append("No small readable text/JSON/JSONL/CSV files sampled.")
        return report

    def infer_archive_schema(self, archive_report: dict[str, Any]) -> dict[str, Any]:
        aggregate_keys: set[str] = set()
        per_file = []
        for sample in archive_report.get("sampled_files", []):
            keys = set(sample.get("sample_keys", []))
            keys.update(sample.get("nested_keys", []))
            aggregate_keys.update(str(key) for key in keys)
            per_file.append(
                {
                    "internal_path": sample.get("internal_path"),
                    "format": sample.get("format"),
                    "record_count_if_known": sample.get("record_count_if_known"),
                    "sample_keys": sorted(keys),
                    "detected_capabilities": self._detected_capabilities(keys),
                }
            )
        return {
            "aggregate_keys": sorted(aggregate_keys),
            "per_file": per_file,
            "detected_capabilities": self._detected_capabilities(aggregate_keys),
            "trace_event_target_coverage": self._trace_event_coverage(aggregate_keys),
        }

    def plan_trace_conversion(self, archive_reports: list[dict[str, Any]]) -> dict[str, Any]:
        families: dict[str, dict[str, Any]] = {}
        for report in archive_reports:
            signature = report.get("schema_signature", "unknown")
            family = families.setdefault(
                signature,
                {
                    "schema_signature": signature,
                    "archive_count": 0,
                    "agent_families": [],
                    "archives": [],
                    "detected_capabilities": {},
                    "blockers": [],
                },
            )
            family["archive_count"] += 1
            if report.get("agent_family") not in family["agent_families"]:
                family["agent_families"].append(report.get("agent_family"))
            family["archives"].append(report.get("archive_path"))
            family["blockers"].extend(report.get("blockers", []))
            for key, value in report.get("detected_fields", {}).get("detected_capabilities", {}).items():
                family["detected_capabilities"][key] = bool(family["detected_capabilities"].get(key) or value)

        usable_families = [
            family
            for family in families.values()
            if family["detected_capabilities"].get("step_id")
            and (
                family["detected_capabilities"].get("action")
                or family["detected_capabilities"].get("tool_call")
                or family["detected_capabilities"].get("command")
                or family["detected_capabilities"].get("model_response")
            )
        ]
        conversion_feasible = bool(usable_families)
        blockers = []
        if not archive_reports:
            blockers.append("No AgentHazard trace archives were found.")
        if not conversion_feasible:
            blockers.append("No sampled schema family exposes both step identity and action/tool/model-response fields.")
        strategy = (
            "Implement one read-only parser per schema family, then emit pre_step/post_step/final TraceEvent records."
            if conversion_feasible
            else "Fall back to dataset.json instance-level traces until archive record schema is confirmed."
        )
        return {
            "status": "ok",
            "schema_family_count": len(families),
            "schema_families": sorted(families.values(), key=lambda row: str(row["schema_signature"])),
            "conversion_feasible": conversion_feasible,
            "recommended_strategy": strategy,
            "target_trace_event_fields": TRACE_EVENT_TARGET_FIELDS,
            "blockers": blockers,
            "rules": {
                "will_call_provider": False,
                "will_run_agenthazard": False,
                "will_run_agentdojo": False,
                "will_train": False,
                "will_calibrate": False,
                "test_split_used": False,
            },
        }

    def build_manifest(self) -> dict[str, Any]:
        return self.inspect_repository()

    def _detect_readmes(self, files: list[Path]) -> list[str]:
        return [_rel(path, self.repo_path) for path in files if path.name.lower() in README_NAMES]

    def _detect_label_files(self, files: list[Path]) -> list[str]:
        return [_rel(path, self.repo_path) for path in files if any(keyword in _rel(path, self.repo_path).lower() for keyword in LABEL_KEYWORDS)]

    def _detect_runner_files(self, files: list[Path]) -> list[str]:
        detected = []
        for path in files:
            rel_text = _rel(path, self.repo_path).lower()
            if path.suffix.lower() in {".py", ".sh", ".ipynb", ".js"} and any(keyword in rel_text for keyword in RUNNER_KEYWORDS):
                detected.append(_rel(path, self.repo_path))
        return detected

    def _detect_environment_files(self, files: list[Path]) -> list[str]:
        return [
            _rel(path, self.repo_path)
            for path in files
            if path.name.lower() in ENVIRONMENT_NAMES
            or path.name.lower().startswith("dockerfile")
            or "docker" in _rel(path, self.repo_path).lower()
        ]

    def _agent_family_for_archive(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.repo_path)
        except ValueError:
            return "unknown"
        if len(relative.parts) >= 2 and relative.parts[0].lower() == "traces":
            return relative.parts[1].lower()
        return relative.parts[0].lower() if relative.parts else "unknown"

    def _family_from_source_batch(self, source_batch: str | None) -> str | None:
        if not source_batch:
            return None
        parts = Path(source_batch.replace("\\", "/")).parts
        lowered = [part.lower() for part in parts]
        for family in SUPPORTED_TRAJECTORY_FAMILIES:
            if family in lowered:
                return family
        return None

    def _trajectory_episode_id(self, trajectory: dict[str, Any], family: str, source_batch: str | None) -> str:
        category = trajectory.get("category") or trajectory.get("risk_category") or "unknown"
        task_id = trajectory.get("id") or trajectory.get("instance_id") or trajectory.get("original_id")
        source_archive = Path((source_batch or family).replace("\\", "/")).stem
        return f"agenthazard:{family}:{source_archive}:{category}:{task_id}"

    def _turn_events(
        self,
        trajectory: dict[str, Any],
        episode_id: str,
        instruction: str,
        base_notes: dict[str, Any],
    ) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        turns = trajectory.get("turns") or []
        if not isinstance(turns, list):
            return events
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            turn_idx = self._coerce_step_id(turn.get("turn_idx") or turn.get("step_id") or turn.get("step"), index)
            step_id = turn_idx * AGENTHAZARD_STEP_SCALE
            notes = dict(base_notes)
            notes.update(
                {
                    "event_kind": "agent_turn",
                    "turn_idx": turn.get("turn_idx"),
                    "start_time": turn.get("start_time"),
                    "end_time": turn.get("end_time"),
                }
            )
            events.append(
                TraceEvent(
                    episode_id=episode_id,
                    step_id=step_id,
                    hook_type="checkpoint",
                    user_instruction=instruction,
                    plan_summary=self._stringify_optional(turn.get("input")),
                    observation=self._stringify_optional(turn.get("output")),
                    notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
                )
            )
        return events

    def _tool_events(
        self,
        trajectory: dict[str, Any],
        episode_id: str,
        instruction: str,
        base_notes: dict[str, Any],
    ) -> list[TraceEvent]:
        detail_logs = trajectory.get("detail_logs")
        if isinstance(detail_logs, list):
            return self._claudecode_tool_events(detail_logs, episode_id, instruction, base_notes)
        if isinstance(detail_logs, dict):
            return self._iflow_tool_events(detail_logs, episode_id, instruction, base_notes)
        return []

    def _claudecode_tool_events(
        self,
        detail_logs: list[Any],
        episode_id: str,
        instruction: str,
        base_notes: dict[str, Any],
    ) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        per_step_hook_counts: Counter[tuple[int, str]] = Counter()
        for index, log in enumerate(detail_logs, start=1):
            if not isinstance(log, dict):
                continue
            log_type = str(log.get("type") or "")
            if log_type not in {"pre_tool", "post_tool"}:
                continue
            base_step = self._step_id_from_sequence(log.get("sequence"), index)
            per_step_hook_counts[(base_step, log_type)] += 1
            offset = AGENTHAZARD_PRE_TOOL_OFFSET if log_type == "pre_tool" else AGENTHAZARD_POST_TOOL_OFFSET
            step_id = base_step * AGENTHAZARD_STEP_SCALE + offset + per_step_hook_counts[(base_step, log_type)]
            notes = dict(base_notes)
            notes.update(
                {
                    "event_kind": log_type,
                    "tool_use_id": log.get("tool_use_id"),
                    "session_id": log.get("session_id"),
                    "cwd": log.get("cwd"),
                    "permission_mode": log.get("permission_mode"),
                    "timestamp": log.get("timestamp"),
                }
            )
            events.append(
                TraceEvent(
                    episode_id=episode_id,
                    step_id=step_id,
                    hook_type="pre_step" if log_type == "pre_tool" else "post_step",
                    user_instruction=instruction,
                    proposed_tool=self._stringify_optional(log.get("tool_name")),
                    tool_args=log.get("tool_input") if isinstance(log.get("tool_input"), dict) else {},
                    observation=self._stringify_observation(log.get("tool_response")) if log_type == "post_tool" else None,
                    notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
                )
            )
        return events

    def _iflow_tool_events(
        self,
        detail_logs: dict[str, Any],
        episode_id: str,
        instruction: str,
        base_notes: dict[str, Any],
    ) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        tool_calls = detail_logs.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            return events
        per_step_counts: Counter[int] = Counter()
        for index, call in enumerate(tool_calls, start=1):
            if not isinstance(call, dict):
                continue
            base_step = self._coerce_step_id(call.get("turn_idx") or call.get("step") or call.get("step_id"), index)
            per_step_counts[base_step] += 1
            step_id = base_step * AGENTHAZARD_STEP_SCALE + AGENTHAZARD_PRE_TOOL_OFFSET + per_step_counts[base_step]
            tool_name = call.get("tool_name") or call.get("name") or call.get("tool")
            tool_args = call.get("tool_input") or call.get("arguments") or call.get("args") or call.get("input")
            observation = call.get("tool_response") or call.get("response") or call.get("result") or call.get("output")
            notes = dict(base_notes)
            notes.update({"event_kind": "iflow_tool_call", "timestamp": call.get("timestamp")})
            events.append(
                TraceEvent(
                    episode_id=episode_id,
                    step_id=step_id,
                    hook_type="pre_step",
                    user_instruction=instruction,
                    proposed_tool=self._stringify_optional(tool_name),
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                    observation=self._stringify_observation(observation),
                    notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
                )
            )
        return events

    def _coerce_step_id(self, value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _step_id_from_sequence(self, value: Any, fallback: int) -> int:
        text = str(value or "")
        digits = "".join(character for character in text if character.isdigit())
        return self._coerce_step_id(digits, fallback)

    def _stringify_optional(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _stringify_observation(self, value: Any) -> str | None:
        return self._stringify_optional(value)

    def _top_level_internal_paths(self, names: list[str]) -> list[str]:
        top_level = []
        for name in names:
            clean = name.strip("/")
            if not clean:
                continue
            parts = clean.split("/")
            value = parts[0] if len(parts) == 1 else f"{parts[0]}/"
            if value not in top_level:
                top_level.append(value)
        return sorted(top_level)[:50]

    def _candidate_internal_files(self, names: list[str], keywords: set[str]) -> list[str]:
        candidates = []
        for name in names:
            if self._is_ignorable_internal_path(name):
                continue
            lower = name.lower()
            suffix = Path(name).suffix.lower()
            if suffix not in TEXT_SAMPLE_SUFFIXES and suffix not in {".jsonl", ".csv"}:
                continue
            if any(keyword in lower for keyword in keywords):
                candidates.append(name)
        return sorted(candidates)[:100]

    def _sample_internal_names(
        self,
        names: list[str],
        trajectory_files: list[str],
        result_files: list[str],
        metadata_files: list[str],
        max_text_samples: int,
    ) -> list[str]:
        ordered = []
        for group in [trajectory_files, result_files, metadata_files, sorted(names)]:
            for name in group:
                if self._is_ignorable_internal_path(name):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix in TEXT_SAMPLE_SUFFIXES and name not in ordered:
                    ordered.append(name)
                if len(ordered) >= max_text_samples:
                    return ordered
        return ordered

    def _sample_internal_file(self, archive: zipfile.ZipFile, name: str, max_sample_bytes: int) -> dict[str, Any]:
        info = archive.getinfo(name)
        sample: dict[str, Any] = {
            "internal_path": name,
            "compressed_size": info.compress_size,
            "uncompressed_size": info.file_size,
            "format": Path(name).suffix.lower().lstrip(".") or "unknown",
            "sample_keys": [],
            "nested_keys": [],
            "record_count_if_known": None,
            "parse_error": None,
        }
        if info.file_size > max_sample_bytes:
            sample["parse_error"] = f"Skipped large internal file over {max_sample_bytes} bytes."
            return sample
        raw = archive.read(name)
        text = raw.decode("utf-8", errors="replace")
        suffix = Path(name).suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(text)
                self._populate_json_sample(sample, payload)
            elif suffix == ".jsonl":
                records = [json.loads(line) for line in text.splitlines() if line.strip()][:20]
                sample["record_count_if_known"] = len(records)
                self._populate_json_sample(sample, records)
            elif suffix == ".csv":
                reader = csv.DictReader(text.splitlines())
                row = next(reader, None)
                sample["sample_keys"] = list(row.keys()) if row else []
            else:
                sample["sample_keys"] = self._keys_from_loose_text(text)
        except (csv.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
            sample["parse_error"] = f"{type(exc).__name__}: {exc}"
        return sample

    def _populate_json_sample(self, sample: dict[str, Any], payload: Any) -> None:
        record = payload[0] if isinstance(payload, list) and payload else payload
        sample["record_count_if_known"] = len(payload) if isinstance(payload, list) else None
        if isinstance(record, dict):
            sample["sample_keys"] = sorted(str(key) for key in record.keys())
            sample["nested_keys"] = sorted(self._collect_nested_keys(record))

    def _collect_nested_keys(self, value: Any, prefix: str = "", max_depth: int = 3) -> set[str]:
        if max_depth <= 0:
            return set()
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                text_key = str(key)
                compound = f"{prefix}.{text_key}" if prefix else text_key
                keys.add(text_key)
                keys.add(compound)
                keys.update(self._collect_nested_keys(child, compound, max_depth=max_depth - 1))
        elif isinstance(value, list) and value:
            keys.update(self._collect_nested_keys(value[0], prefix, max_depth=max_depth - 1))
        return keys

    def _keys_from_loose_text(self, text: str) -> list[str]:
        keys = []
        lowered = text.lower()
        for field, synonyms in FIELD_SYNONYMS.items():
            if any(synonym in lowered for synonym in synonyms):
                keys.append(field)
        return sorted(keys)

    def _detected_capabilities(self, keys: set[str]) -> dict[str, bool]:
        lowered = {str(key).lower() for key in keys}
        return {
            capability: bool(lowered & synonyms)
            for capability, synonyms in FIELD_SYNONYMS.items()
        }

    def _trace_event_coverage(self, keys: set[str]) -> dict[str, bool]:
        capabilities = self._detected_capabilities(keys)
        return {
            "benchmark_name": True,
            "benchmark_role": True,
            "suite": bool({"suite", "category", "risk_category"} & {str(key).lower() for key in keys}),
            "task_id": capabilities["task_id"],
            "episode_id": capabilities["task_id"],
            "step_id": capabilities["step_id"],
            "hook_type": capabilities["action"] or capabilities["tool_call"] or capabilities["model_response"],
            "user_instruction": capabilities["user_instruction"],
            "plan_summary": bool({"plan", "plan_summary", "thought"} & {str(key).lower() for key in keys}),
            "proposed_tool": capabilities["action"] or capabilities["tool_call"] or capabilities["command"],
            "tool_args": capabilities["tool_args"],
            "observation": capabilities["observation"],
            "state_delta": capabilities["state_delta"],
            "self_check": bool({"self_check", "reflection", "safety_check"} & {str(key).lower() for key in keys}),
            "final_outcome": capabilities["final_outcome"],
            "utility": bool({"utility", "task_success"} & {str(key).lower() for key in keys}),
            "security": capabilities["harmful_safe"],
            "source_trace_path": True,
            "source_batch": True,
        }

    def _schema_signature(self, report: dict[str, Any]) -> str:
        sampled_formats = sorted({sample.get("format", "unknown") for sample in report.get("sampled_files", [])})
        capabilities = report.get("detected_fields", {}).get("detected_capabilities", {})
        enabled = sorted(key for key, value in capabilities.items() if value)
        family = report.get("agent_family", "unknown")
        return f"{family}|formats={','.join(sampled_formats) or 'none'}|fields={','.join(enabled) or 'none'}"

    def _is_ignorable_internal_path(self, name: str) -> bool:
        parts = [part for part in name.replace("\\", "/").split("/") if part]
        return any(part == "__MACOSX" for part in parts) or any(part.startswith("._") for part in parts)
