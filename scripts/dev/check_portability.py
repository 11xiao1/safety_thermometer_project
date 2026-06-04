from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = "outputs/dev/portability_check_report.json"
DEFAULT_CSV_OUT = "outputs/dev/portability_check_report.csv"
SCAN_DIRS = ["scripts", "src", "tests", "docs"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
LARGE_OUTPUT_SUFFIXES = {".jsonl", ".csv", ".parquet", ".pkl", ".pt", ".bin"}
WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
WINDOWS_BACKSLASH_PATH_RE = re.compile(r"(?<!\\)(?:[\w.-]+\\){1,}[\w.-]+")
KEY_SCRIPTS = [
    "scripts/agenthazard/convert_agenthazard_traces.py",
    "scripts/agenthazard/audit_agenthazard_trace_events.py",
    "scripts/agenthazard/build_agenthazard_prefix_dataset.py",
    "scripts/agenthazard/inspect_agenthazard_trace_archives.py",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _scan_text_files(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for base in SCAN_DIRS:
        scan_root = root / base
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in {".pytest_tmp", ".git", "__pycache__"} for part in path.parts):
                continue
            text = _read_text(path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if WINDOWS_ABSOLUTE_RE.search(line):
                    findings.append(
                        {
                            "severity": "warn",
                            "check": "windows_absolute_path",
                            "path": path.as_posix(),
                            "line": line_no,
                            "detail": line.strip()[:240],
                        }
                    )
                if path.suffix.lower() == ".py" and WINDOWS_BACKSLASH_PATH_RE.search(line):
                    stripped = line.strip()
                    if "\\n" in stripped or "\\t" in stripped:
                        continue
                    findings.append(
                        {
                            "severity": "info",
                            "check": "possible_backslash_path_literal",
                            "path": path.as_posix(),
                            "line": line_no,
                            "detail": stripped[:240],
                        }
                    )
    return findings


def _gitignore_checks(root: Path) -> list[dict[str, Any]]:
    path = root / ".gitignore"
    text = _read_text(path) if path.exists() else ""
    checks = []
    required = {
        "outputs/": "ignore outputs directory and generated large files",
        "external/AgentHazard/": "ignore external AgentHazard checkout",
    }
    for pattern, detail in required.items():
        checks.append(
            {
                "severity": "pass" if pattern in text.splitlines() else "fail",
                "check": "gitignore_pattern",
                "path": ".gitignore",
                "line": "",
                "detail": f"{pattern}: {detail}",
            }
        )
    return checks


def _tracked_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _tracked_large_output_checks(root: Path) -> list[dict[str, Any]]:
    findings = []
    for rel_path in _tracked_files(root):
        path = Path(rel_path)
        if len(path.parts) >= 2 and path.parts[0] == "outputs" and path.suffix.lower() in LARGE_OUTPUT_SUFFIXES:
            findings.append(
                {
                    "severity": "fail",
                    "check": "tracked_large_output",
                    "path": rel_path,
                    "line": "",
                    "detail": "Generated large output file is tracked by git.",
                }
            )
    if not findings:
        findings.append(
            {
                "severity": "pass",
                "check": "tracked_large_output",
                "path": "outputs/",
                "line": "",
                "detail": "No tracked generated large output files found by git ls-files.",
            }
        )
    return findings


def _key_script_checks(root: Path) -> list[dict[str, Any]]:
    checks = []
    for rel_path in KEY_SCRIPTS:
        path = root / rel_path
        if not path.exists():
            checks.append(
                {
                    "severity": "fail",
                    "check": "key_script_exists",
                    "path": rel_path,
                    "line": "",
                    "detail": "Key script is missing.",
                }
            )
            continue
        text = _read_text(path)
        checks.append(
            {
                "severity": "pass" if "argparse" in text else "warn",
                "check": "key_script_cli",
                "path": rel_path,
                "line": "",
                "detail": "Script exposes argparse CLI and accepts relative/default paths." if "argparse" in text else "No argparse CLI detected.",
            }
        )
        checks.append(
            {
                "severity": "pass" if "Path(" in text or "from pathlib import Path" in text else "warn",
                "check": "key_script_pathlib",
                "path": rel_path,
                "line": "",
                "detail": "Script uses pathlib.Path." if "Path(" in text or "from pathlib import Path" in text else "No pathlib usage detected.",
            }
        )
    return checks


def _requirements_checks(root: Path) -> list[dict[str, Any]]:
    candidates = ["requirements-freeze.txt", "environment.yml", "requirements.txt"]
    existing = [name for name in candidates if (root / name).exists()]
    return [
        {
            "severity": "pass" if existing else "fail",
            "check": "dependency_manifest",
            "path": ", ".join(existing) if existing else "",
            "line": "",
            "detail": "Dependency manifest is available." if existing else "No dependency manifest found.",
        }
    ]


def build_portability_report(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    findings = []
    findings.extend(_gitignore_checks(root))
    findings.extend(_tracked_large_output_checks(root))
    findings.extend(_key_script_checks(root))
    findings.extend(_requirements_checks(root))
    findings.extend(_scan_text_files(root))
    fail_count = sum(1 for finding in findings if finding["severity"] == "fail")
    warn_count = sum(1 for finding in findings if finding["severity"] == "warn")
    return {
        "status": "ready_with_warnings" if fail_count == 0 else "blocked",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "finding_count": len(findings),
        "findings": findings,
        "rules": {
            "will_call_provider": False,
            "will_run_agenthazard": False,
            "will_run_agentdojo": False,
            "will_train": False,
            "will_calibrate": False,
            "test_split_used": False,
        },
    }


def write_report(report: dict[str, Any], json_out: str | Path = DEFAULT_JSON_OUT, csv_out: str | Path = DEFAULT_CSV_OUT) -> None:
    json_path = Path(json_out)
    csv_path = Path(csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "check", "path", "line", "detail"])
        writer.writeheader()
        writer.writerows(report["findings"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Safety Thermometer portability for Linux/AutoDL.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    args = parser.parse_args()
    report = build_portability_report(args.root)
    write_report(report, args.json_out, args.csv_out)
    print(json.dumps({key: report[key] for key in ["status", "fail_count", "warn_count", "finding_count", "rules"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
