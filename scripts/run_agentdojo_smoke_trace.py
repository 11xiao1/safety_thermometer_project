from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.adapters.agentdojo_pipeline_builder import inspect_agentdojo_pipeline


def build_dry_run_report(args: argparse.Namespace) -> dict:
    plan = inspect_agentdojo_pipeline(package_name=args.agentdojo_package)
    return {
        "mode": "dry_run",
        "suite": args.suite,
        "user_task": args.user_task,
        "trace": args.trace,
        "provider": args.provider,
        "model": args.model,
        "agentdojo": plan.to_dict(),
        "will_call_provider": False,
        "will_run_benchmark": False,
        "will_modify_agentdojo_source": False,
        "will_monkey_patch": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a minimal AgentDojo Safety Thermometer smoke trace run.")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--user-task", default="user_task_0")
    parser.add_argument("--trace", default="outputs/agentdojo_smoke_trace.jsonl")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--agentdojo-package",
        default="agentdojo",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.dry_run:
        report = build_dry_run_report(args)
        print(json.dumps(report, indent=2))
        return

    plan = inspect_agentdojo_pipeline(package_name=args.agentdojo_package)
    message = {
        "mode": "run",
        "status": "not_implemented",
        "reason": (
            "Real AgentDojo smoke execution is not wired yet. "
            "This script will not silently run native AgentDojo without TraceHookedToolsExecutor."
        ),
        "suite": args.suite,
        "user_task": args.user_task,
        "trace": args.trace,
        "agentdojo": plan.to_dict(),
        "next_step": "Implement custom pipeline construction that replaces ToolsExecutor with TraceHookedToolsExecutor.",
    }
    print(json.dumps(message, indent=2), file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
