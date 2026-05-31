from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib
import json
import os

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter
from src.adapters.agentdojo_pipeline_builder import (
    build_traced_agentdojo_pipeline,
    inspect_agentdojo_pipeline,
)


def _provider_status(args: argparse.Namespace) -> dict:
    if args.provider != "openai-compatible":
        return {
            "provider": args.provider,
            "model": args.model,
            "requires_env": [],
        }
    return {
        "provider": args.provider,
        "model": args.model,
        "requires_env": ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "openai_base_url_present": bool(os.getenv("OPENAI_BASE_URL")),
    }


def _cost_guard_status(args: argparse.Namespace) -> dict:
    return {
        "enabled": bool(args.cost_guard),
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
    }


def _import_load_suites(package_name: str):
    module_name = f"{package_name}.task_suite.load_suites"
    return importlib.import_module(module_name)


def list_suites(package_name: str, benchmark_version: str) -> dict:
    plan = inspect_agentdojo_pipeline(package_name=package_name)
    if not plan.agentdojo_installed:
        return {"status": "AgentDojo not installed", "suites": [], "agentdojo": plan.to_dict()}
    try:
        load_suites = _import_load_suites(package_name)
        suites = sorted(load_suites.get_suites(benchmark_version).keys())
        return {"status": "ok", "benchmark_version": benchmark_version, "suites": suites}
    except Exception as exc:
        return {
            "status": "error",
            "benchmark_version": benchmark_version,
            "suites": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def list_tasks(package_name: str, benchmark_version: str, suite_name: str) -> dict:
    plan = inspect_agentdojo_pipeline(package_name=package_name)
    if not plan.agentdojo_installed:
        return {"status": "AgentDojo not installed", "suite": suite_name, "user_tasks": [], "agentdojo": plan.to_dict()}
    try:
        load_suites = _import_load_suites(package_name)
        suite = load_suites.get_suite(benchmark_version, suite_name)
        user_tasks = sorted(suite.user_tasks.keys())
        injection_tasks = sorted(suite.injection_tasks.keys())
        return {
            "status": "ok",
            "benchmark_version": benchmark_version,
            "suite": suite_name,
            "user_tasks": user_tasks,
            "injection_tasks": injection_tasks,
        }
    except Exception as exc:
        return {
            "status": "error",
            "benchmark_version": benchmark_version,
            "suite": suite_name,
            "user_tasks": [],
            "injection_tasks": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_dry_run_report(args: argparse.Namespace) -> dict:
    plan = inspect_agentdojo_pipeline(package_name=args.agentdojo_package)
    return {
        "mode": "dry_run",
        "suite": args.suite,
        "user_task": args.user_task,
        "trace": args.trace,
        "provider": _provider_status(args),
        "cost_guard": _cost_guard_status(args),
        "agentdojo": plan.to_dict(),
        "will_call_provider": False,
        "will_run_benchmark": False,
        "will_modify_agentdojo_source": False,
        "will_monkey_patch": False,
    }


def _blocked(message: dict, exit_code: int = 2) -> None:
    print(json.dumps(message, indent=2), file=sys.stderr)
    raise SystemExit(exit_code)


def _build_llm(args: argparse.Namespace):
    if args.model is None:
        raise ValueError(f"--model is required for provider {args.provider}.")

    if args.provider == "openai-compatible":
        import openai
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        llm = OpenAILLM(client, args.model, temperature=args.temperature)
        llm.name = args.model
        return llm

    if args.provider == "local":
        import openai
        from agentdojo.agent_pipeline.llms.local_llm import LocalLLM

        port = os.getenv("LOCAL_LLM_PORT", "8000")
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )
        llm = LocalLLM(client, args.model, temperature=args.temperature)
        llm.name = args.model
        return llm

    raise ValueError(f"Unsupported provider: {args.provider}")


def _build_pipeline_config(args: argparse.Namespace, llm):
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig

    return PipelineConfig(
        llm=llm,
        model_id=None,
        defense=None,
        tool_delimiter="tool",
        system_message_name=None,
        system_message=None,
        tool_output_format="yaml",
    )


def _run_real_smoke(args: argparse.Namespace, plan) -> dict:
    if not plan.executable:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": "AgentDojo pipeline components are not available for traced custom construction.",
            "agentdojo": plan.to_dict(),
        })

    load_suites = _import_load_suites(args.agentdojo_package)
    suite = load_suites.get_suite(args.benchmark_version, args.suite)
    user_task = suite.get_user_task_by_id(args.user_task)

    prompt = getattr(user_task, "PROMPT", getattr(user_task, "GOAL", ""))
    adapter = AgentDojoTraceAdapter(args.trace)
    adapter.start_episode(
        suite_name=args.suite,
        user_task_id=args.user_task,
        injection_task_id=None,
        attack_type=None,
        prompt=prompt,
    )

    llm = _build_llm(args)
    config = _build_pipeline_config(args, llm)
    pipeline = build_traced_agentdojo_pipeline(
        config=config,
        adapter=adapter,
        max_iters=args.max_steps,
        max_tool_calls=args.max_tool_calls if args.cost_guard else None,
    )

    if not args.allow_provider_call:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": (
                "Traced AgentDojo pipeline wiring was constructed, but provider calls require "
                "--allow-provider-call."
            ),
            "suite": args.suite,
            "user_task": args.user_task,
            "trace": args.trace,
            "provider": _provider_status(args),
            "cost_guard": _cost_guard_status(args),
            "agentdojo": plan.to_dict(),
            "wiring": {
                "pipeline": type(pipeline).__module__ + "." + type(pipeline).__name__,
                "native_executor_replaced": "agentdojo.agent_pipeline.tool_execution.ToolsExecutor",
                "replacement": "src.adapters.agentdojo_tools_wrapper.TraceHookedToolsExecutor",
                "monkey_patch": False,
            },
        })

    utility, security = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        injection_task=None,
        injections={},
    )
    adapter.final_hook(
        model_output={
            "status": "completed",
            "suite": args.suite,
            "user_task": args.user_task,
        },
        utility=utility,
        security=security,
    )
    return {
        "mode": "run",
        "status": "ok",
        "suite": args.suite,
        "user_task": args.user_task,
        "trace": args.trace,
        "provider": _provider_status(args),
        "cost_guard": _cost_guard_status(args),
        "utility": utility,
        "security": security,
        "agentdojo": plan.to_dict(),
        "wiring": {
            "native_executor_replaced": "agentdojo.agent_pipeline.tool_execution.ToolsExecutor",
            "replacement": "src.adapters.agentdojo_tools_wrapper.TraceHookedToolsExecutor",
            "monkey_patch": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a minimal AgentDojo Safety Thermometer smoke trace run.")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--user-task", default="user_task_0")
    parser.add_argument("--trace", default="outputs/agentdojo_smoke_trace.jsonl")
    parser.add_argument("--provider", choices=["local", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--allow-real-run", action="store_true")
    parser.add_argument(
        "--allow-provider-call",
        action="store_true",
        help="Allow the smoke run to call the configured LLM provider after traced pipeline wiring succeeds.",
    )
    parser.add_argument("--cost-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--agentdojo-package",
        default="agentdojo",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.list_suites:
        print(json.dumps(list_suites(args.agentdojo_package, args.benchmark_version), indent=2))
        return

    if args.list_tasks:
        print(json.dumps(list_tasks(args.agentdojo_package, args.benchmark_version, args.suite), indent=2))
        return

    if args.dry_run:
        report = build_dry_run_report(args)
        print(json.dumps(report, indent=2))
        return

    plan = inspect_agentdojo_pipeline(package_name=args.agentdojo_package)
    if not args.allow_real_run:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": "Real smoke runs require --allow-real-run.",
            "suite": args.suite,
            "user_task": args.user_task,
            "trace": args.trace,
            "provider": _provider_status(args),
            "cost_guard": _cost_guard_status(args),
            "agentdojo": plan.to_dict(),
        })

    provider = _provider_status(args)
    if args.provider == "openai-compatible":
        missing = [
            name for name in ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
            if not os.getenv(name)
        ]
        if missing:
            _blocked({
                "mode": "run",
                "status": "blocked",
                "reason": "Missing required OpenAI-compatible provider environment variables.",
                "missing_env": missing,
                "provider": provider,
            })

    try:
        result = _run_real_smoke(args, plan)
    except Exception as exc:
        _blocked({
            "mode": "run",
            "status": "blocked",
            "reason": f"{type(exc).__name__}: {exc}",
            "suite": args.suite,
            "user_task": args.user_task,
            "trace": args.trace,
            "provider": provider,
            "cost_guard": _cost_guard_status(args),
            "agentdojo": plan.to_dict(),
            "wiring": {
                "native_constructor_mirrored": "agentdojo.agent_pipeline.agent_pipeline.AgentPipeline.from_config",
                "native_executor_replaced": "agentdojo.agent_pipeline.tool_execution.ToolsExecutor",
                "replacement": "src.adapters.agentdojo_tools_wrapper.TraceHookedToolsExecutor",
                "monkey_patch": False,
            },
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
