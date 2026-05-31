"""Helpers for planning a traced AgentDojo pipeline.

The functions in this module are intentionally safe to import without
AgentDojo installed. They do not monkey patch AgentDojo and do not call any LLM
provider. The first real smoke run can use this module to inspect availability
and then build a custom pipeline in a separate integration script.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, asdict
from typing import Any

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter
from src.adapters.agentdojo_tools_wrapper import TraceHookedToolsExecutor


AGENTDOJO_MODULES = {
    "AgentPipeline": "agentdojo.agent_pipeline.agent_pipeline",
    "PipelineConfig": "agentdojo.agent_pipeline.agent_pipeline",
    "ToolsExecutor": "agentdojo.agent_pipeline.tool_execution",
    "ToolsExecutionLoop": "agentdojo.agent_pipeline.tool_execution",
    "tool_result_to_str": "agentdojo.agent_pipeline.tool_execution",
    "BasePipelineElement": "agentdojo.agent_pipeline.base_pipeline_element",
}


@dataclass(frozen=True)
class AgentDojoPipelinePlan:
    status: str
    agentdojo_installed: bool
    executable: bool
    monkey_patch_required: bool
    native_components: dict[str, str]
    replacement: dict[str, str]
    next_steps: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _class_available(module_name: str, class_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, class_name)


def inspect_agentdojo_pipeline(package_name: str = "agentdojo") -> AgentDojoPipelinePlan:
    installed = importlib.util.find_spec(package_name) is not None
    if not installed:
        return AgentDojoPipelinePlan(
            status="AgentDojo not installed",
            agentdojo_installed=False,
            executable=False,
            monkey_patch_required=False,
            native_components={},
            replacement={
                "ToolsExecutor": "src.adapters.agentdojo_tools_wrapper.TraceHookedToolsExecutor",
            },
            next_steps=[
                "Install AgentDojo in the active environment.",
                "Re-run this inspection helper.",
                "Construct a custom pipeline only after native components are discoverable.",
            ],
            warnings=["No AgentDojo imports were attempted because the package was not found."],
        )

    native_components = {}
    missing = []
    for class_name, module_name in AGENTDOJO_MODULES.items():
        if _class_available(module_name, class_name):
            native_components[class_name] = f"{module_name}.{class_name}"
        else:
            missing.append(f"{module_name}.{class_name}")

    executable = len(missing) == 0
    warnings = []
    if missing:
        warnings.append("Some expected AgentDojo components were not found: " + ", ".join(missing))

    return AgentDojoPipelinePlan(
        status="AgentDojo pipeline components available" if executable else "AgentDojo installed but incomplete",
        agentdojo_installed=True,
        executable=executable,
        monkey_patch_required=False,
        native_components=native_components,
        replacement={
            "replace": "agentdojo.agent_pipeline.tool_execution.ToolsExecutor",
            "with": "src.adapters.agentdojo_tools_wrapper.TraceHookedToolsExecutor",
            "boundary": "inside AgentDojo ToolsExecutionLoop, before each runtime.run_function call",
        },
        next_steps=[
            "Create an AgentDojoTraceAdapter for the desired JSONL path.",
            "Construct AgentDojo's normal system/init/LLM components.",
            "Construct ToolsExecutionLoop with TraceHookedToolsExecutor in the ToolsExecutor position.",
            "Run one selected task only; do not run a full benchmark yet.",
            "Load the emitted JSONL with load_trace_events and replay it into a prefix dataset.",
        ],
        warnings=warnings,
    )


def make_hooked_tools_executor(
    adapter: AgentDojoTraceAdapter,
    tool_output_formatter=None,
) -> TraceHookedToolsExecutor:
    if tool_output_formatter is None:
        plan = inspect_agentdojo_pipeline()
        if plan.agentdojo_installed and "tool_result_to_str" in plan.native_components:
            module = importlib.import_module(AGENTDOJO_MODULES["tool_result_to_str"])
            tool_output_formatter = getattr(module, "tool_result_to_str")
    if tool_output_formatter is None:
        return TraceHookedToolsExecutor(adapter=adapter)
    return TraceHookedToolsExecutor(adapter=adapter, tool_output_formatter=tool_output_formatter)
