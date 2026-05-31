import pytest

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter
from src.adapters.agentdojo_pipeline_builder import (
    build_traced_agentdojo_pipeline,
    inspect_agentdojo_pipeline,
    make_hooked_tools_executor,
)
from src.adapters.agentdojo_tools_wrapper import TraceHookedToolsExecutor


def test_builder_module_imports_without_requiring_agentdojo():
    plan = inspect_agentdojo_pipeline(package_name="definitely_not_installed_agentdojo")

    assert plan.agentdojo_installed is False


def test_builder_reports_agentdojo_not_installed_when_unavailable():
    plan = inspect_agentdojo_pipeline(package_name="definitely_not_installed_agentdojo")

    assert plan.status == "AgentDojo not installed"
    assert plan.executable is False
    assert plan.monkey_patch_required is False


def test_builder_returns_structured_tools_executor_replacement_plan():
    plan = inspect_agentdojo_pipeline(package_name="definitely_not_installed_agentdojo")
    data = plan.to_dict()

    assert "replacement" in data
    assert "TraceHookedToolsExecutor" in data["replacement"]["ToolsExecutor"]
    assert isinstance(data["next_steps"], list)


def test_builder_does_not_monkey_patch_anything():
    plan = inspect_agentdojo_pipeline()

    assert plan.monkey_patch_required is False


def test_make_hooked_tools_executor_requires_no_openai_api(tmp_path):
    adapter = AgentDojoTraceAdapter(tmp_path / "trace.jsonl")
    adapter.start_episode("workspace", "user_task_0")

    executor = make_hooked_tools_executor(adapter, tool_output_formatter=str)

    assert isinstance(executor, TraceHookedToolsExecutor)


def test_build_traced_pipeline_replaces_native_tools_executor(tmp_path):
    pytest.importorskip("agentdojo")
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

    class FakeLLM(BasePipelineElement):
        name = "fake-llm"

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            return query, runtime, env, list(messages), extra_args or {}

    adapter = AgentDojoTraceAdapter(tmp_path / "trace.jsonl")
    adapter.start_episode("workspace", "user_task_0")
    config = PipelineConfig(
        llm=FakeLLM(),
        model_id=None,
        defense=None,
        tool_delimiter="tool",
        system_message_name=None,
        system_message="system",
        tool_output_format="yaml",
    )

    pipeline = build_traced_agentdojo_pipeline(config, adapter, max_iters=2, max_tool_calls=1)
    tools_loop = next(element for element in pipeline.elements if isinstance(element, ToolsExecutionLoop))

    assert isinstance(tools_loop.elements[0], TraceHookedToolsExecutor)
    assert tools_loop.elements[0].max_tool_calls == 1
    assert tools_loop.max_iters == 2
