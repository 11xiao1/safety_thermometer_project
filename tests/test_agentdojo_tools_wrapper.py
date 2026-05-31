from dataclasses import dataclass

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter
from src.adapters.agentdojo_tools_wrapper import TraceHookedToolsExecutor
from src.monitor.logger import load_trace_events


@dataclass
class FakeToolCall:
    function: str
    args: dict
    id: str


@dataclass
class FakeFunction:
    name: str


class FakeRuntime:
    def __init__(self, results=None):
        self.functions = {
            "read_file": FakeFunction("read_file"),
            "send_email": FakeFunction("send_email"),
        }
        self.calls = []
        self.results = results or {}

    def run_function(self, env, function, args):
        self.calls.append((env, function, args))
        return self.results.get(function, (f"result:{function}", None))


def _executor(tmp_path, results=None):
    trace_path = tmp_path / "agentdojo_trace.jsonl"
    adapter = AgentDojoTraceAdapter(trace_path)
    adapter.start_episode("workspace", "user_task_0", prompt="Read then summarize.")
    return TraceHookedToolsExecutor(adapter), trace_path


def _assistant_message(tool_calls):
    return {
        "role": "assistant",
        "content": [{"type": "text", "content": "I need to use a tool."}],
        "tool_calls": tool_calls,
    }


def test_one_tool_call_emits_pre_and_post_events(tmp_path):
    executor, trace_path = _executor(tmp_path)
    runtime = FakeRuntime()
    tool_call = FakeToolCall("read_file", {"path": "doc.md"}, "call_1")

    executor.query("query", runtime, env={}, messages=[_assistant_message([tool_call])], extra_args={})

    events = load_trace_events(trace_path)
    assert [event.hook_type for event in events] == ["pre_step", "post_step"]
    assert [event.step_id for event in events] == [1, 1]


def test_multiple_tool_calls_emit_ordered_step_ids(tmp_path):
    executor, trace_path = _executor(tmp_path)
    runtime = FakeRuntime()

    executor.query(
        "query",
        runtime,
        env={},
        messages=[_assistant_message([
            FakeToolCall("read_file", {"path": "a.md"}, "call_1"),
            FakeToolCall("send_email", {"to": "a@example.com"}, "call_2"),
        ])],
        extra_args={},
    )

    events = load_trace_events(trace_path)
    assert [(event.hook_type, event.step_id) for event in events] == [
        ("pre_step", 1),
        ("post_step", 1),
        ("pre_step", 2),
        ("post_step", 2),
    ]


def test_runtime_run_function_called_with_expected_function_and_args(tmp_path):
    executor, _ = _executor(tmp_path)
    runtime = FakeRuntime()
    env = {"state": "fake"}
    tool_call = FakeToolCall("read_file", {"path": "doc.md"}, "call_1")

    executor.query("query", runtime, env=env, messages=[_assistant_message([tool_call])], extra_args={})

    assert runtime.calls == [(env, "read_file", {"path": "doc.md"})]


def test_post_step_observation_includes_tool_result(tmp_path):
    executor, trace_path = _executor(tmp_path, results={"read_file": ("Document body", None)})
    runtime = FakeRuntime(results={"read_file": ("Document body", None)})
    tool_call = FakeToolCall("read_file", {"path": "doc.md"}, "call_1")

    _, _, _, messages, _ = executor.query(
        "query",
        runtime,
        env={},
        messages=[_assistant_message([tool_call])],
        extra_args={},
    )

    events = load_trace_events(trace_path)
    assert events[-1].observation == "Document body"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"][0]["content"] == "Document body"


def test_runtime_errors_are_captured_in_post_step(tmp_path):
    executor, trace_path = _executor(tmp_path)
    runtime = FakeRuntime(results={"read_file": ("", "ValueError: missing file")})
    tool_call = FakeToolCall("read_file", {"path": "missing.md"}, "call_1")

    executor.query("query", runtime, env={}, messages=[_assistant_message([tool_call])], extra_args={})

    events = load_trace_events(trace_path)
    assert events[-1].hook_type == "post_step"
    assert events[-1].observation == "ValueError: missing file"
    assert "ValueError: missing file" in events[-1].notes


def test_wrapper_jsonl_can_be_loaded_with_existing_loader(tmp_path):
    executor, trace_path = _executor(tmp_path)
    runtime = FakeRuntime()

    executor.query(
        "query",
        runtime,
        env={},
        messages=[_assistant_message([FakeToolCall("read_file", {"path": "doc.md"}, "call_1")])],
        extra_args={},
    )

    events = load_trace_events(trace_path)
    assert len(events) == 2
    assert events[0].episode_id == "workspace:user_task_0:none:none"


def test_tool_call_limit_blocks_extra_runtime_calls(tmp_path):
    trace_path = tmp_path / "agentdojo_trace.jsonl"
    adapter = AgentDojoTraceAdapter(trace_path)
    adapter.start_episode("workspace", "user_task_0", prompt="Read then summarize.")
    executor = TraceHookedToolsExecutor(adapter, max_tool_calls=1)
    runtime = FakeRuntime()

    _, _, _, messages, _ = executor.query(
        "query",
        runtime,
        env={},
        messages=[_assistant_message([
            FakeToolCall("read_file", {"path": "a.md"}, "call_1"),
            FakeToolCall("send_email", {"to": "a@example.com"}, "call_2"),
        ])],
        extra_args={},
    )

    assert [call[1] for call in runtime.calls] == ["read_file"]
    assert messages[-1]["error"] == "Tool call limit exceeded: max_tool_calls=1."
    events = load_trace_events(trace_path)
    assert [event.hook_type for event in events] == ["pre_step", "post_step", "pre_step", "post_step"]
