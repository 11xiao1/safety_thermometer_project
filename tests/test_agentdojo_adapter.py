from dataclasses import dataclass

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter
from src.monitor.logger import load_trace_events


@dataclass
class FakeToolCall:
    function: str
    args: dict
    id: str


def _adapter(tmp_path) -> AgentDojoTraceAdapter:
    adapter = AgentDojoTraceAdapter(tmp_path / "agentdojo_trace.jsonl")
    adapter.start_episode(
        suite_name="workspace",
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        attack_type="none",
        prompt="Summarize the latest document.",
    )
    return adapter


def test_pre_step_emits_valid_trace_event(tmp_path):
    adapter = _adapter(tmp_path)

    event = adapter.pre_step_hook(
        FakeToolCall("read_file", {"path": "doc.md"}, "call_1"),
        assistant_message={"content": [{"type": "text", "content": "I will read the document."}]},
    )

    assert event.hook_type == "pre_step"
    assert event.step_id == 1
    assert event.proposed_tool == "read_file"
    assert event.tool_args == {"path": "doc.md"}


def test_post_step_emits_valid_trace_event_with_observation(tmp_path):
    adapter = _adapter(tmp_path)
    tool_call = FakeToolCall("read_file", {"path": "doc.md"}, "call_1")
    adapter.pre_step_hook(tool_call)

    event = adapter.post_step_hook(tool_call, formatted_observation="Document text")

    assert event.hook_type == "post_step"
    assert event.step_id == 1
    assert event.observation == "Document text"


def test_final_hook_emits_valid_final_trace_event(tmp_path):
    adapter = _adapter(tmp_path)

    event = adapter.final_hook(model_output="Done", utility=True, security=True)

    assert event.hook_type == "final"
    assert event.observation == "Done"
    assert event.step_id == 1


def test_multiple_tool_calls_get_increasing_step_id(tmp_path):
    adapter = _adapter(tmp_path)

    first = adapter.pre_step_hook(FakeToolCall("read_file", {"path": "a.md"}, "call_1"))
    adapter.post_step_hook(FakeToolCall("read_file", {"path": "a.md"}, "call_1"), formatted_observation="A")
    second = adapter.pre_step_hook(FakeToolCall("read_file", {"path": "b.md"}, "call_2"))

    assert first.step_id == 1
    assert second.step_id == 2


def test_agentdojo_adapter_jsonl_loads_with_existing_loader(tmp_path):
    trace_path = tmp_path / "agentdojo_trace.jsonl"
    adapter = AgentDojoTraceAdapter(trace_path)
    adapter.start_episode("workspace", "user_task_0", prompt="Read the file.")
    tool_call = FakeToolCall("read_file", {"path": "doc.md"}, "call_1")
    adapter.pre_step_hook(tool_call)
    adapter.post_step_hook(tool_call, formatted_observation="Document text")
    adapter.final_hook(model_output="Done", utility=True, security=True)

    events = load_trace_events(trace_path)
    assert [event.hook_type for event in events] == ["pre_step", "post_step", "final"]
    assert all(event.episode_id == "workspace:user_task_0:none:none" for event in events)
