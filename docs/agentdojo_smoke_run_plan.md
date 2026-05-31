# AgentDojo Smoke Run Plan

This is a design note for the first real AgentDojo smoke run. It does not
implement the run yet.

## Goal

Verify that one small AgentDojo task can emit Safety Thermometer `TraceEvent`
JSONL without changing AgentDojo package source and without running a full
benchmark.

## Planned Integration

Use the existing `AgentDojoTraceAdapter` as the trace sink. A future local
`TraceHookedToolsExecutor` should mirror AgentDojo's `ToolsExecutor` behavior
and emit:

- `pre_step_hook(...)` immediately before each `runtime.run_function(...)`
- `post_step_hook(...)` immediately after the tool result is formatted
- `final_hook(...)` after `TaskSuite.run_task_with_pipeline(...)` returns

The first smoke run should replace `ToolsExecutor` during local pipeline
construction, not by editing installed AgentDojo files.

## Preferred Construction

1. Build a helper in this repository that mirrors `AgentPipeline.from_config`.
2. Use AgentDojo's normal LLM, system message, init query, and tools loop.
3. Replace only:

```python
ToolsExecutor(...)
```

with:

```python
TraceHookedToolsExecutor(adapter=AgentDojoTraceAdapter(...), ...)
```

4. Run one selected suite/task with `force_rerun=True` and a dedicated trace
   output path.
5. Load the resulting JSONL with `load_trace_events(...)`.
6. Replay it through the existing prefix dataset pipeline.

## What Not To Do Yet

- Do not monkey patch `ToolsExecutor.query`.
- Do not modify AgentDojo package source.
- Do not run the full benchmark.
- Do not add RL or intervention learning.
- Do not use benchmark `utility/security` as online model inputs.

## Open Questions

- Which suite/task should be the first smoke case?
- Should `state_delta` stay empty initially, or should we add compact
  environment diffs?
- How should AgentDojo `utility/security` map to `risk_event`,
  `future_risk_label`, and `future_severity` for supervised training?
- Should nested function calls be logged as separate internal steps or skipped
  in the first integration?

## Next Real Smoke Run Plan

The local `TraceHookedToolsExecutor` now mirrors the logical boundary of
AgentDojo's `ToolsExecutor` in mock tests. The next real smoke run should wire
it into a custom AgentDojo pipeline without modifying installed AgentDojo
source.

### Replacement strategy

Create a project-local pipeline construction helper that follows
`AgentPipeline.from_config(...)` but replaces only the tool executor:

```python
TraceHookedToolsExecutor(
    adapter=AgentDojoTraceAdapter("outputs/agentdojo_smoke_trace.jsonl"),
    tool_output_formatter=tool_result_to_str,
)
```

This hooked executor should sit inside AgentDojo's existing
`ToolsExecutionLoop` in the same position normally occupied by
`ToolsExecutor(...)`. The rest of the pipeline should remain native
AgentDojo components: system message, initial query setup, LLM element, and
tools loop.

### Still unimplemented

- The custom pipeline construction helper.
- A real AgentDojo suite/task selection for the first smoke run.
- Final hook emission from the `TaskSuite.run_task_with_pipeline(...)`
  completion boundary.
- Environment diffing for `state_delta`.
- Label/severity mapping from AgentDojo `utility/security` to training labels.

### May require API or local model later

- Running a real AgentDojo LLM pipeline may require an OpenAI-compatible API,
  another provider credential, or a local model server.
- Tests should continue to use fakes and should not require provider access.
- Full benchmark execution should wait until one JSONL smoke trace replays
  cleanly through the existing prefix dataset pipeline.
