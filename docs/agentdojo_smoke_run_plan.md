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
