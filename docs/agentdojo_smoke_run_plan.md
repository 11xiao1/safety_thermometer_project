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

## Real Smoke Run Checklist

Before the first real AgentDojo smoke run:

1. Install AgentDojo in the active environment and verify that
   `inspect_agentdojo_pipeline()` reports the native pipeline components.
2. Choose one minimal suite/task, preferably a short utility-only task before
   adding injections.
3. Configure either a provider credential or a local OpenAI-compatible model
   server.
4. Create an `AgentDojoTraceAdapter` pointing at a dedicated JSONL path such as
   `outputs/agentdojo_smoke_trace.jsonl`.
5. Construct a custom AgentDojo pipeline that mirrors the native pipeline but
   places `TraceHookedToolsExecutor` where native `ToolsExecutor` would be.
6. Run exactly one task with force-rerun enabled; do not run the full benchmark.
7. Confirm the adapter writes AgentDojo trace JSONL in the project
   `TraceEvent` schema.
8. Load the trace with `load_trace_events(...)`.
9. Replay the trace into a prefix dataset with the existing replay pipeline.
10. Keep AgentDojo `utility` and `security` as final benchmark diagnostics;
    do not use them as online monitor inputs.

The exact first smoke command will need the selected suite/task and model
provider. The expected shape should be a project-local script command such as:

```powershell
python scripts\run_agentdojo_smoke_trace.py --suite workspace --user-task user_task_0 --trace outputs\agentdojo_smoke_trace.jsonl
```

The script now supports a dry run. Use this first:

```powershell
python scripts\run_agentdojo_smoke_trace.py --suite workspace --user-task user_task_0 --trace outputs\agentdojo_smoke_trace.jsonl --dry-run
```

The future non-dry-run command should be:

```powershell
python scripts\run_agentdojo_smoke_trace.py --suite workspace --user-task user_task_0 --trace outputs\agentdojo_smoke_trace.jsonl --provider local
```

The non-dry-run path is intentionally not implemented yet; it should fail
safely rather than running native AgentDojo without tracing.

### Future API or model configuration

For a real task, configure one of:

- a local OpenAI-compatible model server and `--provider local`
- the relevant provider API key and provider/model flags
- a small deterministic/local test model if AgentDojo supports it cleanly

Tests must continue to use dry-run and fakes only.

### Validating a real trace

After a real run writes `outputs/agentdojo_smoke_trace.jsonl`, validate that it
can be loaded:

```powershell
python -c "from src.monitor.logger import load_trace_events; events = load_trace_events('outputs/agentdojo_smoke_trace.jsonl'); print(len(events)); print([e.hook_type for e in events[:5]])"
```

Then pass it into replay:

```powershell
python scripts\replay.py --trace outputs\agentdojo_smoke_trace.jsonl --out outputs\agentdojo_smoke_prefix_dataset.csv
```

Only after this succeeds should the trace be considered usable for prefix
dataset experiments.
