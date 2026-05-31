# Codex Handoff: Safety Thermometer Current State

## Current method

Trajectory Prefix
→ Evidence Streams
→ future_risk_label
→ Supervised Risk Estimator
→ uncalibrated risk probability
→ Calibration
→ 0–100 Thermometer Score
→ threshold policy
→ Table 3 / Table 4

Oracle is auxiliary supervision/diagnostic only.
RL is not part of the current phase.

## Completed

- Toy replay pipeline works.
- Prefix dataset generation works.
- Oracle tests work.
- Supervised Risk Estimator baseline works.
- Calibration layer works.
- Thermometer Score CSV works.
- Toy Table 3 / Table 4 works.
- AgentDojo installed in safetythermo.
- AgentDojo adapter skeleton works.
- TraceHookedToolsExecutor mock wrapper works.
- AgentDojo pipeline inspection helper works.
- run_agentdojo_smoke_trace.py dry-run works.
- list-suites and list-tasks work.
- Minimal custom AgentDojo pipeline wiring is implemented locally via
  build_traced_agentdojo_pipeline(...).

## Current environment

Use this Python:

F:\Anaconda_envs\envs\safetythermo\python.exe

Tests:

F:\Anaconda_envs\envs\safetythermo\python.exe -m pytest -q --basetemp .pytest_tmp

AgentDojo is installed:

agentdojo==0.1.35

Available suites:

banking
slack
travel
workspace

## Current blocking point

Real AgentDojo smoke run is now blocked only on explicit provider-call opt-in
and valid provider configuration. The script builds the traced custom pipeline
before any provider request and requires `--allow-provider-call` to actually
call the LLM.

## Current task

Next task:

Run one real traced AgentDojo smoke task with a configured provider, then
validate and replay the emitted JSONL.

Do not:
- modify AgentDojo source
- monkey patch by default
- modify replay.py
- modify Risk Estimator
- modify calibration
- add RL
- call provider in tests
- print API keys
- silently run native AgentDojo without tracing

Need:
- inspect installed AgentDojo source
- determine how AgentPipeline.from_config builds ToolsExecutionLoop
- determine how ToolsExecutor is inserted
- build custom pipeline with TraceHookedToolsExecutor if feasible
- otherwise fail safely with precise blocker
- keep dry-run/list-suites/list-tasks working

## Current exact next task

Implement minimal real AgentDojo pipeline wiring.

Primary files:
- src/adapters/agentdojo_pipeline_builder.py
- scripts/run_agentdojo_smoke_trace.py
- tests/test_agentdojo_smoke_script.py
- docs/agentdojo_smoke_run_plan.md

Exit criteria:
- non-dry-run requires --allow-real-run
- OpenAI-compatible env vars checked but never printed
- no provider calls in tests
- if real wiring infeasible, return exact blocker
