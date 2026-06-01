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
- AgentDojo pipeline builder works.
- run_agentdojo_smoke_trace.py dry-run works.
- First traced AgentDojo smoke task works.
- AgentDojo trace can be replayed into prefix dataset.
- add agentdojo smoke report
- AgentDojo prefix dataset splitter works by episode_id.
- Guarded AgentDojo mini-batch runner works in dry-run and tested fake-run paths.
- Offline AgentDojo mini-batch report generator works on mock outputs.


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

## Current exact next task

Prepare and validate the first real guarded AgentDojo mini-batch execution.

Primary goal:
Run a small real AgentDojo mini-batch only after an explicit manual command with provider permission. The goal is to produce real mini-batch outputs for later offline reporting, splitting, and validation.

Important:
This task should not call providers inside tests. Codex should only ensure the runner, docs, and safety checks are ready. The actual provider run will be executed manually by the user.

Primary files:
- scripts/run_agentdojo_mini_batch.py
- tests/test_agentdojo_mini_batch_runner.py
- docs/agentdojo_mini_batch_plan.md
- docs/codex_handoff_current.md

Expected real-run outputs:
- outputs/agentdojo_mini_batch/run_summary.json
- outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl
- outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv

Required preflight behavior:
- Dry-run must work without API key.
- Dry-run must list selected tasks.
- Dry-run must show output paths.
- Dry-run must show cost guards.
- Non-dry-run must require:
  - --allow-real-run
  - --allow-provider-call
- Provider calls must never happen in tests.
- API keys and base URLs must not be printed or written to files.

First real mini-batch command shape:
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\run_agentdojo_mini_batch.py ^
  --suite workspace ^
  --limit 5 ^
  --task-start user_task_0 ^
  --trace-dir outputs\agentdojo_mini_batch\traces ^
  --prefix-dir outputs\agentdojo_mini_batch\prefix ^
  --merged-out outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv ^
  --provider openai-compatible ^
  --model gpt-3.5-turbo ^
  --max-steps 3 ^
  --max-tool-calls 3 ^
  --max-output-tokens 512 ^
  --temperature 0 ^
  --cost-guard ^
  --allow-real-run ^
  --allow-provider-call

Validation after manual run:
- Confirm run_summary.json exists.
- Confirm merged trace JSONL exists.
- Confirm merged prefix dataset exists.
- Then run report_agentdojo_mini_batch.py.
- Then run split_prefix_dataset.py.
- Then create/update docs/agentdojo_mini_batch_validation.md.

Rules:
- Do not call provider in tests.
- Do not run AgentDojo in tests.
- Do not modify Risk Estimator.
- Do not modify calibration.
- Do not modify replay.py unless strictly necessary.
- Do not add RL.
- Keep existing tests passing.

Exit criteria:
- F:\Anaconda_envs\envs\safetythermo\python.exe -m pytest -q --basetemp .pytest_tmp passes.
- Dry-run command for mini-batch works.
- Real-run command is documented but not executed by Codex.
- If required output path or arguments are inconsistent, fix the runner/docs before manual execution.
