The guarded AgentDojo smoke script is complete.

Current status:
- AgentDojo is installed and visible in the safetythermo environment.
- scripts/run_agentdojo_smoke_trace.py supports:
  --dry-run
  --list-suites
  --list-tasks
  --allow-real-run
  --provider openai-compatible
  --model
  --max-steps
  --max-tool-calls
  --max-output-tokens
  --temperature
  --cost-guard
- It does not print or store API keys.
- It currently blocks real run with not_implemented.
- AgentDojoTraceAdapter exists.
- TraceHookedToolsExecutor exists.
- agentdojo_pipeline_builder inspection helper exists.

Next goal:
Implement the minimal real AgentDojo custom pipeline wiring so that a single AgentDojo task can be run with TraceHookedToolsExecutor replacing the native ToolsExecutor.

Important:
Do not modify AgentDojo source code.
Do not monkey patch by default.
Do not run a full benchmark.
Do not modify replay.py.
Do not modify Risk Estimator or calibration.
Do not add RL.
Do not require API key in tests.
Do not print API keys.
Do not silently run native AgentDojo without tracing.
If custom pipeline construction is too version-dependent, fail safely and document the exact blocker.

Please inspect installed AgentDojo source:
- agentdojo.agent_pipeline.agent_pipeline
- agentdojo.agent_pipeline.tool_execution
- agentdojo.task_suite.task_suite
- agentdojo.benchmark
- agentdojo.logging
- agentdojo.types

Please inspect project files:
- scripts/run_agentdojo_smoke_trace.py
- src/adapters/agentdojo_adapter.py
- src/adapters/agentdojo_tools_wrapper.py
- src/adapters/agentdojo_pipeline_builder.py
- docs/agentdojo_smoke_run_plan.md
- tests/test_agentdojo_smoke_script.py

Task:
1. Extend src/adapters/agentdojo_pipeline_builder.py with a real or near-real custom pipeline construction helper.

Goal:
Construct an AgentDojo AgentPipeline that is equivalent to the native pipeline but replaces native ToolsExecutor with our TraceHookedToolsExecutor.

2. Determine how AgentDojo builds pipelines in the installed version:
- How AgentPipeline.from_config(...) works
- Where ToolsExecutionLoop is constructed
- What parameters ToolsExecutor receives
- How LLM provider elements are inserted
- How max steps / max iterations are controlled

3. If feasible, implement:
   build_traced_agentdojo_pipeline(...)

The helper should accept:
- trace adapter
- provider/model config
- max_steps
- max_tool_calls
- temperature
- max_output_tokens if supported

It should return a traced pipeline object ready for TaskSuite.run_task_with_pipeline(...).

4. Update scripts/run_agentdojo_smoke_trace.py:
- keep dry-run unchanged
- keep list-suites/list-tasks unchanged
- in non-dry-run with --allow-real-run:
  - validate OPENAI_API_KEY and OPENAI_BASE_URL only for provider=openai-compatible
  - construct traced pipeline
  - run exactly one selected task if real wiring is available
  - write TraceEvent JSONL to --trace
  - never run native untraced pipeline silently
  - enforce max_steps/max_tool_calls if feasible
  - if wiring is incomplete, fail with clear JSON error and exit code 2

5. Do not actually call any provider in tests.

6. Add or update tests:
- dry-run still passes
- list-suites/list-tasks still pass
- non-dry-run requires --allow-real-run
- provider=openai-compatible requires env vars
- API key is never printed
- if build_traced_agentdojo_pipeline is unavailable, script fails safely
- if mocked build_traced_agentdojo_pipeline succeeds, script uses the traced pipeline path, not native untraced path
- max_steps/max_tool_calls are passed through or reported

7. Update docs/agentdojo_smoke_run_plan.md:
Add a section:
"Real pipeline wiring status"
including:
- whether build_traced_agentdojo_pipeline is implemented
- exact command for first real smoke run
- how to set OPENAI_API_KEY and OPENAI_BASE_URL
- warning about low balance and max_tool_calls=3
- validation commands:
  python scripts/run_agentdojo_smoke_trace.py --list-suites
  python scripts/run_agentdojo_smoke_trace.py --suite workspace --list-tasks
  python scripts/replay.py --trace outputs/agentdojo_smoke_trace.jsonl --out outputs/agentdojo_smoke_prefix_dataset.csv

After changes, report:
- modified files
- whether real pipeline wiring is implemented or still blocked
- exact blocker if blocked
- tests added
- commands to run
- first real smoke run command