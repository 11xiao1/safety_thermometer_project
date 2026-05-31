The AgentDojo pipeline builder / inspection helper is complete.

Current completed work:
- AgentDojoTraceAdapter
- TraceHookedToolsExecutor
- agentdojo_pipeline_builder inspection helper
- mock tests pass
- no AgentDojo source modification
- no monkey patching
- no full benchmark run yet

Next goal:
Create the first real AgentDojo smoke trace script.

Important:
Do not modify AgentDojo source code.
Do not monkey patch by default.
Do not run full benchmark in tests.
Do not modify replay.py.
Do not modify Risk Estimator or calibration.
Do not add RL.
Do not require OpenAI API in tests.

Please inspect:
- docs/agentdojo_adapter_notes.md
- docs/agentdojo_smoke_run_plan.md
- src/adapters/agentdojo_adapter.py
- src/adapters/agentdojo_tools_wrapper.py
- src/adapters/agentdojo_pipeline_builder.py
- AgentDojo installed package if available

Task:
1. Create:
   scripts/run_agentdojo_smoke_trace.py

2. The script should provide a CLI shape:

   python scripts/run_agentdojo_smoke_trace.py ^
     --suite workspace ^
     --user-task user_task_0 ^
     --trace outputs/agentdojo_smoke_trace.jsonl ^
     --dry-run

3. Implement --dry-run first:
   - inspect AgentDojo availability
   - print selected suite/task/provider configuration
   - print the proposed ToolsExecutor replacement plan
   - do not call any LLM provider
   - do not run benchmark
   - do not require API key
   - exit successfully if AgentDojo is installed and the plan is valid

4. If feasible, add a non-dry-run skeleton:
   - clearly raise NotImplementedError or a helpful message if full pipeline wiring is not complete
   - do not silently run native AgentDojo without tracing

5. Add tests:
   tests/test_agentdojo_smoke_script.py

Tests should verify:
- --dry-run runs without API key
- --dry-run does not call provider
- script exits cleanly with structured output
- if AgentDojo is unavailable, it reports a clear message instead of crashing
- non-dry-run fails safely with a clear "not implemented" message until real wiring is complete

6. Update:
   docs/agentdojo_smoke_run_plan.md

Add:
- exact dry-run command
- exact future non-dry-run command
- what API/model configuration will be needed later
- how to validate outputs/agentdojo_smoke_trace.jsonl after a real run
- how to pass the trace into replay.py

After changes, report:
- modified files
- commands to run
- tests added
- whether dry-run works
- what remains before a real AgentDojo task can run