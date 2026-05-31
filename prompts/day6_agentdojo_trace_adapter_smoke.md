The toy Safety Thermometer pipeline is complete.

Current completed chain:
Trajectory Prefix
→ Evidence Streams
→ future_risk_label
→ Supervised Risk Estimator
→ Calibration
→ 0–100 Thermometer Score
→ Table 3 / Table 4

Now we move to AgentDojo, but only for a minimal trace adapter smoke run.

Important:
Do not modify AgentDojo source code.
Do not monkey patch by default.
Do not run full benchmark.
Do not modify replay.py unless strictly necessary.
Do not modify Risk Estimator or calibration logic.
Do not add RL.
Do not require OpenAI API in tests.

Please inspect:
- docs/agentdojo_adapter_notes.md
- src/monitor/schema.py
- src/monitor/logger.py
- src/adapters/agentdojo_adapter.py
- tests/

Task:
Implement a minimal local AgentDojo trace adapter skeleton.

Requirements:
1. Create or update src/adapters/agentdojo_adapter.py.
2. Implement an AgentDojoTraceAdapter class that can:
   - start_episode(...)
   - emit pre_step_hook(...)
   - emit post_step_hook(...)
   - emit final_hook(...)
3. The adapter should write project TraceEvent JSONL using the existing TraceEvent schema/logger.
4. Do not depend on actually running AgentDojo in tests.
5. Add mock tests using fake AgentDojo-like tool_call / assistant_message / tool_result objects.
6. Tests should verify:
   - pre_step emits a valid TraceEvent
   - post_step emits a valid TraceEvent with observation
   - final_hook emits a valid final TraceEvent
   - multiple tool calls get increasing step_id
   - output JSONL can be loaded by the existing replay pipeline or TraceEvent loader
7. Add a short docs/agentdojo_smoke_run_plan.md explaining how a future real AgentDojo run would replace ToolsExecutor with TraceHookedToolsExecutor.

Do not implement TraceHookedToolsExecutor yet unless it can be done cleanly without importing AgentDojo in tests.

After changes, report:
- modified files
- tests added
- commands to run
- what is still mock-only