The local TraceHookedToolsExecutor mock wrapper is complete.

Current completed AgentDojo integration work:
- AgentDojoTraceAdapter works with mock events.
- TraceHookedToolsExecutor wraps tool-call execution in mock tests.
- It emits pre_step/post_step TraceEvents.
- It does not modify AgentDojo source.
- It does not monkey patch.
- It does not require OpenAI API in tests.

Next goal:
Design and implement a minimal custom AgentDojo pipeline construction helper that can replace AgentDojo's native ToolsExecutor with our TraceHookedToolsExecutor.

Important:
Do not modify AgentDojo source code.
Do not monkey patch by default.
Do not run a full benchmark.
Do not require OpenAI API in tests.
Do not modify replay.py.
Do not modify Risk Estimator or calibration.
Do not add RL.

Please inspect:
- docs/agentdojo_adapter_notes.md
- docs/agentdojo_smoke_run_plan.md
- src/adapters/agentdojo_adapter.py
- src/adapters/agentdojo_tools_wrapper.py
- tests/test_agentdojo_tools_wrapper.py

Also inspect installed AgentDojo package if available:
- agent_pipeline/agent_pipeline.py
- agent_pipeline/tool_execution.py
- agent_pipeline/base_pipeline_element.py
- benchmark.py
- task_suite/task_suite.py

Task:
1. Create:
   src/adapters/agentdojo_pipeline_builder.py

2. Implement a minimal helper function or class that documents and, where possible, constructs a custom AgentDojo pipeline using the same components as the native AgentPipeline but replacing ToolsExecutor with TraceHookedToolsExecutor.

3. If direct construction is too version-dependent, implement a safe inspection/design helper instead:
   - detect whether AgentDojo is installed
   - locate the native AgentPipeline / ToolsExecutor classes
   - expose a function that returns a structured integration plan
   - do not fail tests if AgentDojo is not installed

4. Add tests:
   tests/test_agentdojo_pipeline_builder.py

Tests should verify:
- the builder module imports without AgentDojo installed
- it returns a clear "AgentDojo not installed" status if unavailable
- it can produce a structured plan for replacing ToolsExecutor
- it does not monkey patch anything
- it does not require OpenAI API

5. Update:
   docs/agentdojo_smoke_run_plan.md

Add a section:
"Real smoke run checklist"

It should include:
- install AgentDojo
- choose minimal suite/task
- configure provider or local model
- construct custom pipeline with TraceHookedToolsExecutor
- run one task
- write AgentDojo trace JSONL
- replay trace into prefix dataset
- do not use utility/security as online input

After changes, report:
- modified files
- tests added
- whether builder is executable or design-only
- what exact command will be needed for the first real smoke run