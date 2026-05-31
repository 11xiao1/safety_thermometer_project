The AgentDojo trace adapter skeleton is complete.

Current completed work:
- AgentDojoTraceAdapter exists.
- start_episode / pre_step_hook / post_step_hook / final_hook work.
- Mock tests pass.
- The adapter writes project TraceEvent JSONL.
- No real AgentDojo benchmark has been run yet.

Next goal:
Implement a local TraceHookedToolsExecutor wrapper that mirrors AgentDojo's ToolsExecutor boundary and emits Safety Thermometer pre/post TraceEvents around each tool call.

Important:
Do not modify AgentDojo source code.
Do not monkey patch AgentDojo.
Do not run a full benchmark.
Do not require OpenAI API in tests.
Do not modify replay.py.
Do not modify Risk Estimator or calibration code.
Do not add RL.

Please inspect:
- docs/agentdojo_adapter_notes.md
- docs/agentdojo_smoke_run_plan.md
- src/adapters/agentdojo_adapter.py
- src/monitor/schema.py
- src/monitor/logger.py
- tests/test_agentdojo_adapter.py

Also inspect the installed AgentDojo package if available, especially:
- agent_pipeline/tool_execution.py
- agent_pipeline/base_pipeline_element.py
- functions_runtime.py
- types.py

Task:
1. Create or update:
   src/adapters/agentdojo_tools_wrapper.py

2. Implement a local TraceHookedToolsExecutor class.

Purpose:
It should wrap the same logical boundary as AgentDojo ToolsExecutor:
- read the latest assistant message
- find tool_calls
- for each tool_call:
  - emit adapter.pre_step_hook(...)
  - call runtime.run_function(env, tool_call.function, tool_call.args)
  - format or preserve the tool result as observation
  - emit adapter.post_step_hook(...)
  - append or return a tool-result-like message structure if feasible

3. Keep the implementation minimally coupled:
- Use duck typing where possible.
- Tests should use fake messages, fake tool calls, fake runtime, and fake env.
- It is acceptable if the wrapper is not yet wired into AgentDojo's actual AgentPipeline.
- Do not import OpenAI or call any provider.

4. Add tests:
   tests/test_agentdojo_tools_wrapper.py

Tests should verify:
- one fake tool call emits one pre_step and one post_step event
- multiple tool calls emit ordered step_ids
- runtime.run_function is called with expected function name and args
- post_step observation includes tool result
- errors from runtime are captured in post_step notes or observation
- produced JSONL can be loaded with the existing loader

5. Update docs:
   docs/agentdojo_smoke_run_plan.md

Add a section:
"Next real smoke run plan"
including:
- how to replace AgentDojo ToolsExecutor with TraceHookedToolsExecutor in a custom pipeline
- what remains unimplemented
- what may require API or local model later

After changes, report:
- modified files
- new tests
- command to run tests
- what is still mock-only
- what is needed for the first real AgentDojo smoke run