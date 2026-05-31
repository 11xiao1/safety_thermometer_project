The AgentDojo smoke dry-run is working.

Current status:
- AgentDojo is installed and visible in the safetythermo environment.
- scripts/run_agentdojo_smoke_trace.py --dry-run works.
- It detects AgentPipeline, PipelineConfig, ToolsExecutor, ToolsExecutionLoop, and tool_result_to_str.
- AgentDojoTraceAdapter exists.
- TraceHookedToolsExecutor exists.
- We have not run a real AgentDojo task yet.

New constraint:
The user has an OpenAI-compatible proxy API endpoint that can route to OpenAI / Anthropic / Google models.
The available balance is small, around $7.88.
Therefore the first real smoke run must be extremely cost-limited and explicit.

Important:
Do not modify AgentDojo source code.
Do not monkey patch AgentDojo by default.
Do not run full benchmark.
Do not modify replay.py.
Do not modify Risk Estimator or calibration.
Do not add RL.
Do not silently run native AgentDojo without tracing.
Do not require API key in tests.
Do not hard-code API keys or base URLs.
Do not print API keys.
Do not write API keys into logs, docs, JSONL traces, or git-tracked files.

Provider configuration:
Support OpenAI-compatible API through environment variables:
- OPENAI_API_KEY
- OPENAI_BASE_URL

Support CLI options:
- --provider openai-compatible
- --model
- --max-steps
- --list-suites
- --list-tasks
- --allow-real-run
- --dry-run
- --cost-guard
- --max-tool-calls
- --max-output-tokens if feasible
- --temperature 0 if feasible

Recommended defaults for the first real run:
- provider: openai-compatible
- model: should be passed explicitly by user
- max_steps: 3
- max_tool_calls: 3
- temperature: 0
- cost_guard: enabled

Task:
1. Extend scripts/run_agentdojo_smoke_trace.py with guarded non-dry-run mode.

2. Add or update CLI options:
   --provider
   --model
   --max-steps
   --max-tool-calls
   --max-output-tokens
   --temperature
   --list-suites
   --list-tasks
   --allow-real-run
   --cost-guard

3. Real run must require --allow-real-run.
   If --allow-real-run is absent, exit safely with a clear message.

4. For provider=openai-compatible:
   - read OPENAI_API_KEY from environment
   - read OPENAI_BASE_URL from environment
   - fail clearly if either is missing
   - never print the key
   - print only whether the key/base URL is present
   - never store these values in output traces

5. Implement suite/task discovery if feasible:
   - list available suites
   - list available user tasks for a suite
   If AgentDojo APIs are version-dependent, implement best-effort discovery with clear errors.

6. Implement minimal custom pipeline wiring if feasible:
   - construct AgentDojo normal pipeline components
   - replace native ToolsExecutor with TraceHookedToolsExecutor
   - run exactly one selected task
   - write TraceEvent JSONL to the --trace path
   - stop after max_steps or max_tool_calls if feasible

7. If full wiring is too version-dependent, do not fake success.
   Instead:
   - keep non-dry-run failing safely
   - print the exact AgentDojo constructor/function that must be mirrored
   - keep list-suites/list-tasks and provider checks working

8. Add tests:
   - dry-run still works
   - --allow-real-run is required for non-dry-run
   - list-suites/list-tasks do not require API key
   - no provider call happens in tests
   - provider=openai-compatible fails clearly if env vars are missing
   - provider=openai-compatible does not print API key
   - non-dry-run fails safely if real wiring is incomplete
   - max_steps/max_tool_calls arguments are parsed and reported

9. Update docs/agentdojo_smoke_run_plan.md:
   Add a section for OpenAI-compatible proxy API:
   - how to set OPENAI_API_KEY
   - how to set OPENAI_BASE_URL
   - warning not to commit secrets
   - exact list-suites command
   - exact list-tasks command
   - exact first real smoke command with max_steps=3 and max_tool_calls=3
   - expected output file
   - validation command:
     python scripts/replay.py --trace outputs/agentdojo_smoke_trace.jsonl --out outputs/agentdojo_smoke_prefix_dataset.csv

After changes, report:
- modified files
- whether real wiring is implemented or still blocked
- commands to list suites/tasks
- command for first real smoke run
- what environment variables are required
- what cost guards are implemented