# AgentDojo Adapter Notes

This note designs how to convert AgentDojo benchmark execution events into the
project `TraceEvent` schema. It is design-only: do not modify AgentDojo source,
do not change the toy replay pipeline, and do not implement full integration
yet.

## Local Package Inspected

Installed package path:

`F:\Anaconda_envs\envs\safetythermo\Lib\site-packages\agentdojo`

Key files inspected:

- `agent_pipeline/agent_pipeline.py`
- `agent_pipeline/tool_execution.py`
- `agent_pipeline/base_pipeline_element.py`
- `agent_pipeline/llms/openai_llm.py`
- `agent_pipeline/llms/local_llm.py`
- `functions_runtime.py`
- `task_suite/task_suite.py`
- `benchmark.py`
- `logging.py`
- `types.py`

## AgentDojo Execution Boundaries

### 1. Where AgentDojo proposes or selects an action/tool call

Tool calls are selected by LLM pipeline elements that append an assistant
message with `tool_calls`.

Observed locations:

- `agent_pipeline/llms/openai_llm.py`
  - `OpenAILLM.query(...)`
  - converts provider output into `ChatAssistantMessage`
  - appends that message to `messages`
  - tool calls live at `messages[-1]["tool_calls"]`
- `agent_pipeline/llms/local_llm.py`
  - `LocalLLM.query(...)`
  - parses `<function=...>{...}</function>` output into `FunctionCall`
  - appends `ChatAssistantMessage(..., tool_calls=tool_calls)`
- Other LLM providers follow the same `BasePipelineElement.query(...)`
  contract and should also return assistant messages with optional
  `tool_calls`.

The adapter should treat the last assistant message containing `tool_calls` as
the action proposal boundary.

### 2. Where tool calls are executed

Tool calls are executed in:

- `agent_pipeline/tool_execution.py`
  - `ToolsExecutor.query(...)`
  - iterates over `messages[-1]["tool_calls"]`
  - validates empty or unknown tool names
  - normalizes stringified list args
  - calls `runtime.run_function(env, tool_call.function, tool_call.args)`
- `functions_runtime.py`
  - `FunctionsRuntime.run_function(...)`
  - validates args against the tool schema
  - extracts environment dependencies
  - invokes the registered `Function`
  - returns `(tool_result, error)`

The cleanest pre/post boundary is around each individual call to
`runtime.run_function(...)` inside `ToolsExecutor.query(...)`.

### 3. Where observations are returned

Observations are returned as tool-result chat messages in:

- `agent_pipeline/tool_execution.py`
  - `ToolsExecutor.query(...)`
  - formats the raw tool result with `tool_result_to_str(...)`
  - appends `ChatToolResultMessage(...)` to `messages`

Relevant message fields:

- `role="tool"`
- `content=[{"type": "text", "content": formatted_tool_call_result}]`
- `tool_call_id`
- `tool_call`
- `error`

The adapter should map this tool message to the post-step `TraceEvent`
`observation` field. If `error` is not `None`, include the error in
`observation` and/or `notes`.

### 4. Where benchmark episodes terminate

Benchmark task execution terminates in:

- `task_suite/task_suite.py`
  - `TaskSuite.run_task_with_pipeline(...)`
  - loads and initializes the environment
  - calls `agent_pipeline.query(...)`
  - retries up to 3 times if `model_output` is `None`
  - computes `functions_stack_trace_from_messages(messages)`
  - evaluates utility and security
  - returns `(utility, security)`

Suite-level benchmark loops live in:

- `benchmark.py`
  - `run_task_without_injection_tasks(...)`
  - `run_task_with_injection_tasks(...)`
  - `benchmark_suite_without_injections(...)`
  - `benchmark_suite_with_injections(...)`

These functions use `TraceLogger` to capture AgentDojo's native task result
JSON. For Safety Thermometer traces, the adapter should emit an optional
`hook_type="final"` `TraceEvent` at this boundary with benchmark metadata in
`notes`, not as model input.

### 5. Where benchmark results are logged

AgentDojo logs benchmark results through:

- `logging.py`
  - `TraceLogger`
  - `TraceLogger.log(messages)` stores current chat messages
  - `TraceLogger.set_contextarg("utility", utility)`
  - `TraceLogger.set_contextarg("security", security)`
  - `TraceLogger.save()` writes JSON under:
    `logdir / pipeline_name / suite_name / user_task_id / attack_type / injection_task_id.json`
- `benchmark.py`
  - wraps task execution in `TraceLogger(...)`
  - loads cached results with `load_task_results(...)`

This logging is useful for cross-checking but should not be the primary source
of prefix events, because it sees messages after they have already been
aggregated. Safety Thermometer should emit JSONL `TraceEvent`s during execution
at pre-step and post-step time.

## TraceEvent Mapping

For each AgentDojo `FunctionCall`:

### pre_step_hook(event)

Emit before `runtime.run_function(...)` is called.

Suggested mapping:

| TraceEvent field | AgentDojo source |
|---|---|
| `episode_id` | Stable ID derived from `suite_name`, `user_task_id`, `attack_type`, `injection_task_id` |
| `step_id` | Monotonic count of tool calls within the episode |
| `hook_type` | `"pre_step"` |
| `user_instruction` | `user_task.PROMPT` or injection task `GOAL` for injection-only runs |
| `plan_summary` | Assistant message text content, if present |
| `proposed_tool` | `tool_call.function` |
| `tool_args` | `dict(tool_call.args)` |
| `observation` | `None` |
| `state_delta` | `{}` initially; optionally filled later by environment diff |
| `self_check` | `{}` unless a later defense/model provides self-check metadata |
| `notes` | AgentDojo metadata: suite, user task, injection task, attack, tool_call_id |

### post_step_hook(event)

Emit after `runtime.run_function(...)` returns and the tool message is built.

Suggested mapping:

| TraceEvent field | AgentDojo source |
|---|---|
| `episode_id` | Same as matching pre-step event |
| `step_id` | Same tool-call counter as matching pre-step event |
| `hook_type` | `"post_step"` |
| `user_instruction` | Same prompt or blank |
| `plan_summary` | Optional assistant text from proposing message |
| `proposed_tool` | `tool_call.function` |
| `tool_args` | Final normalized `tool_call.args` |
| `observation` | Formatted tool result, or error string |
| `state_delta` | Best-effort environment diff before/after tool call |
| `self_check` | `{}` initially |
| `notes` | tool_call_id, error, raw result type, AgentDojo task metadata |

### final event

At `TaskSuite.run_task_with_pipeline(...)` completion, optionally emit:

| TraceEvent field | AgentDojo source |
|---|---|
| `episode_id` | Same episode ID |
| `step_id` | Last step ID, or last step ID + 1 |
| `hook_type` | `"final"` |
| `observation` | Final model output text |
| `risk_event` | Derived later from benchmark security/utility policy, not online input |
| `notes` | `utility`, `security`, attack metadata, path to native AgentDojo log |

Do not use `utility` or `security` as online model inputs. They are post-hoc
benchmark labels.

## Proposed Integration Points

### Preferred: wrapper pipeline element around tool execution

Create a local wrapper equivalent to `ToolsExecutor` that delegates to the same
runtime calls but emits Safety Thermometer hooks around each call.

Conceptual interface:

```python
class TraceHookedToolsExecutor(BasePipelineElement):
    def __init__(self, trace_logger, episode_context, tool_output_formatter):
        ...

    def query(self, query, runtime, env, messages, extra_args):
        # If latest assistant message has tool_calls:
        #   for each tool_call:
        #       emit pre_step TraceEvent
        #       run runtime.run_function(...)
        #       build ChatToolResultMessage exactly like AgentDojo
        #       emit post_step TraceEvent
        # return updated messages
```

Then build a custom local `AgentPipeline` using the same components as
`AgentPipeline.from_config(...)`, but replacing:

```python
ToolsExecutor(...)
```

with:

```python
TraceHookedToolsExecutor(...)
```

This avoids editing AgentDojo package files and avoids global monkey patches.

### Alternative: wrapper around `FunctionsRuntime`

AgentDojo `TaskSuite.run_task_with_pipeline(...)` accepts `runtime_class`.
In principle, a subclass/wrapper of `FunctionsRuntime` could override
`run_function(...)` and emit pre/post events there.

Pros:

- Uses an existing extension point: `runtime_class`.
- No need to rebuild `AgentPipeline.from_config(...)`.

Cons:

- `run_function(...)` sees function name and args, but not the full assistant
  message, `tool_call_id`, prompt, or easy episode metadata unless injected
  externally.
- Nested function calls also go through `run_function(...)`, so step accounting
  must distinguish top-level tool calls from nested calls.
- It cannot directly capture the final formatted observation string generated
  by `ToolsExecutor.output_formatter`.

This is viable for low-level execution logging, but less ideal for prefix
dataset quality.

### Avoid by default: monkey patching AgentDojo package functions

Monkey patching `ToolsExecutor.query` or `FunctionsRuntime.run_function` would
capture the right boundary but is fragile:

- global side effects across benchmark runs
- harder tests and cleanup
- version-sensitive behavior
- risk of silently diverging from AgentDojo's native execution semantics

Use monkey patching only as a temporary exploratory probe, not as the planned
adapter.

### Custom defense/hook mechanism

AgentDojo defenses are assembled in `AgentPipeline.from_config(...)` by adding
pipeline elements before/inside the tools loop. A custom defense-like pipeline
element could observe messages before or after tools execute, but it would not
wrap each individual tool call unless it replaces or wraps `ToolsExecutor`.

Recommendation: use a custom pipeline wrapper rather than registering this as a
defense. The Safety Thermometer adapter is instrumentation, not a defense.

## Minimal Adapter Interface

The local adapter should keep its public surface small:

```python
class AgentDojoTraceAdapter:
    def __init__(self, trace_path: str):
        ...

    def start_episode(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None,
        attack_type: str | None,
        prompt: str,
    ) -> str:
        ...

    def pre_step_hook(
        self,
        episode_id: str,
        step_id: int,
        tool_call,
        assistant_message,
        env_before,
    ) -> None:
        ...

    def post_step_hook(
        self,
        episode_id: str,
        step_id: int,
        tool_call,
        tool_result,
        formatted_observation: str,
        error: str | None,
        env_before,
        env_after,
    ) -> None:
        ...

    def final_hook(
        self,
        episode_id: str,
        step_id: int,
        model_output,
        utility: bool | None,
        security: bool | None,
    ) -> None:
        ...
```

The adapter should write JSONL using the project `JsonlTraceLogger` and
`TraceEvent` schema.

## State Delta Design

Initial implementation can set `state_delta={}` to avoid overfitting to one
AgentDojo suite.

Later, add a best-effort diff:

- capture `env_before = env.model_copy(deep=True)` before tool execution
- capture `env_after = env.model_copy(deep=True)` after tool execution
- compare `model_dump()` outputs
- store compact keys changed, not full environment snapshots

Do not store large private environment dumps in `state_delta` until privacy and
size policy is explicit.

## Risks and Unknowns

- Multiple tool calls can appear in one assistant message. The adapter should
  emit one pre/post pair per `FunctionCall`, preserving order.
- `step_id` should count tool calls, not AgentDojo outer loop iterations.
- Some provider adapters may emit assistant text plus tool calls; preserve text
  as `plan_summary`.
- Some tools mutate `env` through dependencies. Environment diffing must copy
  before execution; otherwise mutations are hard to recover.
- Nested `FunctionCall` arguments can cause nested runtime execution. Decide
  whether to log nested calls as internal details or skip them in the first
  adapter.
- AgentDojo native `TraceLogger` writes full conversation logs after each
  update, but it is not a pre/post hook source.
- `utility` and `security` are final benchmark judgments, not online monitor
  inputs.
- The exact severity mapping from AgentDojo task outcomes to
  `future_severity` is not defined yet. Keep severity blank initially.
- Confirmation semantics are domain-specific; AgentDojo tasks may not encode
  explicit confirmation states directly.

## Recommendation

Use a local wrapper around `ToolsExecutor` and a custom pipeline construction
function. Do not patch AgentDojo source. Do not monkey patch by default. Do not
model this as an AgentDojo defense unless later experiments require policy
intervention.

Recommended path:

1. Build `TraceHookedToolsExecutor` locally in this repository.
2. Build a helper that mirrors `AgentPipeline.from_config(...)` but injects the
   hooked executor.
3. Run one tiny suite/task and emit JSONL `TraceEvent`s.
4. Replay the JSONL through the existing toy-compatible prefix pipeline.
5. Only after traces are stable, define AgentDojo-specific label and severity
   mapping.
