# AgentDojo Mini-Batch Plan

## Goal

Run a guarded mini batch of AgentDojo `workspace` user tasks to check whether
the traced pipeline and replay flow remain stable beyond the single smoke task.
This is a small engineering validation run, not a benchmark.

## Fixed Run Configuration

- Suite: `workspace`
- Batch size: 5-10 user tasks
- `max_steps`: 3
- `max_tool_calls`: 3
- `max_output_tokens`: 512
- `temperature`: 0
- Provider calls: require explicit `--allow-provider-call`
- Benchmark mode: do not run full AgentDojo benchmark
- Injection tasks: exclude from this mini batch

## Task Selection Strategy

Use utility-only `workspace` user tasks first, with no injection task and no
attack-generated injections. Select tasks by stable numeric task order, not
lexicographic order, so `user_task_2` comes before `user_task_10`.

Recommended first batch:

```text
user_task_0
user_task_1
user_task_2
user_task_3
user_task_4
```

If the first 5 tasks pass the stop conditions, extend once to 10 tasks:

```text
user_task_5
user_task_6
user_task_7
user_task_8
user_task_9
```

Before running the batch, validate task availability with:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\run_agentdojo_smoke_trace.py --list-tasks --suite workspace
```

Do not include tasks selected for known injection behavior until the
utility-only batch can emit and replay clean traces.

## Expected Output Files

Write each task to its own trace and prefix dataset so partial failures do not
overwrite successful runs:

```text
outputs/agentdojo_mini_batch/
  manifest.json
  run_summary.json
  traces/
    workspace_user_task_0_trace.jsonl
    workspace_user_task_1_trace.jsonl
    ...
  prefixes/
    workspace_user_task_0_prefix_dataset.csv
    workspace_user_task_1_prefix_dataset.csv
    ...
  merged/
    workspace_mini_batch_trace.jsonl
    workspace_mini_batch_prefix_dataset.csv
```

`manifest.json` should record non-secret run metadata:

- suite
- task IDs
- provider name
- model name
- max steps
- max tool calls
- max output tokens
- temperature
- output paths
- start/end timestamps

It must not record API keys, base URLs, request payloads, or provider secrets.

## Replay And Merge Strategy

1. Run each selected user task independently with the traced custom pipeline.
2. After each task, validate that the trace JSONL can be loaded with
   `load_trace_events(...)`.
3. Replay each trace independently with `scripts/replay.py`.
4. Merge trace JSONL files by task order into
   `workspace_mini_batch_trace.jsonl`.
5. Merge prefix CSV files by keeping the first header and appending all data
   rows in task order.
6. Verify the merged prefix dataset has:
   - one header row
   - at least one data row per completed task
   - stable column count across all appended rows
   - no duplicate header rows in the middle of the file

If an individual task fails, keep its trace and error metadata but do not merge
an incomplete prefix CSV unless replay succeeded.

## Meaningful Mini-Scale Metrics

These metrics are meaningful for the 5-10 task validation:

- completed task count
- failed task count
- TraceEvent row count per task
- hook type counts per task and merged total
- tool call count per task
- unique tools called
- tasks with zero tool calls
- tasks that hit `max_steps`
- tasks that hit `max_tool_calls`
- trace load success rate
- replay success rate
- prefix dataset row count and column count
- utility/security booleans as post-hoc diagnostics only

Use these metrics to evaluate pipeline health, trace completeness, replay
compatibility, and cost-guard behavior.

## Non-Meaningful Metrics

These metrics should not be claimed from this mini batch:

- benchmark-level utility rate
- benchmark-level security rate
- model ranking or provider comparison
- calibrated AgentDojo risk probability
- calibrated thermometer score validity on AgentDojo
- task-suite generalization
- injection robustness
- attack success rate
- statistical significance

The batch is too small and utility-only for those claims.

## Stop Conditions

Stop the mini batch immediately if any of these occur:

- a provider key, base URL, or other secret would be printed or written
- the runner would execute without `TraceHookedToolsExecutor`
- a task would run without `--allow-provider-call`
- the same task retries more than once after a provider/runtime failure
- any task exceeds `max_steps=3`
- any task exceeds `max_tool_calls=3`
- any response exceeds `max_output_tokens=512` if the provider path supports
  enforcement
- trace JSONL cannot be parsed after a task
- replay fails for two tasks in a row
- merged prefix CSV has inconsistent columns
- unexpected writes occur outside the planned `outputs/agentdojo_mini_batch/`
  tree
- total batch size would exceed 10 tasks

When a stop condition triggers, preserve completed outputs and write a summary
with the blocker. Do not continue into a full benchmark run.

## Next Coding Task

Implement a guarded mini-batch runner with:

- `--suite workspace`
- `--limit`
- `--task-start` or explicit `--tasks`
- `--trace-dir`
- `--prefix-dir`
- `--merged-out`
- `--max-steps 3`
- `--max-tool-calls 3`
- `--max-output-tokens 512`
- `--temperature 0`
- `--allow-real-run`
- `--allow-provider-call`
- dry-run mode that lists selected tasks and output paths without provider
  calls

The runner should reuse the existing traced AgentDojo pipeline construction and
single-task replay path. It must not modify AgentDojo source, monkey patch by
default, run the full benchmark, or touch the Risk Estimator/calibration code.
