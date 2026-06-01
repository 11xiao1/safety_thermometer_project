# AgentDojo Mini-Batch Validation

## Status

Preflight is ready. The first real guarded AgentDojo mini-batch has not been
executed by Codex.

No provider call was made, and AgentDojo was not run during this validation.

## Dry-Run Preflight

The dry-run command completed successfully:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\run_agentdojo_mini_batch.py --suite workspace --limit 5 --task-start user_task_0 --trace-dir outputs\agentdojo_mini_batch\traces --prefix-dir outputs\agentdojo_mini_batch\prefix --merged-out outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --provider openai-compatible --model gpt-3.5-turbo --max-steps 3 --max-tool-calls 3 --max-output-tokens 512 --temperature 0 --cost-guard --dry-run
```

Dry-run selected tasks:

- `user_task_0`
- `user_task_1`
- `user_task_2`
- `user_task_3`
- `user_task_4`

Dry-run confirmed:

- provider calls disabled
- full benchmark disabled
- AgentDojo source modification disabled
- monkey patching disabled
- cost guards shown: `max_steps=3`, `max_tool_calls=3`,
  `max_output_tokens=512`, `temperature=0`
- output paths shown for per-task traces and prefix datasets

## Expected Real-Run Outputs

The expected real-run files are currently missing because the manual provider
run has not been executed:

- `outputs/agentdojo_mini_batch/run_summary.json`
- `outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl`
- `outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv`

## Offline Validation Status

- Report generation: blocked until real mini-batch outputs exist.
- Split generation: blocked until the merged prefix dataset exists.

## Manual Real-Run Command

Run this manually only after provider environment variables are configured:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\run_agentdojo_mini_batch.py --suite workspace --limit 5 --task-start user_task_0 --trace-dir outputs\agentdojo_mini_batch\traces --prefix-dir outputs\agentdojo_mini_batch\prefix --merged-out outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --provider openai-compatible --model gpt-3.5-turbo --max-steps 3 --max-tool-calls 3 --max-output-tokens 512 --temperature 0 --cost-guard --allow-real-run --allow-provider-call
```

Do not paste API keys or base URLs into this document.

## After Manual Run

Generate the mini-batch report:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\report_agentdojo_mini_batch.py --summary outputs\agentdojo_mini_batch\run_summary.json --trace outputs\agentdojo_mini_batch\merged\workspace_mini_batch_trace.jsonl --prefix outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --out reports\agentdojo_mini_batch_report.md
```

Split the merged prefix dataset by `episode_id`:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\split_prefix_dataset.py --input outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --out-dir outputs\splits --seed 20260601
```

This validation note should then be updated with report and split results. Any
claims should remain mini-batch only: no benchmark-scale claim and no
calibration claim on AgentDojo yet.
