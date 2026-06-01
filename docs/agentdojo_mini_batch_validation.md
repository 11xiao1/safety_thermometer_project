# AgentDojo Mini-Batch Validation

## Status

Offline validation succeeded for the first real guarded AgentDojo mini-batch
outputs now present under `outputs/agentdojo_mini_batch`.

Codex did not call a provider or run AgentDojo during this validation step. It
only read existing mini-batch outputs, generated the offline report, and split
the merged prefix dataset.

## Real-Run Inputs

All expected real-run input files exist:

- `outputs/agentdojo_mini_batch/run_summary.json`
- `outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl`
- `outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv`

Run summary status:

- status: `ok`
- selected tasks: `user_task_0`, `user_task_1`, `user_task_2`,
  `user_task_3`, `user_task_4`
- completed task count: 5
- failed task count: 0
- merged trace rows: 39
- merged prefix rows: 39
- merged prefix columns: 30
- secrets recorded: `false`
- full benchmark run: `false`
- AgentDojo source modified: `false`
- monkey patching: `false`

## Offline Report

Report generation succeeded:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\report_agentdojo_mini_batch.py --summary outputs\agentdojo_mini_batch\run_summary.json --trace outputs\agentdojo_mini_batch\merged\workspace_mini_batch_trace.jsonl --prefix outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --out reports\agentdojo_mini_batch_report.md
```

Output:

- `reports/agentdojo_mini_batch_report.md`

Report highlights:

- TraceEvent rows: 39
- Hook type counts: `pre_step=17`, `post_step=17`, `final=5`
- Tool call count: 17
- Unique tools: `create_calendar_event`, `get_current_day`,
  `get_day_calendar_events`, `search_calendar_events`
- `future_risk_label` distribution: `0=33`, `1=6`
- `oracle_violation` distribution: `0=36`, `1=3`
- utility diagnostics: 4 successful utility tasks, 1 failed utility task
- security diagnostics: 5 successful security checks

## Prefix Split

Episode-level split generation succeeded:

```powershell
F:\Anaconda_envs\envs\safetythermo\python.exe scripts\split_prefix_dataset.py --input outputs\agentdojo_mini_batch\merged\workspace_mini_batch_prefix_dataset.csv --out-dir outputs\splits --seed 20260601
```

Outputs:

- `outputs/splits/agentdojo_train.csv`
- `outputs/splits/agentdojo_val.csv`
- `outputs/splits/agentdojo_test.csv`
- `outputs/splits/split_manifest.json`

Split summary:

- seed: `20260601`
- ratios: train `0.6`, validation `0.2`, test `0.2`
- train episodes: 3
- validation episodes: 1
- test episodes: 1
- train rows: 31
- validation rows: 5
- test rows: 3
- warnings: none

Episode allocation:

- train: `workspace:user_task_4:none:none`,
  `workspace:user_task_2:none:none`, `workspace:user_task_3:none:none`
- validation: `workspace:user_task_0:none:none`
- test: `workspace:user_task_1:none:none`

Label counts:

- train: `future_risk_label 0=25`, `future_risk_label 1=6`
- validation: `future_risk_label 0=5`
- test: `future_risk_label 0=3`

The split is by `episode_id`, not by prefix row. Do not fit calibration on test
data.

## Limitations

- This is a 5-task mini-batch smoke validation only.
- This is not a benchmark-scale result.
- There is no calibration claim on AgentDojo yet.
- Results depend on the selected tasks and model behavior.
- The validation uses existing outputs; it does not establish provider
  reproducibility.

## Next Step

Use the train/validation/test split only for the intended method positions:

- train: future Risk Estimator training
- validation: future calibration and threshold selection
- test: reserved for final Table 3 / Table 4 reporting

The next coding task should be to add a guarded training entrypoint that can
consume `outputs/splits/agentdojo_train.csv` and report validation metrics on
`outputs/splits/agentdojo_val.csv` without touching the test split.
