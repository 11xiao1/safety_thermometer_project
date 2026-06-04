# Run On Linux / AutoDL

This guide is for moving the Safety Thermometer project from Windows to a Linux or AutoDL machine before large AgentHazard prefix generation, judge labeling, or model training.

## Clone Project

```bash
git clone git@github.com:11xiao1/safety_thermometer_project.git
cd safety_thermometer_project
```

If SSH is not configured on the target machine, use the HTTPS clone URL from GitHub.

## Create Conda Environment

```bash
conda create -n safetythermo python=3.11 -y
conda activate safetythermo
python -m pip install --upgrade pip
python -m pip install -r requirements-freeze.txt
```

If `agentdojo==0.1.35` is not available from the configured package index, install the rest of the requirements first and then install AgentDojo using the method that worked on the original machine.

## Clone AgentHazard

AgentHazard is an external benchmark checkout and must not be committed into this repository.

```bash
mkdir -p external
git clone <AGENTHAZARD_REPO_URL> external/AgentHazard
```

Keep this path exactly:

```text
external/AgentHazard
```

The project `.gitignore` ignores `external/AgentHazard/`.

## Large Outputs

Large generated artifacts should stay under:

```text
outputs/
```

Do not commit generated outputs such as:

```text
outputs/**/*.jsonl
outputs/**/*.csv
outputs/**/*.parquet
outputs/**/*.pkl
outputs/**/*.pt
outputs/**/*.bin
```

The project `.gitignore` ignores `outputs/`.

## Rerun AgentHazard Trace Conversion

This conversion is read-only over AgentHazard trace zip archives. It does not run AgentHazard and does not call providers.

```bash
python scripts/agenthazard/convert_agenthazard_traces.py
```

Expected main output:

```text
outputs/agenthazard/agenthazard_claudecode_iflow_trace_events.jsonl
outputs/agenthazard/agenthazard_claudecode_iflow_trace_conversion_summary.json
outputs/agenthazard/agenthazard_claudecode_iflow_trace_conversion_summary.csv
```

`openclaw` is intentionally excluded from the current main conversion path.

## Rerun TraceEvent Audit

```bash
python scripts/agenthazard/audit_agenthazard_trace_events.py
```

Expected outputs:

```text
outputs/agenthazard/agenthazard_trace_event_audit.json
outputs/agenthazard/agenthazard_trace_event_audit.csv
```

## Build AgentHazard Prefix Dataset Preview

```bash
python scripts/agenthazard/build_agenthazard_prefix_dataset.py --max-rows 5000
```

Expected outputs:

```text
outputs/agenthazard/agenthazard_prefix_dataset_preview.csv
outputs/agenthazard/agenthazard_prefix_dataset_preview_manifest.json
```

The preview keeps label columns empty and sets `label_status=missing_pending_judge` when judge labels are not available.

## Portability Check

Run the static portability check:

```bash
python scripts/dev/check_portability.py
```

Expected outputs:

```text
outputs/dev/portability_check_report.json
outputs/dev/portability_check_report.csv
```

## Tests

Run tests manually after moving the repository:

```bash
python -m pytest -q --basetemp .pytest_tmp
```

## Safety Rules

Do not run AgentHazard, AgentDojo, provider calls, judge labeling, training, calibration, or test-split evaluation until the method and data protocol are frozen.
