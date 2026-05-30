"""AgentDojo adapter placeholder.

Do not implement this before the toy trace loop is working.

Goal:
- Locate AgentDojo's action proposal and tool execution boundary.
- Emit TraceEvent at `pre_step` before tool execution.
- Emit TraceEvent at `post_step` after observation/state delta.
- Save JSONL traces in the same schema as toy data.
"""

from __future__ import annotations

from src.monitor.schema import TraceEvent
from src.monitor.logger import JsonlTraceLogger


class AgentDojoMonitorAdapter:
    def __init__(self, trace_path: str):
        self.logger = JsonlTraceLogger(trace_path)

    def pre_step_hook(self, event: TraceEvent) -> None:
        self.logger.log(event)

    def post_step_hook(self, event: TraceEvent) -> None:
        self.logger.log(event)
