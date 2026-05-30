from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from .schema import TraceEvent


class JsonlTraceLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")


def load_trace_events(path: str | Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TraceEvent.model_validate(json.loads(line)))
            except Exception as exc:  # pragma: no cover
                raise ValueError(f"Invalid JSONL trace at line {line_no}: {exc}") from exc
    return events


def group_by_episode(events: Iterable[TraceEvent]) -> dict[str, list[TraceEvent]]:
    grouped: dict[str, list[TraceEvent]] = {}
    for event in events:
        grouped.setdefault(event.episode_id, []).append(event)
    for episode_events in grouped.values():
        episode_events.sort(key=lambda e: (e.step_id, e.hook_type))
    return grouped
