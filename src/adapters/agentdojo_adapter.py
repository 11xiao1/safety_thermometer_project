"""Minimal AgentDojo trace adapter skeleton.

This module intentionally avoids importing AgentDojo. The adapter accepts
AgentDojo-like objects by duck typing so tests can use simple fakes and future
integration can pass real FunctionCall/message objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.monitor.logger import JsonlTraceLogger
from src.monitor.schema import TraceEvent


@dataclass
class AgentDojoEpisodeContext:
    episode_id: str
    suite_name: str
    user_task_id: str
    injection_task_id: str | None
    attack_type: str | None
    prompt: str
    step_id: int = 0


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _assistant_text(assistant_message: Any) -> str:
    return _content_to_text(_value(assistant_message, "content", ""))


def _tool_call_metadata(tool_call: Any) -> dict[str, Any]:
    return {
        "tool_call_id": _value(tool_call, "id"),
        "tool_function": _value(tool_call, "function"),
    }


class AgentDojoTraceAdapter:
    def __init__(self, trace_path: str | Path):
        self.logger = JsonlTraceLogger(trace_path)
        self.context: AgentDojoEpisodeContext | None = None

    def start_episode(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None = None,
        attack_type: str | None = None,
        prompt: str = "",
        episode_id: str | None = None,
    ) -> str:
        safe_attack = attack_type or "none"
        safe_injection = injection_task_id or "none"
        episode_id = episode_id or f"{suite_name}:{user_task_id}:{safe_attack}:{safe_injection}"
        self.context = AgentDojoEpisodeContext(
            episode_id=episode_id,
            suite_name=suite_name,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            attack_type=attack_type,
            prompt=prompt,
        )
        return episode_id

    def _require_context(self) -> AgentDojoEpisodeContext:
        if self.context is None:
            raise RuntimeError("start_episode(...) must be called before emitting hooks.")
        return self.context

    def _notes(self, extra: dict[str, Any] | None = None) -> str:
        ctx = self._require_context()
        payload: dict[str, Any] = {
            "suite_name": ctx.suite_name,
            "user_task_id": ctx.user_task_id,
            "injection_task_id": ctx.injection_task_id,
            "attack_type": ctx.attack_type,
        }
        if extra:
            payload.update(extra)
        return json.dumps(_jsonable(payload), sort_keys=True)

    def pre_step_hook(
        self,
        tool_call: Any,
        assistant_message: Any | None = None,
        env_before: Any | None = None,
        step_id: int | None = None,
    ) -> TraceEvent:
        ctx = self._require_context()
        ctx.step_id = step_id if step_id is not None else ctx.step_id + 1
        event = TraceEvent(
            episode_id=ctx.episode_id,
            step_id=ctx.step_id,
            hook_type="pre_step",
            user_instruction=ctx.prompt,
            plan_summary=_assistant_text(assistant_message),
            proposed_tool=str(_value(tool_call, "function", "")),
            tool_args=dict(_value(tool_call, "args", {}) or {}),
            observation=None,
            state_delta={},
            self_check={},
            notes=self._notes(_tool_call_metadata(tool_call)),
        )
        self.logger.log(event)
        return event

    def post_step_hook(
        self,
        tool_call: Any,
        tool_result: Any | None = None,
        formatted_observation: str | None = None,
        error: str | None = None,
        assistant_message: Any | None = None,
        env_before: Any | None = None,
        env_after: Any | None = None,
        step_id: int | None = None,
    ) -> TraceEvent:
        ctx = self._require_context()
        event_step_id = step_id if step_id is not None else ctx.step_id
        observation = formatted_observation
        if observation is None:
            observation = error if error is not None else _content_to_text(_jsonable(tool_result))
        event = TraceEvent(
            episode_id=ctx.episode_id,
            step_id=event_step_id,
            hook_type="post_step",
            user_instruction=ctx.prompt,
            plan_summary=_assistant_text(assistant_message),
            proposed_tool=str(_value(tool_call, "function", "")),
            tool_args=dict(_value(tool_call, "args", {}) or {}),
            observation=observation,
            state_delta={},
            self_check={},
            notes=self._notes({
                **_tool_call_metadata(tool_call),
                "error": error,
                "raw_result_type": type(tool_result).__name__ if tool_result is not None else None,
            }),
        )
        self.logger.log(event)
        return event

    def final_hook(
        self,
        model_output: Any | None = None,
        utility: bool | None = None,
        security: bool | None = None,
        step_id: int | None = None,
    ) -> TraceEvent:
        ctx = self._require_context()
        event = TraceEvent(
            episode_id=ctx.episode_id,
            step_id=step_id if step_id is not None else ctx.step_id + 1,
            hook_type="final",
            user_instruction=ctx.prompt,
            observation=_content_to_text(_jsonable(model_output)),
            state_delta={},
            self_check={},
            notes=self._notes({"utility": utility, "security": security}),
        )
        self.logger.log(event)
        return event


class AgentDojoMonitorAdapter(AgentDojoTraceAdapter):
    """Backward-compatible alias for the earlier placeholder class."""

