"""Local AgentDojo-like tools executor with Safety Thermometer hooks.

This wrapper mirrors the logical boundary of AgentDojo's ToolsExecutor without
importing AgentDojo. It is intentionally duck-typed so tests can use fake
messages, tool calls, runtimes, and environments.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable, Sequence
from typing import Any

from src.adapters.agentdojo_adapter import AgentDojoTraceAdapter


EMPTY_FUNCTION_NAME = "EMPTY"
MAX_TOOL_CALLS_EXCEEDED_REASON = "max_tool_calls_exceeded"


class MaxToolCallsExceeded(RuntimeError):
    def __init__(self, max_tool_calls: int, tool_calls_seen: int) -> None:
        self.max_tool_calls = max_tool_calls
        self.tool_calls_seen = tool_calls_seen
        super().__init__(
            f"{MAX_TOOL_CALLS_EXCEEDED_REASON}: max_tool_calls={max_tool_calls}, "
            f"tool_calls_seen={tool_calls_seen}"
        )


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_value(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _is_string_list(value: str) -> bool:
    try:
        return isinstance(literal_eval(value), list)
    except (ValueError, SyntaxError):
        return False


def _default_tool_output_formatter(tool_result: Any) -> str:
    if hasattr(tool_result, "model_dump"):
        return str(tool_result.model_dump())
    return str(tool_result)


def _text_content_block(content: str) -> dict[str, str]:
    return {"type": "text", "content": content}


class TraceHookedToolsExecutor:
    def __init__(
        self,
        adapter: AgentDojoTraceAdapter,
        tool_output_formatter: Callable[[Any], str] = _default_tool_output_formatter,
        empty_function_name: str = EMPTY_FUNCTION_NAME,
        max_tool_calls: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.output_formatter = tool_output_formatter
        self.empty_function_name = empty_function_name
        self.max_tool_calls = max_tool_calls
        self.tool_calls_seen = 0

    def _available_tool_names(self, runtime: Any) -> set[str]:
        functions = _value(runtime, "functions", {}) or {}
        names = set()
        for key, tool in functions.items():
            names.add(str(_value(tool, "name", key)))
        return names

    def _normalize_args(self, tool_call: Any) -> dict[str, Any]:
        args = dict(_value(tool_call, "args", {}) or {})
        for arg_key, arg_value in list(args.items()):
            if isinstance(arg_value, str) and _is_string_list(arg_value):
                args[arg_key] = literal_eval(arg_value)
        _set_value(tool_call, "args", args)
        return args

    def _tool_result_message(
        self,
        tool_call: Any,
        formatted_result: str,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "content": [_text_content_block(formatted_result)],
            "tool_call_id": _value(tool_call, "id"),
            "tool_call": tool_call,
            "error": error,
        }

    def _append_error_result(
        self,
        tool_result_messages: list[dict[str, Any]],
        tool_call: Any,
        assistant_message: Any,
        env: Any,
        error: str,
    ) -> None:
        self.adapter.pre_step_hook(tool_call, assistant_message=assistant_message, env_before=env)
        self.adapter.post_step_hook(
            tool_call,
            formatted_observation=error,
            error=error,
            assistant_message=assistant_message,
            env_before=env,
            env_after=env,
        )
        tool_result_messages.append(self._tool_result_message(tool_call, "", error))

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: Sequence[dict[str, Any]] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, Any, Any, list[dict[str, Any]], dict]:
        extra_args = extra_args or {}
        messages_out = list(messages)
        if not messages_out:
            return query, runtime, env, messages_out, extra_args

        assistant_message = messages_out[-1]
        if _value(assistant_message, "role") != "assistant":
            return query, runtime, env, messages_out, extra_args

        tool_calls = _value(assistant_message, "tool_calls", None)
        if not tool_calls:
            return query, runtime, env, messages_out, extra_args

        tool_result_messages = []
        available_tool_names = self._available_tool_names(runtime)
        for tool_call in tool_calls:
            function_name = str(_value(tool_call, "function", ""))
            if function_name == self.empty_function_name:
                self._append_error_result(
                    tool_result_messages,
                    tool_call,
                    assistant_message,
                    env,
                    "Empty function name provided. Provide a valid function name.",
                )
                continue

            if available_tool_names and function_name not in available_tool_names:
                self._append_error_result(
                    tool_result_messages,
                    tool_call,
                    assistant_message,
                    env,
                    f"Invalid tool {function_name} provided.",
                )
                continue

            args = self._normalize_args(tool_call)
            if self.max_tool_calls is not None and self.tool_calls_seen >= self.max_tool_calls:
                raise MaxToolCallsExceeded(
                    max_tool_calls=self.max_tool_calls,
                    tool_calls_seen=self.tool_calls_seen,
                )
            self.tool_calls_seen += 1
            pre_event = self.adapter.pre_step_hook(tool_call, assistant_message=assistant_message, env_before=env)
            tool_result, error = runtime.run_function(env, function_name, args)
            formatted_result = self.output_formatter(tool_result)
            self.adapter.post_step_hook(
                tool_call,
                tool_result=tool_result,
                formatted_observation=error if error else formatted_result,
                error=error,
                assistant_message=assistant_message,
                env_before=env,
                env_after=env,
                step_id=pre_event.step_id,
            )
            tool_result_messages.append(self._tool_result_message(tool_call, formatted_result, error))

        return query, runtime, env, [*messages_out, *tool_result_messages], extra_args
