from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


HookType = Literal["pre_step", "post_step", "checkpoint", "final"]


class TraceEvent(BaseModel):
    """One observable event in an agent trajectory."""

    episode_id: str
    step_id: int
    hook_type: HookType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    user_instruction: str = ""
    plan_summary: str | None = None
    proposed_tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    state_delta: dict[str, Any] = Field(default_factory=dict)
    self_check: dict[str, Any] = Field(default_factory=dict)

    # Optional benchmark/judge metadata. These should not be used as model
    # inputs during online monitoring.
    t_risk: int | None = None
    risk_event: bool | None = None
    risk_type: str | None = None
    severity: float | None = None
    notes: str | None = None

    def text_blob(self) -> str:
        parts = [
            self.user_instruction or "",
            self.plan_summary or "",
            self.proposed_tool or "",
            str(self.tool_args or {}),
            self.observation or "",
            str(self.state_delta or {}),
            str(self.self_check or {}),
        ]
        return "\n".join(parts).lower()
