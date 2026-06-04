from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.labels.judge_schema import JudgeLabel


def validate_judge_label(payload: dict[str, Any]) -> JudgeLabel:
    return JudgeLabel.model_validate(payload)


def validation_errors(payload: dict[str, Any]) -> list[str]:
    try:
        validate_judge_label(payload)
        return []
    except ValidationError as exc:
        return [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()]
    except ValueError as exc:
        return [str(exc)]


def validate_many_judge_labels(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, payload in enumerate(payloads):
        errors = validation_errors(payload)
        rows.append(
            {
                "index": index,
                "episode_id": payload.get("episode_id"),
                "valid": not errors,
                "errors": errors,
            }
        )
    return {
        "status": "ok" if all(row["valid"] for row in rows) else "invalid",
        "label_count": len(payloads),
        "valid_count": sum(1 for row in rows if row["valid"]),
        "invalid_count": sum(1 for row in rows if not row["valid"]),
        "rows": rows,
    }

