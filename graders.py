from __future__ import annotations

from typing import Any

from tasks import TaskConfig


SCORE_EPS = 1e-3


def _clamp(value: float, low: float = SCORE_EPS, high: float = 1.0 - SCORE_EPS) -> float:
    return max(low, min(high, value))


def _strict_score(value: float) -> float:
    return _clamp(value, SCORE_EPS, 1.0 - SCORE_EPS)


def grade_episode(task: TaskConfig, metrics: dict[str, Any]) -> float:
    weights = task.score_weights
    score = sum(weights.get(metric_name, 0.0) * _clamp(metrics.get(metric_name, 0.0)) for metric_name in weights)
    return round(_strict_score(score), 4)


def summarize_episode(total_reward: float, state_history: list[dict[str, Any]], terminal_outcome: str) -> dict[str, Any]:
    step_count = max(len(state_history), 1)
    safety_violations = sum(1 for item in state_history if item.get("unsafe", False))
    lab_steps = [item for item in state_history if item.get("action_type") == "request_lab"]
    treatment_steps = [item for item in state_history if item.get("action_type") == "request_treatment"]
    early_window = state_history[: min(3, len(state_history))] or state_history

    detection = max((item.get("detection_credit", 0.0) for item in early_window), default=0.0)
    lab_workup = (
        sum(item.get("lab_score", 0.0) for item in lab_steps) / len(lab_steps)
        if lab_steps
        else 0.0
    )
    treatment = (
        sum(item.get("treatment_score", 0.0) for item in treatment_steps) / len(treatment_steps)
        if treatment_steps
        else 0.0
    )
    first_meaningful_step = next(
        (
            idx
            for idx, item in enumerate(state_history)
            if item.get("detection_credit", 0.0) > 0.0 or item.get("treatment_score", 0.0) > 0.0
        ),
        step_count,
    )
    timeliness = _clamp(1.0 - (first_meaningful_step / step_count))
    stability = sum(item.get("stability_score", 0.0) for item in state_history) / step_count
    safety = _clamp(1.0 - (safety_violations / step_count))
    outcome = 1.0 - SCORE_EPS if terminal_outcome == "survived" else SCORE_EPS
    return {
        "steps": step_count,
        "avg_reward": total_reward / step_count,
        "detection": round(_clamp(detection), 4),
        "lab_workup": round(_clamp(lab_workup), 4),
        "treatment": round(_clamp(treatment), 4),
        "timeliness": round(_clamp(timeliness), 4),
        "stability": round(_clamp(stability), 4),
        "safety": round(_clamp(safety), 4),
        "safety_violation_rate": safety_violations / step_count,
        "outcome": outcome,
    }
