from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from client import SepsisTreatmentEnv
from models import SepsisAction, SepsisObservation


OUTPUT_DIR = Path("outputs")
MAX_STEPS_PER_TASK = {"easy": 8, "medium": 12, "hard": 16}


def heuristic_action(observation: SepsisObservation) -> SepsisAction:
    severity = observation.severity_proxy
    features = observation.features
    shock = features.get("Shock_Index", 0.0)
    lactate = features.get("Arterial_lactate", 0.0)
    pressure = features.get("MeanBP", 0.0)

    if severity < 0.75:
        fluid_bin = 1 if shock > 0.0 else 0
        pressor_bin = 0
    elif severity < 1.5:
        fluid_bin = 2 if shock > 0.25 else 1
        pressor_bin = 1 if pressure < 0.0 else 0
    elif severity < 2.5:
        fluid_bin = 2 if lactate < 0.5 else 3
        pressor_bin = 2 if pressure < 0.0 else 1
    else:
        fluid_bin = 3
        pressor_bin = 3 if pressure < 0.0 else 2

    return SepsisAction(
        fluid_bin=min(4, max(0, int(fluid_bin))),
        pressor_bin=min(4, max(0, int(pressor_bin))),
        rationale="Deterministic severity-based baseline.",
    )


def build_prompt(observation: SepsisObservation) -> str:
    return (
        "You are controlling a sepsis treatment simulator.\n"
        f"Task: {observation.task_description}\n"
        f"Step: {observation.step_index + 1}/{observation.max_steps}\n"
        f"Severity proxy: {observation.severity_proxy:.2f}\n"
        f"Mortality flag in logged trajectory: {observation.mortality_risk_flag}\n"
        f"Features: {json.dumps(observation.features)}\n"
        "Return JSON with keys fluid_bin, pressor_bin, rationale. "
        "Bins must be integers from 0 to 4."
    )


def model_action(client: OpenAI | None, model_name: str | None, observation: SepsisObservation) -> SepsisAction:
    if client is None or not model_name:
        return heuristic_action(observation)

    messages = [
        {"role": "system", "content": "Return only valid JSON for a sepsis treatment action."},
        {"role": "user", "content": build_prompt(observation)},
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        content = completion.choices[0].message.content or ""
        payload = json.loads(content)
        return SepsisAction(**payload)
    except Exception:
        return heuristic_action(observation)


def run_task(task_id: str, client: OpenAI | None, model_name: str | None) -> dict:
    env = SepsisTreatmentEnv(base_url=os.getenv("ENV_BASE_URL"), task_id=task_id)
    final_info: dict = {}
    try:
        result = env.reset()
        observation = result.observation
        final_info = getattr(result, "info", {}) or {}

        for _ in range(MAX_STEPS_PER_TASK[task_id]):
            action = model_action(client, model_name, observation)
            result = env.step(action)
            observation = result.observation
            final_info = getattr(result, "info", {}) or {}
            if result.done:
                break

        state = env.state()
        metrics = final_info.get("metrics", {})
        return {
            "task_id": task_id,
            "episode_id": state.episode_id,
            "score": metrics.get("score", 0.0),
            "avg_reward": metrics.get("avg_reward", 0.0),
            "agreement_rate": metrics.get("agreement_rate", 0.0),
            "safety_violation_rate": metrics.get("safety_violation_rate", 0.0),
            "terminal_success": metrics.get("terminal_success", 0.0),
            "steps": metrics.get("steps", state.step_count),
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "episode_id": "failed",
            "score": 0.0,
            "avg_reward": 0.0,
            "agreement_rate": 0.0,
            "safety_violation_rate": 1.0,
            "terminal_success": 0.0,
            "steps": 0,
            "error": str(exc),
        }
    finally:
        env.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    llm_client = None
    if api_base_url and model_name and hf_token:
        llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)

    results = [run_task(task_id, llm_client, model_name) for task_id in ["easy", "medium", "hard"]]
    summary = {
        "results": results,
        "mean_score": round(sum(item["score"] for item in results) / len(results), 4),
        "model_name": model_name or "heuristic-baseline",
    }
    output_path = OUTPUT_DIR / "baseline_scores.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
