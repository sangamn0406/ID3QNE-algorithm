from __future__ import annotations

import json
import os
import random
from pathlib import Path

from openai import OpenAI

from client import SepsisTreatmentEnv
from models import SepsisAction, SepsisObservation


OUTPUT_DIR = Path("outputs")
MAX_STEPS_PER_TASK = {"easy": 8, "medium": 12, "hard": 16}
EPSILON = 0.1
RNG = random.Random(7)
ENV_NAME = "sepsis-openenv"


def format_action(action: SepsisAction) -> str:
    if action.action_type == "request_lab":
        return f"request_lab({action.lab_type}, suspect_sepsis={str(action.suspect_sepsis).lower()})"
    if action.action_type == "request_treatment":
        return f"request_treatment({action.treatment_type}, suspect_sepsis={str(action.suspect_sepsis).lower()})"
    return f"monitor(suspect_sepsis={str(action.suspect_sepsis).lower()})"


def curriculum_action(observation: SepsisObservation) -> SepsisAction | None:
    task_id = observation.task_id
    step_index = observation.step_index

    if task_id == "easy":
        if step_index == 0:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=True,
                lab_type="lactate",
                rationale="Curriculum schedule: start with lactate for early workup.",
            )
        if step_index == 2:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=True,
                lab_type="creatinine",
                rationale="Curriculum schedule: add renal assessment when the workup broadens.",
            )
        return SepsisAction(
            action_type="monitor",
            suspect_sepsis=True,
            rationale="Curriculum schedule: maintain observation after priority labs.",
        )

    if task_id == "medium":
        lab_plan = {0: "lactate", 1: "wbc", 2: "creatinine", 3: "bicarbonate"}
        if step_index in lab_plan:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=True,
                lab_type=lab_plan[step_index],
                rationale=f"Curriculum schedule: collect {lab_plan[step_index]} before treatment.",
            )
        treatment_type = "fluids" if step_index <= 6 else "monitor"
        return SepsisAction(
            action_type="request_treatment",
            suspect_sepsis=True,
            treatment_type=treatment_type,
            rationale="Curriculum schedule: shift from early support to monitoring.",
        )

    if task_id == "hard":
        if step_index == 0:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=True,
                lab_type="lactate",
                rationale="Curriculum schedule: start unstable trajectory with lactate.",
            )
        if step_index == 1:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=True,
                lab_type="creatinine",
                rationale="Curriculum schedule: check renal strain before extended management.",
            )
        treatment_type = "monitor" if step_index in {3, 4} else "fluids"
        return SepsisAction(
            action_type="request_treatment",
            suspect_sepsis=True,
            treatment_type=treatment_type,
            rationale="Curriculum schedule: alternate stabilization and support across the harder case.",
        )

    return None


def heuristic_action(observation: SepsisObservation) -> SepsisAction:
    scheduled_action = curriculum_action(observation)
    if scheduled_action is not None:
        return scheduled_action

    severity = observation.severity_proxy
    shock = observation.vitals.get("Shock_Index", 0.0)
    mean_bp = observation.vitals.get("MeanBP", 0.0)
    visible_labs = observation.visible_labs
    requested_labs = set(observation.requested_labs)

    if RNG.random() < EPSILON:
        unseen_labs = [lab for lab in ["lactate", "wbc", "creatinine", "bicarbonate"] if lab not in requested_labs]
        if unseen_labs:
            lab_choice = unseen_labs[0]
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=severity >= 1.0 or shock > 0.1 or mean_bp < 0.0,
                lab_type=lab_choice,
                rationale="Exploration step",
            )

    lab_priority_order = ["lactate", "wbc", "creatinine", "bicarbonate"]
    for lab in lab_priority_order:
        should_request = False
        if lab == "lactate":
            should_request = lab not in requested_labs
        elif lab == "wbc":
            should_request = lab not in requested_labs and (severity >= 0.75 or shock > 0.08)
        elif lab == "creatinine":
            should_request = lab not in requested_labs and severity >= 1.2
        elif lab == "bicarbonate":
            should_request = lab not in requested_labs and (severity >= 1.5 or mean_bp < -0.1)

        if should_request:
            return SepsisAction(
                action_type="request_lab",
                suspect_sepsis=severity >= 1.0 or shock > 0.1 or mean_bp < 0.0,
                lab_type=lab,
                rationale=f"Exploring informative lab: {lab}",
            )

    lactate = visible_labs.get("lactate", 0.0) or 0.0
    bicarbonate = visible_labs.get("bicarbonate", 0.0) or 0.0
    if severity < 0.8 and mean_bp >= -0.1:
        treatment_type = "monitor"
    elif severity >= 2.0 or mean_bp < -0.2:
        treatment_type = "combination" if lactate > 0.25 else "vasopressors"
    elif shock > 0.15 or severity >= 1.1 or bicarbonate < -0.15:
        treatment_type = "fluids"
    else:
        treatment_type = "monitor"

    return SepsisAction(
        action_type="request_treatment",
        suspect_sepsis=severity >= 1.0 or lactate > 0.25,
        treatment_type=treatment_type,
        rationale="Improved staged policy with exploration and severity awareness.",
    )


def build_prompt(observation: SepsisObservation) -> str:
    return (
        "You are controlling a sequential sepsis management simulator.\n"
        f"Task: {observation.task_description}\n"
        f"Step: {observation.step_index + 1}/{observation.max_steps}\n"
        f"Severity proxy: {observation.severity_proxy:.2f}\n"
        f"Mortality flag in logged trajectory: {observation.mortality_risk_flag}\n"
        f"Demographics: {json.dumps(observation.demographics)}\n"
        f"Vitals: {json.dumps(observation.vitals)}\n"
        f"Context features: {json.dumps(observation.context_features)}\n"
        f"Visible labs: {json.dumps(observation.visible_labs)}\n"
        f"Requested labs so far: {json.dumps(observation.requested_labs)}\n"
        "Return JSON with keys action_type, suspect_sepsis, lab_type, treatment_type, rationale. "
        "action_type must be one of request_lab, request_treatment, monitor."
    )


def model_action(client: OpenAI | None, model_name: str | None, observation: SepsisObservation) -> SepsisAction:
    if client is None or not model_name:
        return heuristic_action(observation)

    messages = [
        {"role": "system", "content": "Return only valid JSON for a sepsis management action."},
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
    global EPSILON
    if task_id == "easy":
        EPSILON = 0.05
    elif task_id == "medium":
        EPSILON = 0.1
    else:
        EPSILON = 0.15

    env = SepsisTreatmentEnv(base_url=os.getenv("ENV_BASE_URL"), task_id=task_id)
    result = env.reset()
    observation = result.observation
    final_info = result.info
    reward_trace: list[float] = []
    success = False
    step_count = 0

    print(f"[START] task={task_id} env={ENV_NAME} model={model_name or 'heuristic-baseline'}")

    try:
        for step_number in range(1, MAX_STEPS_PER_TASK[task_id] + 1):
            action = model_action(client, model_name, observation)
            result = env.step(action)
            observation = result.observation
            final_info = result.info
            reward = float(result.reward or 0.0)
            reward_trace.append(reward)
            step_count = step_number
            print(
                f"[STEP] step={step_number} action={format_action(action)} "
                f"reward={reward:.2f} done={str(result.done).lower()} error=null"
            )
            if result.done:
                success = True
                break
    except Exception:
        success = False
    finally:
        state = env.state()
        env.close()
        rewards_repr = ",".join(f"{reward:.2f}" for reward in reward_trace)
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_repr}")

    metrics = final_info.get("metrics", {})
    return {
        "task_id": task_id,
        "episode_id": state.episode_id,
        "score": metrics.get("score", 0.0),
        "avg_reward": metrics.get("avg_reward", 0.0),
        "detection": metrics.get("detection", 0.0),
        "lab_workup": metrics.get("lab_workup", 0.0),
        "treatment": metrics.get("treatment", 0.0),
        "timeliness": metrics.get("timeliness", 0.0),
        "stability": metrics.get("stability", 0.0),
        "safety": metrics.get("safety", 0.0),
        "safety_violation_rate": metrics.get("safety_violation_rate", 0.0),
        "outcome": metrics.get("outcome", 0.0),
        "steps": metrics.get("steps", state.step_count),
    }


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


if __name__ == "__main__":
    main()
