from __future__ import annotations

import os

from fastapi import FastAPI
import uvicorn

from models import SepsisAction, SepsisObservation, SepsisState
from openenv_compat import OPENENV_AVAILABLE, create_app
from server.sepsis_environment import SepsisTreatmentEnvironment


if OPENENV_AVAILABLE and create_app is not None:
    app = create_app(SepsisTreatmentEnvironment, SepsisAction, SepsisObservation, env_name="sepsis-openenv")
else:
    environment = SepsisTreatmentEnvironment()
    app = FastAPI(title="Sepsis OpenEnv", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict:
        return environment.metadata()

    @app.get("/schema")
    def schema() -> dict:
        return {
            "action_schema": SepsisAction.model_json_schema(),
            "observation_schema": SepsisObservation.model_json_schema(),
            "state_schema": SepsisState.model_json_schema(),
        }

    @app.post("/reset")
    def reset(payload: dict | None = None) -> dict:
        task_id = None
        if payload:
            task_id = payload.get("task_id")
        observation = environment.reset(task_id=task_id)
        return {
            "observation": observation.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {
                "tasks": environment.available_tasks(),
                "metrics": environment.current_metrics(),
            },
        }

    @app.post("/step")
    def step(payload: dict) -> dict:
        action = SepsisAction(**payload)
        observation = environment.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": observation.reward,
            "done": observation.done,
            "info": {
                "metrics": environment.current_metrics(),
            },
        }

    @app.get("/state")
    def state() -> dict:
        return environment.state.model_dump()


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
