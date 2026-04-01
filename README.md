---
title: Sepsis OpenEnv
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
tags:
  - openenv
  - healthcare
  - offline-rl
  - sepsis
---

# Sepsis OpenEnv

Sepsis OpenEnv is a real-world sequential sepsis management environment for the OpenEnv hackathon workflow. It exposes a standard `reset()` / `step()` / `state()` loop and evaluates how well an agent gathers information, chooses treatment, and manages a logged ICU trajectory under partial observability.

The environment is designed to satisfy the Round 1 requirements:

- real-world task: ICU sepsis workup and treatment decisions
- typed models for action, observation, and state
- 3 graded tasks: `easy`, `medium`, `hard`
- dense rewards with safety penalties and partial-progress signal
- reproducible root-level `inference.py`
- Dockerized server for local and Hugging Face deployment

## What The Environment Simulates

At each step, the agent can:

- request one lab from a clinically meaningful set
- request one treatment plan from a sepsis-management action set
- optionally mark the current state as suspected sepsis

The environment advances along a logged patient trajectory and rewards the agent for:

- detecting likely sepsis early
- requesting informative labs instead of repeatedly querying low-value tests
- selecting treatment plans that fit the hidden severity pattern in the logged stay
- avoiding obviously unsafe escalation or under-treatment

This is an offline environment built from a compact processed bundle derived from the MIMIC-III demo cohort. It is inspired by the WD3QNE sepsis-treatment paper, but the environment is purpose-built for OpenEnv evaluation rather than paper reproduction.

## Tasks

Task definitions live in `tasks.py`.

- `easy`: early sepsis workup from partial bedside data with an emphasis on timely lab selection
- `medium`: diagnosis plus early treatment initiation after iterative lab requests
- `hard`: full sepsis management across longer unstable trajectories with stabilization and outcome pressure

Each task has a deterministic grader in `graders.py` that returns a score in `[0.0, 1.0]`.

## Action Space

Defined in `models.py`.

- `action_type`: `request_lab`, `request_treatment`, or `monitor`
- `suspect_sepsis`: boolean detection signal
- `lab_type`: one of `lactate`, `wbc`, `creatinine`, `bicarbonate`, `platelets`, `bilirubin`
- `treatment_type`: one of `monitor`, `fluids`, `vasopressors`, `combination`

## Observation Space

Defined in `models.py`.

Each observation contains:

- task id and task description
- current patient trajectory id
- current step and max steps
- severity proxy
- mortality flag from the logged stay
- demographics and always-visible vitals
- visible non-lab context features
- only the labs explicitly requested so far
- current cumulative reward and last reward

Hidden logged treatment choices and unrevealed labs are intentionally not exposed in observations.

## Reward Design

The reward function is dense, not purely terminal.

Per step:

- positive signal for early sepsis suspicion on high-risk states
- reward for requesting priority labs that fit the current presentation
- reward for selecting treatment plans that match the hidden severity pattern
- progress bonus when the next logged state becomes less severe
- novelty bonus for new state-action exploration
- penalties for duplicate labs, repeated low-value actions, unsafe escalation, or obvious under-treatment

At the end of the episode:

- bonus for survival trajectories
- penalty for death trajectories

## Core Files

- `openenv.yaml`: OpenEnv metadata
- `models.py`: typed action / observation / state models
- `tasks.py`: task catalog
- `graders.py`: deterministic graders
- `client.py`: client wrapper
- `server/app.py`: FastAPI app and server entrypoint
- `server/sepsis_environment.py`: environment implementation
- `inference.py`: baseline runner
- `validate_local.py`: local smoke checks
- `prepare_submission.py`: creates a clean submission bundle

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run local validation:

```bash
.venv\Scripts\python.exe validate_local.py
```

Run the official OpenEnv validator:

```bash
.venv\Scripts\openenv.exe validate
```

Start the environment server locally:

```bash
.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Quick checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
```

## Baseline Inference

The required root-level baseline script is `inference.py`.

Run locally:

```bash
.venv\Scripts\python.exe inference.py
```

The script:

- writes reproducible scores to `outputs/baseline_scores.json`
- emits OpenEnv-style `[START]`, `[STEP]`, and `[END]` lines to stdout
- uses the OpenAI client if `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set
- otherwise falls back to a deterministic staged baseline policy

Current deterministic baseline scores from the local run:

- `easy`: `1.0`
- `medium`: `1.0`
- `hard`: `0.96`
- mean score: `0.9867`

## Docker

Build:

```bash
docker build -t sepsis-openenv .
```

Run:

```bash
docker run -p 8000:8000 sepsis-openenv
```

The container exposes a working `/health` endpoint and responds to `/reset`.

## Submission Bundle

To prepare a clean hackathon-ready bundle:

```bash
.venv\Scripts\python.exe prepare_submission.py
```

This creates `submission_bundle/` with only the files needed for the environment runtime and submission packaging.

## Runtime Assets

The runtime uses the preprocessed assets in:

- `env_data/processed_demo_dataset.pkl`
- `env_data/selected_features.json`

This keeps the environment lightweight enough for the hackathon resource limits.

## Validation Status

The following checks have been run locally:

- `python validate_local.py`: passed
- `python inference.py`: passed
- `openenv validate`: passed
- `docker build -t sepsis-openenv .`: passed
- `docker run -p 8000:8000 sepsis-openenv`: passed
- `/health` and `/metadata`: passed

## Inspiration

Wu, X., Li, R., He, Z. et al. *A value-based deep reinforcement learning model with human expertise in optimal treatment of sepsis.* npj Digital Medicine 6, 15 (2023). https://doi.org/10.1038/s41746-023-00755-5
