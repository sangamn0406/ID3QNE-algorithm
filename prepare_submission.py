from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "submission_bundle"

FILES_TO_COPY = [
    "README.md",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "openenv.yaml",
    "client.py",
    "models.py",
    "tasks.py",
    "graders.py",
    "openenv_compat.py",
    "inference.py",
    "validate_local.py",
    "__init__.py",
]

DIRS_TO_COPY = [
    "server",
    "env_data",
]


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path in FILES_TO_COPY:
        source = ROOT / relative_path
        target = BUNDLE_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative_path in DIRS_TO_COPY:
        source = ROOT / relative_path
        target = BUNDLE_DIR / relative_path
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    print(f"Prepared submission bundle at: {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
