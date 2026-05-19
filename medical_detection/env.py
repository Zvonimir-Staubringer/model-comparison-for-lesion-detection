from __future__ import annotations

from pathlib import Path


def resolve_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "medical_detection").exists():
            return candidate

    kaggle_candidate = Path("/kaggle/working/model-comparison-for-lesion-detection")
    if (kaggle_candidate / "medical_detection").exists():
        return kaggle_candidate

    raise FileNotFoundError("Could not locate a project root containing the medical_detection package.")


def resolve_dataset_root(project_root: Path) -> Path:
    kaggle_candidate = Path("/kaggle/input/datasets/capsuleyolo/kyucapsule")
    if kaggle_candidate.exists():
        return kaggle_candidate
    return project_root