from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    dataset_root: Path

    @property
    def image_dir(self) -> Path:
        return self.dataset_root / "SEE_AI_project_all_images" / "SEE_AI_project_all_images"

    @property
    def label_dir(self) -> Path:
        return self.dataset_root / "SEE_AI_project_all_txt" / "SEE_AI_project_all_txt"

    @property
    def csv_path(self) -> Path:
        return self.dataset_root / "all_annotation.csv"

    @property
    def splits_dir(self) -> Path:
        return self.project_root / "dataset_splits"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"


def class_names_from_csv(csv_path: Path) -> list[str]:
    class_mapping: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        class_names_row = next(reader)
        class_ids_row = next(reader)
        for name, class_id_str in zip(class_names_row, class_ids_row):
            value = class_id_str.strip()
            if value.isdigit():
                class_mapping[int(value)] = name.strip()

    if not class_mapping:
        raise ValueError(f"No class mapping found in {csv_path}")

    max_class_id = max(class_mapping)
    return [class_mapping.get(index, f"missing_class_{index}") for index in range(max_class_id + 1)]