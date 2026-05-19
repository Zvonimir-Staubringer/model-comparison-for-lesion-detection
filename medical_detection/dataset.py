from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class AnnotationRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    group_id: str
    annotations: tuple[AnnotationRecord, ...]

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def has_annotations(self) -> bool:
        return bool(self.annotations)

    @property
    def classes(self) -> set[int]:
        return {annotation.class_id for annotation in self.annotations}


@dataclass(frozen=True)
class DatasetIndex:
    records: tuple[ImageRecord, ...]
    class_names: tuple[str, ...]

    @property
    def positive_records(self) -> tuple[ImageRecord, ...]:
        return tuple(record for record in self.records if record.has_annotations)

    @property
    def negative_records(self) -> tuple[ImageRecord, ...]:
        return tuple(record for record in self.records if not record.has_annotations)

    @property
    def by_filename(self) -> dict[str, ImageRecord]:
        return {record.filename: record for record in self.records}


def _load_class_names(csv_path: Path) -> list[str]:
    class_mapping: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        class_names_row = next(reader)
        class_ids_row = next(reader)
        for name, class_id_str in zip(class_names_row, class_ids_row):
            value = class_id_str.strip()
            if value.isdigit():
                class_mapping[int(value)] = name.strip()
    max_class_id = max(class_mapping)
    return [class_mapping.get(index, f"missing_class_{index}") for index in range(max_class_id + 1)]


def _parse_annotation_line(line: str) -> AnnotationRecord | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    class_id = int(float(parts[0]))
    x_center, y_center, width, height = map(float, parts[1:])

    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return None

    return AnnotationRecord(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def _read_label_file(label_path: Path) -> tuple[AnnotationRecord, ...]:
    if not label_path.exists():
        return ()

    annotations: list[AnnotationRecord] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            annotation = _parse_annotation_line(line)
            if annotation is not None:
                annotations.append(annotation)
    return tuple(annotations)


def _infer_group_id(filename: str, group_size: int = 25) -> str:
    stem = Path(filename).stem
    digits = "".join(character for character in stem if character.isdigit())
    if not digits:
        return stem
    index = int(digits)
    group_start = ((index - 1) // group_size) * group_size + 1
    group_end = group_start + group_size - 1
    return f"{group_start:05d}-{group_end:05d}"


def build_dataset_index(image_dir: Path, label_dir: Path, csv_path: Path, group_size: int = 25) -> DatasetIndex:
    class_names = _load_class_names(csv_path)

    image_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(image_dir.rglob(pattern)))

    records: list[ImageRecord] = []
    for image_path in sorted(image_paths):
        label_path = label_dir / f"{image_path.stem}.txt"
        annotations = _read_label_file(label_path)
        records.append(
            ImageRecord(
                image_path=image_path,
                label_path=label_path,
                group_id=_infer_group_id(image_path.name, group_size=group_size),
                annotations=annotations,
            )
        )

    return DatasetIndex(records=tuple(records), class_names=tuple(class_names))


def _class_counts(records: Iterable[ImageRecord]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        for class_id in record.classes:
            counts[class_id] += 1
    return counts


def grouped_split(
    dataset_index: DatasetIndex,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in dataset_index.records:
        groups[record.group_id].append(record)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    target_sizes = {
        "train": int(len(dataset_index.records) * train_ratio),
        "val": int(len(dataset_index.records) * val_ratio),
    }
    target_sizes["test"] = len(dataset_index.records) - target_sizes["train"] - target_sizes["val"]
    desired_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_records = max(len(dataset_index.records), 1)

    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    split_class_counts: dict[str, Counter[int]] = {name: Counter() for name in splits}
    split_negative_counts = {name: 0 for name in splits}
    total_class_counts = _class_counts(dataset_index.records)
    total_negative_count = len(dataset_index.negative_records)

    def score_split(split_name: str, records: list[ImageRecord]) -> tuple[float, float, float, float]:
        prospective_sizes = {name: len(split_records) for name, split_records in splits.items()}
        prospective_sizes[split_name] += len(records)

        # Prevent overshooting a target when another split can still absorb the group.
        overshoot_penalty = sum(
            max(0, prospective_sizes[name] - target_sizes[name]) / max(target_sizes[name], 1)
            for name in splits
        )
        size_penalty = sum(
            ((prospective_sizes[name] / total_records) - desired_ratios[name]) ** 2
            for name in splits
        )

        added_class_counts: Counter[int] = Counter()
        added_negative_count = 0
        for record in records:
            if record.has_annotations:
                for class_id in record.classes:
                    added_class_counts[class_id] += 1
            else:
                added_negative_count += 1

        imbalance_penalty = 0.0
        for class_id, total_count in total_class_counts.items():
            for candidate_split_name in splits:
                count = split_class_counts[candidate_split_name][class_id]
                if candidate_split_name == split_name:
                    count += added_class_counts[class_id]
                imbalance_penalty += ((count / total_count) - desired_ratios[candidate_split_name]) ** 2

        negative_penalty = 0.0
        if total_negative_count:
            for candidate_split_name in splits:
                count = split_negative_counts[candidate_split_name]
                if candidate_split_name == split_name:
                    count += added_negative_count
                negative_penalty += ((count / total_negative_count) - desired_ratios[candidate_split_name]) ** 2

        return (overshoot_penalty, size_penalty, imbalance_penalty, negative_penalty)

    for _, records in sorted(group_items, key=lambda item: len(item[1]), reverse=True):
        best_split = min(splits, key=lambda split_name: score_split(split_name, records))
        splits[best_split].extend(records)
        for record in records:
            if not record.has_annotations:
                split_negative_counts[best_split] += 1
            for class_id in record.classes:
                split_class_counts[best_split][class_id] += 1

    for name in splits:
        splits[name] = sorted(splits[name], key=lambda record: record.filename)

    return splits


def downsample_negative_records(
    records: Iterable[ImageRecord],
    target_negative_ratio: float,
    seed: int = 42,
) -> list[ImageRecord]:
    if not 0.0 <= target_negative_ratio < 1.0:
        raise ValueError("target_negative_ratio must be in the range [0.0, 1.0)")

    record_list = list(records)
    positive_records = [record for record in record_list if record.has_annotations]
    negative_records = [record for record in record_list if not record.has_annotations]

    if not negative_records:
        return sorted(record_list, key=lambda record: record.filename)
    if target_negative_ratio == 0.0:
        return sorted(positive_records, key=lambda record: record.filename)
    if not positive_records:
        return sorted(record_list, key=lambda record: record.filename)

    max_negative_count = math.floor((target_negative_ratio * len(positive_records)) / (1.0 - target_negative_ratio))
    if max_negative_count >= len(negative_records):
        return sorted(record_list, key=lambda record: record.filename)

    rng = random.Random(seed)
    sampled_negative_records = rng.sample(negative_records, k=max_negative_count)
    return sorted([*positive_records, *sampled_negative_records], key=lambda record: record.filename)


def write_split_files(
    split_dir: Path,
    splits: dict[str, list[ImageRecord]],
    relative_to: Path | None = None,
    absolute_paths: bool = False,
    suffix: str = "",
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        split_path = split_dir / f"{split_name}{suffix}.txt"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in records:
                if absolute_paths:
                    value = str(record.image_path)
                elif relative_to is not None:
                    value = record.image_path.relative_to(relative_to).as_posix()
                else:
                    value = record.image_path.name
                handle.write(f"{value}\n")


def records_from_filenames(dataset_index: DatasetIndex, filenames: Iterable[str]) -> list[ImageRecord]:
    lookup = dataset_index.by_filename
    records: list[ImageRecord] = []
    for name in filenames:
        record = lookup.get(Path(name).name)
        if record is None:
            raise KeyError(f"Filename {name} not found in dataset index")
        records.append(record)
    return records


def image_level_class_distribution(records: Iterable[ImageRecord], num_classes: int) -> dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for record in records:
        for class_id in record.classes:
            counts[class_id] += 1
    return counts


def box_level_class_distribution(records: Iterable[ImageRecord], num_classes: int) -> dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for record in records:
        for annotation in record.annotations:
            counts[annotation.class_id] += 1
    return counts