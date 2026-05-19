from pathlib import Path

from medical_detection.dataset import (
    AnnotationRecord,
    DatasetIndex,
    ImageRecord,
    downsample_negative_records,
    grouped_split,
)


def _record(index: int, group_id: str, class_ids: tuple[int, ...]) -> ImageRecord:
    annotations = tuple(
        AnnotationRecord(class_id=class_id, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        for class_id in class_ids
    )
    return ImageRecord(
        image_path=Path(f"image{index:05d}.jpg"),
        label_path=Path(f"image{index:05d}.txt"),
        group_id=group_id,
        annotations=annotations,
    )


def test_grouped_split_keeps_groups_together() -> None:
    records = (
        _record(1, "00001-00025", (0,)),
        _record(2, "00001-00025", (1,)),
        _record(26, "00026-00050", (0, 1)),
        _record(27, "00026-00050", ()),
        _record(51, "00051-00075", (2,)),
        _record(52, "00051-00075", (2,)),
    )
    dataset_index = DatasetIndex(records=records, class_names=("a", "b", "c"))

    splits = grouped_split(dataset_index, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7)

    assigned_groups: dict[str, str] = {}
    for split_name, split_records in splits.items():
        for record in split_records:
            assigned_groups.setdefault(record.group_id, split_name)
            assert assigned_groups[record.group_id] == split_name


def test_grouped_split_preserves_dataset_size() -> None:
    records = tuple(_record(index, f"{index:05d}-{index:05d}", (index % 3,)) for index in range(1, 13))
    dataset_index = DatasetIndex(records=records, class_names=("a", "b", "c"))

    splits = grouped_split(dataset_index, seed=21)

    total = sum(len(records_for_split) for records_for_split in splits.values())
    assert total == len(records)


def test_grouped_split_tracks_requested_ratios() -> None:
    records = tuple(_record(index, f"{index:05d}-{index:05d}", (index % 4,)) for index in range(1, 101))
    dataset_index = DatasetIndex(records=records, class_names=("a", "b", "c", "d"))

    splits = grouped_split(dataset_index, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=21)

    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 20
    assert len(splits["test"]) == 10


def test_downsample_negative_records_hits_requested_ratio() -> None:
    positives = [_record(index, f"p-{index:05d}", (0,)) for index in range(1, 6)]
    negatives = [_record(index + 100, f"n-{index:05d}", ()) for index in range(1, 6)]

    sampled = downsample_negative_records([*positives, *negatives], target_negative_ratio=0.25, seed=7)

    assert sum(record.has_annotations for record in sampled) == 5
    assert sum(not record.has_annotations for record in sampled) == 1


def test_downsample_negative_records_keeps_all_records_when_below_target() -> None:
    positives = [_record(index, f"p-{index:05d}", (0,)) for index in range(1, 6)]
    negatives = [_record(index + 100, f"n-{index:05d}", ()) for index in range(1, 2)]

    sampled = downsample_negative_records([*positives, *negatives], target_negative_ratio=0.25, seed=7)

    assert len(sampled) == 6
    assert sum(not record.has_annotations for record in sampled) == 1