from __future__ import annotations

from pathlib import Path
import os
import shutil
import yaml


def load_split_filenames(split_path: Path) -> list[str]:
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)


def materialize_yolo_dataset(
    splits_dir: Path,
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    split_names: tuple[str, ...] = ("train", "val", "test"),
    split_suffix: str = "",
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split_name in split_names:
        filenames = load_split_filenames(splits_dir / f"{split_name}{split_suffix}.txt")
        image_output = output_dir / "images" / split_name
        label_output = output_dir / "labels" / split_name

        for name in filenames:
            source_name = Path(name).name
            source_image = image_dir / source_name
            source_label = label_dir / f"{Path(source_name).stem}.txt"
            _link_or_copy(source_image, image_output / source_name)
            if source_label.exists():
                _link_or_copy(source_label, label_output / source_label.name)


def write_yolo_data_yaml(dataset_dir: Path, class_names: list[str], output_path: Path) -> Path:
    payload = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path