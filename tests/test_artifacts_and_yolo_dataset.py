from pathlib import Path

import yaml

from medical_detection.artifacts import find_best_frcnn_checkpoint, find_best_yolo_weights
from medical_detection.yolo_dataset import load_split_filenames, materialize_yolo_dataset, write_yolo_data_yaml


def test_find_best_yolo_weights_prefers_helper_file(tmp_path: Path) -> None:
    detect_dir = tmp_path / "runs" / "detect"
    run_dir = detect_dir / "lesion_yolov8s"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_path = weights_dir / "best.pt"
    best_path.write_text("weights", encoding="utf-8")
    (detect_dir / "latest_yolo_run.txt").write_text(str(run_dir), encoding="utf-8")

    assert find_best_yolo_weights(tmp_path) == best_path


def test_find_best_frcnn_checkpoint_prefers_best_file(tmp_path: Path) -> None:
    frcnn_dir = tmp_path / "runs" / "frcnn"
    frcnn_dir.mkdir(parents=True)
    latest_path = frcnn_dir / "faster_rcnn_latest.pth"
    best_path = frcnn_dir / "faster_rcnn_best.pth"
    latest_path.write_text("latest", encoding="utf-8")
    best_path.write_text("best", encoding="utf-8")

    assert find_best_frcnn_checkpoint(tmp_path) == best_path


def test_materialize_yolo_dataset_and_write_yaml(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "materialized"
    splits_dir.mkdir()
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / "image00001.jpg"
    image_path.write_bytes(b"image")
    label_path = label_dir / "image00001.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    for split_name in ("train", "val", "test"):
        (splits_dir / f"{split_name}.txt").write_text("image00001.jpg\n", encoding="utf-8")

    materialize_yolo_dataset(splits_dir, image_dir, label_dir, output_dir)

    for split_name in ("train", "val", "test"):
        assert (output_dir / "images" / split_name / "image00001.jpg").exists()
        assert (output_dir / "labels" / split_name / "image00001.txt").exists()

    yaml_path = write_yolo_data_yaml(output_dir, ["lesion"], tmp_path / "data.yaml")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["path"] == str(output_dir)
    assert payload["names"] == ["lesion"]
    assert load_split_filenames(splits_dir / "train.txt") == ["image00001.jpg"]


def test_materialize_yolo_dataset_can_use_split_suffix(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "materialized"
    splits_dir.mkdir()
    image_dir.mkdir()
    label_dir.mkdir()

    for image_name in ("image00001.jpg", "image00002.jpg"):
        (image_dir / image_name).write_bytes(b"image")
        (label_dir / f"{Path(image_name).stem}.txt").write_text("", encoding="utf-8")

    (splits_dir / "train_bg15.txt").write_text("image00002.jpg\n", encoding="utf-8")
    (splits_dir / "val_bg15.txt").write_text("image00001.jpg\n", encoding="utf-8")
    (splits_dir / "test_bg15.txt").write_text("image00001.jpg\n", encoding="utf-8")

    materialize_yolo_dataset(splits_dir, image_dir, label_dir, output_dir, split_suffix="_bg15")

    assert (output_dir / "images" / "train" / "image00002.jpg").exists()
    assert not (output_dir / "images" / "train" / "image00001.jpg").exists()