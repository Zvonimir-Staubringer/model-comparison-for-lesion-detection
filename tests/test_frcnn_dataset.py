from pathlib import Path

from PIL import Image

from medical_detection.dataset import AnnotationRecord, ImageRecord
from medical_detection.frcnn_dataset import FRCNNDataset, FRCNNDatasetConfig


def test_frcnn_dataset_can_filter_empty_images(tmp_path: Path) -> None:
    image_path = tmp_path / "image00001.jpg"
    Image.new("RGB", (64, 32), color=(255, 255, 255)).save(image_path)
    label_path = tmp_path / "image00001.txt"
    label_path.write_text("", encoding="utf-8")

    positive_image_path = tmp_path / "image00002.jpg"
    Image.new("RGB", (64, 32), color=(255, 255, 255)).save(positive_image_path)
    positive_label_path = tmp_path / "image00002.txt"
    positive_label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    records = [
        ImageRecord(image_path=image_path, label_path=label_path, group_id="1", annotations=()),
        ImageRecord(
            image_path=positive_image_path,
            label_path=positive_label_path,
            group_id="2",
            annotations=(AnnotationRecord(class_id=0, x_center=0.5, y_center=0.5, width=0.25, height=0.25),),
        ),
    ]

    dataset = FRCNNDataset(records, FRCNNDatasetConfig(include_empty_images=False, train=False))

    assert len(dataset) == 1
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 32, 64)
    assert target["labels"].tolist() == [1]