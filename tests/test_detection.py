from medical_detection.dataset import AnnotationRecord
from medical_detection.detection import build_frcnn_target, convert_yolo_to_xyxy


def test_convert_yolo_to_xyxy_clamps_and_preserves_float_precision() -> None:
    box = convert_yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, image_width=1000, image_height=500)
    assert box.xmin == 400.0
    assert box.ymin == 150.0
    assert box.xmax == 600.0
    assert box.ymax == 350.0


def test_build_frcnn_target_offsets_labels_for_background() -> None:
    annotations = (
        AnnotationRecord(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1),
        AnnotationRecord(class_id=3, x_center=0.4, y_center=0.4, width=0.2, height=0.2),
    )

    target = build_frcnn_target(annotations, image_width=200, image_height=100, image_id=5)

    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 4]
    assert target["image_id"].tolist() == [5]