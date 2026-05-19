from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import AnnotationRecord


@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def convert_yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    xmin = (x_center - box_width / 2.0) * float(image_width)
    ymin = (y_center - box_height / 2.0) * float(image_height)
    xmax = (x_center + box_width / 2.0) * float(image_width)
    ymax = (y_center + box_height / 2.0) * float(image_height)

    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(image_width), xmax)
    ymax = min(float(image_height), ymax)

    if xmax <= xmin:
        xmax = min(float(image_width), xmin + 1.0)
    if ymax <= ymin:
        ymax = min(float(image_height), ymin + 1.0)

    return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def load_yolo_annotations(label_path: Path) -> tuple[AnnotationRecord, ...]:
    if not label_path.exists():
        return ()

    annotations: list[AnnotationRecord] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            if width <= 0.0 or height <= 0.0:
                continue
            annotations.append(
                AnnotationRecord(
                    class_id=class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            )
    return tuple(annotations)


def build_frcnn_target(
    annotations: tuple[AnnotationRecord, ...],
    image_width: int,
    image_height: int,
    image_id: int,
    include_background_offset: bool = True,
) -> dict[str, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for annotation in annotations:
        box = convert_yolo_to_xyxy(
            annotation.x_center,
            annotation.y_center,
            annotation.width,
            annotation.height,
            image_width,
            image_height,
        )
        boxes.append([box.xmin, box.ymin, box.xmax, box.ymax])
        labels.append(annotation.class_id + 1 if include_background_offset else annotation.class_id)

    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        label_tensor = torch.zeros((0,), dtype=torch.int64)
        area = torch.zeros((0,), dtype=torch.float32)

    return {
        "boxes": box_tensor,
        "labels": label_tensor,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "iscrowd": torch.zeros((len(label_tensor),), dtype=torch.int64),
        "area": area,
    }