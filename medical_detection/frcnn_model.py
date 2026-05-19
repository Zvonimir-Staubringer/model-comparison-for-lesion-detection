from __future__ import annotations

import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_model(num_classes: int, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate_faster_rcnn(model, dataloader, device: torch.device, class_metrics: bool = True) -> dict[str, torch.Tensor]:
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=class_metrics)
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = [{key: value.cpu() for key, value in output.items()} for output in outputs]
            refs = [{"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()} for target in targets]
            metric.update(preds, refs)
    return metric.compute()