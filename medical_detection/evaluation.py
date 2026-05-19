from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from typing import Iterable

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class DetectionMatchSummary:
    matched_pairs: int
    mean_matched_iou: float
    mean_best_iou_per_gt: float
    mean_best_iou_per_pred: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _empty_summary() -> DetectionMatchSummary:
    return DetectionMatchSummary(
        matched_pairs=0,
        mean_matched_iou=0.0,
        mean_best_iou_per_gt=0.0,
        mean_best_iou_per_pred=0.0,
    )


def compute_detection_match_summary(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> DetectionMatchSummary:
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return _empty_summary()
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return _empty_summary()

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    label_mask = pred_labels[:, None] == gt_labels[None, :]
    iou_matrix = torch.where(label_mask, iou_matrix, torch.zeros_like(iou_matrix))

    best_iou_per_gt = iou_matrix.max(dim=0).values if gt_boxes.numel() else torch.zeros((0,), dtype=torch.float32)
    best_iou_per_pred = iou_matrix.max(dim=1).values if pred_boxes.numel() else torch.zeros((0,), dtype=torch.float32)

    matches: list[float] = []
    working_matrix = iou_matrix.clone()
    while working_matrix.numel() > 0:
        max_iou = working_matrix.max()
        if max_iou.item() < iou_threshold:
            break
        pred_index, gt_index = (working_matrix == max_iou).nonzero(as_tuple=False)[0]
        matches.append(float(max_iou.item()))
        working_matrix[pred_index, :] = -1.0
        working_matrix[:, gt_index] = -1.0

    return DetectionMatchSummary(
        matched_pairs=len(matches),
        mean_matched_iou=mean(matches) if matches else 0.0,
        mean_best_iou_per_gt=float(best_iou_per_gt.mean().item()) if best_iou_per_gt.numel() else 0.0,
        mean_best_iou_per_pred=float(best_iou_per_pred.mean().item()) if best_iou_per_pred.numel() else 0.0,
    )


def summarize_detection_rows(rows: Iterable[DetectionMatchSummary]) -> dict[str, float]:
    collected = list(rows)
    if not collected:
        return {
            "dataset_mean_matched_iou": 0.0,
            "dataset_mean_best_iou_per_gt": 0.0,
            "dataset_mean_best_iou_per_pred": 0.0,
        }

    return {
        "dataset_mean_matched_iou": mean(row.mean_matched_iou for row in collected),
        "dataset_mean_best_iou_per_gt": mean(row.mean_best_iou_per_gt for row in collected),
        "dataset_mean_best_iou_per_pred": mean(row.mean_best_iou_per_pred for row in collected),
    }