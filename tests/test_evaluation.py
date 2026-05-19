import torch

from medical_detection.evaluation import compute_detection_match_summary, summarize_detection_rows


def test_class_aware_detection_summary_rejects_wrong_label_match() -> None:
    pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    pred_labels = torch.tensor([2], dtype=torch.int64)
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)

    summary = compute_detection_match_summary(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5)

    assert summary.matched_pairs == 0
    assert summary.mean_best_iou_per_gt == 0.0
    assert summary.mean_best_iou_per_pred == 0.0


def test_class_aware_detection_summary_accepts_correct_label_match() -> None:
    pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    pred_labels = torch.tensor([1], dtype=torch.int64)
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.int64)

    summary = compute_detection_match_summary(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5)

    assert summary.matched_pairs == 1
    assert summary.mean_matched_iou == 1.0


def test_summarize_detection_rows_averages_values() -> None:
    rows = [
        compute_detection_match_summary(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
        ),
        compute_detection_match_summary(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
        ),
    ]

    summary = summarize_detection_rows(rows)

    assert summary["dataset_mean_matched_iou"] == 0.5
