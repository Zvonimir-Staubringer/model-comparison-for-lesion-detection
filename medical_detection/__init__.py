"""Shared utilities for the medical detection project."""

from .config import ProjectPaths, class_names_from_csv
from .dataset import (
    DatasetIndex,
    ImageRecord,
    AnnotationRecord,
    build_dataset_index,
    box_level_class_distribution,
    downsample_negative_records,
    grouped_split,
    image_level_class_distribution,
    records_from_filenames,
    write_split_files,
)
from .detection import (
    build_frcnn_target,
    convert_yolo_to_xyxy,
    load_yolo_annotations,
)
from .env import resolve_dataset_root, resolve_project_root
from .evaluation import (
    DetectionMatchSummary,
    compute_detection_match_summary,
    summarize_detection_rows,
)
from .frcnn_dataset import FRCNNDataset, FRCNNDatasetConfig, collate_fn
from .frcnn_model import build_faster_rcnn_model, evaluate_faster_rcnn
from .artifacts import find_best_frcnn_checkpoint, find_best_yolo_weights
from .training_presets import COMMON_COMPARISON_TRAINING_CONFIG
from .yolo_dataset import load_split_filenames, materialize_yolo_dataset, write_yolo_data_yaml

__all__ = [
    "AnnotationRecord",
    "COMMON_COMPARISON_TRAINING_CONFIG",
    "DatasetIndex",
    "DetectionMatchSummary",
    "FRCNNDataset",
    "FRCNNDatasetConfig",
    "ImageRecord",
    "ProjectPaths",
    "build_faster_rcnn_model",
    "build_dataset_index",
    "build_frcnn_target",
    "box_level_class_distribution",
    "class_names_from_csv",
    "collate_fn",
    "compute_detection_match_summary",
    "convert_yolo_to_xyxy",
    "downsample_negative_records",
    "evaluate_faster_rcnn",
    "find_best_frcnn_checkpoint",
    "find_best_yolo_weights",
    "grouped_split",
    "image_level_class_distribution",
    "load_split_filenames",
    "load_yolo_annotations",
    "materialize_yolo_dataset",
    "records_from_filenames",
    "resolve_dataset_root",
    "resolve_project_root",
    "summarize_detection_rows",
    "write_split_files",
    "write_yolo_data_yaml",
]