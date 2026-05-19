# Medical Detection Project

This project compares two object detection approaches for gastrointestinal lesion detection in capsule endoscopy images:

- YOLOv8
- Faster R-CNN

The repository includes dataset preparation utilities, training and evaluation notebooks, experiment outputs, and a reusable Python package in `medical_detection/`.

## Project structure

- `medical_detection/` - shared Python utilities for dataset handling, YOLO/Faster R-CNN support, evaluation, and artifact helpers
- `01-data-visualisation-and-split.ipynb` - dataset inspection and train/validation/test split generation
- `02-yolov8-training.ipynb` - YOLOv8 training workflow
- `03-faster-rcnn-training.ipynb` - Faster R-CNN training workflow
- `04-evaluation-and-comparison.ipynb` - model evaluation and comparison
- `05-dataset-class-and-annotation-analysis.ipynb` - dataset analysis
- `06-bounding-box-annotation-examples.ipynb` - annotation examples

## Dataset

This project uses the KYU Capsule dataset from Kaggle:

https://www.kaggle.com/datasets/capsuleyolo/kyucapsule

After downloading and extracting the dataset, place the dataset files where the notebooks and project code can access them. The project can also detect the Kaggle input path automatically when running inside Kaggle.

After downloading and extracting it locally, place the dataset contents in the project root so these paths are available:

- `SEE_AI_project_all_images/SEE_AI_project_all_images`
- `SEE_AI_project_all_txt/SEE_AI_project_all_txt`
- `all_annotation.csv`

## Setup

1. Create a virtual environment.
2. Activate it.
3. Install the dependencies from `requirements.txt`.

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run

The project is mainly run through the notebooks in this order:

1. `01-data-visualisation-and-split.ipynb`
2. `02-yolov8-training.ipynb`
3. `03-faster-rcnn-training.ipynb`
4. `04-evaluation-and-comparison.ipynb`

The remaining notebooks are optional and provide additional dataset and annotation analysis.

To start Jupyter:

```powershell
jupyter notebook
```

Then open the notebooks above and run them in sequence.

## Outputs

Training outputs and generated artifacts are stored in directories such as:

- `runs/`
- `yolo_dataset/`
- `dataset_splits/`

## Summary

This repository is a practical experiment and comparison project for lesion detection, covering dataset preparation, YOLOv8 training, Faster R-CNN training, and evaluation in a notebook-driven workflow.
