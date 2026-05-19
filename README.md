# Model Comparison for Lesion Detection

This project compares lesion detection approaches for capsule endoscopy images, with experiments and evaluation for YOLOv8 and Faster R-CNN.

## Dataset

The dataset used in this project is available on Kaggle:

https://www.kaggle.com/datasets/capsuleyolo/kyucapsule

After downloading and extracting it locally, place the dataset contents in the project root so these paths are available:

- `SEE_AI_project_all_images/SEE_AI_project_all_images`
- `SEE_AI_project_all_txt/SEE_AI_project_all_txt`
- `all_annotation.csv`

## Setup

Create a virtual environment, activate it, and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

This project is primarily run through the notebooks in numerical order:

1. `01-data-visualisation-and-split.ipynb`
2. `02-yolov8-training.ipynb`
3. `03-faster-rcnn-training.ipynb`
4. `04-evaluation-and-comparison.ipynb`
5. `05-dataset-class-and-annotation-analysis.ipynb`
6. `06-bounding-box-annotation-examples.ipynb`

Start Jupyter and open the notebooks:

```bash
jupyter notebook
```

You can also run the test suite with:

```bash
pytest
```
