from __future__ import annotations


COMMON_COMPARISON_TRAINING_CONFIG = {
    "epochs": 100,
    "seed": 42,
    "patience": 30,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.05,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "cos_lr": True,
    "workers": 4,
}