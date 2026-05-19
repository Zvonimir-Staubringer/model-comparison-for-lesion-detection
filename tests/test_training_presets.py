from medical_detection import COMMON_COMPARISON_TRAINING_CONFIG


def test_common_comparison_training_config_matches_expected_values() -> None:
    assert COMMON_COMPARISON_TRAINING_CONFIG["epochs"] == 100
    assert COMMON_COMPARISON_TRAINING_CONFIG["seed"] == 42
    assert COMMON_COMPARISON_TRAINING_CONFIG["patience"] == 30
    assert COMMON_COMPARISON_TRAINING_CONFIG["optimizer"] == "AdamW"
    assert COMMON_COMPARISON_TRAINING_CONFIG["lr0"] == 0.001
    assert COMMON_COMPARISON_TRAINING_CONFIG["lrf"] == 0.05
    assert COMMON_COMPARISON_TRAINING_CONFIG["weight_decay"] == 0.0005
    assert COMMON_COMPARISON_TRAINING_CONFIG["warmup_epochs"] == 5
    assert COMMON_COMPARISON_TRAINING_CONFIG["cos_lr"] is True
    assert COMMON_COMPARISON_TRAINING_CONFIG["workers"] == 4