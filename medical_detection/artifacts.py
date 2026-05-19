from __future__ import annotations

from pathlib import Path


def find_best_yolo_weights(project_root: Path) -> Path:
    detect_dir = project_root / "runs" / "detect"
    helper_path = detect_dir / "latest_yolo_run.txt"
    if helper_path.exists():
        run_dir = Path(helper_path.read_text(encoding="utf-8").strip())
        candidate = run_dir / "weights" / "best.pt"
        if candidate.exists():
            return candidate

    candidates = sorted(detect_dir.glob("*/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No YOLO best.pt file found under {detect_dir}")
    return candidates[0]


def find_best_frcnn_checkpoint(project_root: Path) -> Path:
    frcnn_dir = project_root / "runs" / "frcnn"
    preferred = frcnn_dir / "faster_rcnn_best.pth"
    if preferred.exists():
        return preferred

    candidates = sorted(frcnn_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Faster R-CNN checkpoint found under {frcnn_dir}")
    return candidates[0]