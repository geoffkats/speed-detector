"""Export YOLO weights to ONNX for the production inference path.

Usage::

    uv run python -m scripts.export_model            # yolo11n.pt -> models/yolo11n.onnx
    uv run python -m scripts.export_model yolo11s.pt # a different backbone

The exported ``.onnx`` runs through ONNX Runtime (no PyTorch) and is what the
Docker production image will load. Keeping this as an explicit, repeatable
script — rather than auto-exporting on first run — is the dev -> deploy story.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def export(weights: str = "yolo11n.pt", out_dir: str = "models") -> Path:
    """Export ``weights`` to ONNX and place the result in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    onnx = model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)
    onnx_path = Path(str(onnx))
    target = out / onnx_path.name
    if onnx_path.resolve() != target.resolve():
        shutil.move(str(onnx_path), target)
    print(f"Exported ONNX model -> {target}")
    return target


if __name__ == "__main__":
    args = sys.argv[1:]
    export(args[0] if args else "yolo11n.pt")
