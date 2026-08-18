# Sample media

This directory holds sample videos/images for running the pipeline. Media is
**not committed** (see `.gitignore`) — fetch your own, then run.

## Quick verification (single image)

Grab any traffic photo, then:

```bash
uv run python -m scripts.verify_detection traffic.jpg annotated.jpg
```

This loads the ONNX model, detects vehicles, and writes an annotated copy.
A verified example: Wikimedia's *Road traffic in Gwalior.jpg* (3888×2592)
detected **17 vehicles** (11 cars, 2 motorcycles, 4 trucks) via ONNX Runtime on
CPU.

## Full video run

```bash
uv run speed-detector run traffic.mp4 --annotate out.mp4
uv run speed-detector run traffic.mp4 --annotate out.mp4 --calibration road.json
```

Speeds (mph) require a `--calibration` JSON — the 4 pixel corners of a known
road rectangle plus that rectangle's real-world size in meters:

```json
{"src": [[120, 400], [600, 400], [600, 700], [120, 700]],
 "real_width_m": 3.5, "real_length_m": 20.0}
```

Without `--calibration`, boxes + tracker IDs are drawn but mph labels are
omitted (real-world units require a perspective transform).

## Exporting the model

```bash
uv run python -m scripts.export_model            # yolo11n.pt -> models/yolo11n.onnx
```
