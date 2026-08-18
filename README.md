# Speed Detector

> Real-time vehicle speed detection and traffic analytics platform — a 2026
> ground-up rewrite of a 2019 OpenCV/KNN speed detector.

A modern, CPU-deployable traffic-vision platform: YOLO detection, ByteTrack
tracking, perspective-transform calibration for **real-world** speeds (mph/kph,
not pixel heuristics), a FastAPI + React dashboard, Docker, and CI.

## Status

🚧 Under active development. The blueprint lives in
[PLAN.md](./PLAN.md) (roadmap) and [ARCHITECTURE.md](./ARCHITECTURE.md)
(technical design).

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation: packaging, lint, types, tests, CI | 🚧 in progress |
| 2 | Modern CV pipeline: YOLO + ByteTrack + calibration | ⏳ pending |
| 3 | Web dashboard & API | ⏳ pending |
| 4 | Production hardening: Docker, multi-camera, observability | ⏳ pending |
| 5 | Futuristic differentiators: lane seg, anomaly, privacy, LLM | ⏳ pending |

## Quickstart

```bash
git clone <repo>
cd speed-detector
uv sync --extra dev          # creates .venv + installs Python 3.12 via uv
uv run speed-detector --help
uv run speed-detector run path/to/video.mp4   # Phase 1: prints frame shape + FPS
```

Run the checks locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -v
```

## Tech stack

- **Python 3.12**, **uv**, **hatchling** — reproducible from clone
- **YOLOv11n** (Ultralytics) for dev; **ONNX Runtime** for production inference
- **ByteTrack** multi-object tracking
- **FastAPI** + **Vite/React/TypeScript** dashboard
- **SQLite** + **DuckDB** analytics; **ruff** / **mypy** / **pytest**; **Docker**; **GitHub Actions**

## Legacy

The original 2019 implementation (KNN background subtraction, hand-rolled
centroid tracking, dead Caltrans stream URLs) is preserved in
[`legacy/`](./legacy/) for reference. It is **not** imported by the new
`speed_detector` package.

## License

MIT — see [LICENSE](./LICENSE).
