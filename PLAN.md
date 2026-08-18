# Speed Detector — Modernization Plan

> Blueprint for rewriting the 2019 OpenCV/KNN speed detector into a 2026-standard,
> CPU-deployable, portfolio-grade real-time traffic analytics platform.

**Status:** Draft for review — no code written until this is approved.
**Target audience:** Recruiters, engineering managers, and the author's future self.
**Scope:** Full rewrite. The legacy `main.py` / `vehicle_counter.py` / `threadedcam.py`
are kept only as historical reference and will not be imported by the new system.

---

## 1. Why rewrite?

The existing project works, but it is anchored to a technology baseline that is now
~6 years old:

| Dimension | Legacy (2019) | Cost of keeping it |
|---|---|---|
| Detection | KNN background subtraction + contour filter | Fragile to shadows, weather, camera shake; no vehicle classes |
| Tracking | Hand-rolled centroid matching | Loses IDs on occlusion; "ghost tracks" (acknowledged TODO in code) |
| Speed | Pixel-distance heuristics, no real-world units | Numbers are only relative; not defensible |
| Python | Old-style `class X(object):`, no type hints, f-strings half-adopted | Reads as dated in 2026 |
| Packaging | None — run `python main.py` from the repo root | Not installable, not reproducible |
| UI | `cv2.imshow` windows | Not shareable, not screenshot-worthy for a portfolio |
| Sources | Hardcoded Caltrans `.m3u8` URLs (now 404) | Demo is dead on arrival |
| Tests / CI | None | No safety net, no signal of engineering rigor |

A patch-on-top would inherit every fragility above. A rewrite is the only path to a
result that looks, reads, and runs like a 2026 project.

---

## 2. Design principles

1. **CPU-first by default.** Everything must run on a laptop without a GPU. GPU/edge
   paths are optional accelerators, not prerequisites.
2. **Portfolio-grade readability.** Every module should be readable in one sitting.
   Type hints, docstrings on public APIs, no clever tricks without a comment.
3. **Dual inference paths.** An Ultralytics (PyTorch) path for development/iteration
   and an ONNX Runtime path for "production" inference. Showing you understand the
   dev→deploy distinction is itself a portfolio signal.
4. **Real units.** Speeds in mph/kph grounded in a perspective transform, not pixel
   heuristics. Calibration is a first-class UI flow, not a magic constant.
5. **Thin core, rich surface.** A small, well-typed pipeline core. Everything else
   (web, analytics, reports, LLM) is a consumer of that core and can be swapped.
6. **Reproducible from clone.** `git clone` → `uv sync` → `docker compose up` →
   working dashboard. No "works on my machine".
7. **Honest telemetry.** Show FPS, latency, model confidence, and failure modes in
   the UI. Hiding weaknesses is worse than surfacing them.

---

## 3. Target tech stack (CPU-first)

### Core / CV
- **Python 3.12+**
- **uv** for dependency management (replaces pip/poetry; fast, lockfile-backed)
- **Ultralytics YOLOv11n** (nano) for detection — dev path, handles model download
- **ONNX Runtime** (`onnxruntime`) for the production inference path on CPU
- **OpenVINO** (optional) — Intel CPU acceleration where available
- **ByteTrack** (via `ultralytics` `.track()` or `supervision`) for multi-object tracking
- **OpenCV 4.x** only for I/O, drawing, and geometric ops (no longer the detector)
- **NumPy** for array work

### Web / API
- **FastAPI** + **Uvicorn** (REST + WebSocket)
- **Pydantic v2** / **SQLModel** for models and settings
- **SQLite** for events (zero-config); **DuckDB** for analytics queries (trendy, fast)
- **Vite + React + TypeScript** frontend
- **Tailwind CSS + shadcn/ui** components
- **Recharts** / **visx** for charts; **leaflet** for map overlays (multi-camera)

### Quality / Ops
- **ruff** (lint + format), **mypy** (types), **pre-commit**
- **pytest** + **pytest-cov** (target ≥80% on the pipeline core)
- **GitHub Actions** — ruff, mypy, pytest, build, docker build, publish image
- **Docker** multi-stage build (CPU image ~600MB); **docker-compose** for local stack
- **devcontainer** for one-click VS Code setup

### "Futuristic" optional layer
- **Ollama** + a local LLM (e.g. `qwen2.5:3b` or `llama3.2:3b`) for natural-language
  incident summaries ("A white sedan averaged 78 mph in the left lane over 30s")
- **RT-DETR** ONNX export as an alternative detector (transformer-based, SOTA) to show
  breadth — selectable via config

---

## 4. Phased delivery

Each phase ends in a demonstrable, committed checkpoint. Nothing here is "we'll get to
it" — every phase has a concrete definition of done.

### Phase 1 — Foundation ⏳
**Deliverable:** A clean, installable, linted, tested skeleton with a "hello world"
pipeline that reads a video file and prints frame shape.

- [ ] `pyproject.toml` (uv, ruff, mypy, pytest config; Python `>=3.12`)
- [ ] `src/speed_detector/` package layout (see ARCHITECTURE.md §4)
- [ ] `speed_detector.cli` entry point with **Typer** — `speed-detector run <source>`
- [ ] `config.py` via `pydantic-settings`, reads `.env`
- [ ] `.env.example`, `.gitignore` updated for new layout
- [ ] `ruff` + `mypy` + `pre-commit` clean from day one
- [ ] `tests/` with `conftest.py` and one smoke test
- [ ] `.github/workflows/ci.yml` — ruff, mypy, pytest on push
- [ ] Legacy files moved to `legacy/` (kept for reference, not imported)
- [ ] **Definition of done:** `uv run speed-detector --help` works; CI is green on a
  trivial PR.

### Phase 2 — Modern CV pipeline
**Deliverable:** A script that ingests a sample video, detects vehicles with YOLO,
tracks them with ByteTrack, and reports real-world speeds via a perspective transform.

- [ ] `pipeline/source.py` — unified `VideoSource` (RTSP / file / USB / YouTube via
  `yt-dlp`; drops frames under load; async queue)
- [ ] `pipeline/detector.py` — `UltralyticsDetector` (dev) and `ONNXDetector`
  (production); both return typed `Detection` records
- [ ] `pipeline/tracker.py` — ByteTrack wrapper returning typed `Track` records
- [ ] `pipeline/calibrator.py` — perspective transform; 4-point calibration stored
  per-camera; converts pixels → meters
- [ ] `pipeline/speed.py` — ground-plane speed in m/s, mph, kph; EMA smoothing
- [ ] `pipeline/pipeline.py` — orchestrates source→detect→track→calibrate→speed
- [ ] `annotator.py` — draws boxes, IDs, trails, speed labels, lane polygons
- [ ] `scripts/download_sample_video.py` — fetches a royalty-free traffic clip so the
  demo is never dead (no reliance on Caltrans)
- [ ] Unit tests for `calibrator` (known geometry → known speed) and `speed` (synthetic
  tracks)
- [ ] **Definition of done:** `speed-detector run sample.mp4 --annotate out.mp4`
  produces a visibly correct annotated video with mph labels.

### Phase 3 — Web dashboard & API
**Deliverable:** Browser dashboard showing a live annotated stream + real-time metrics.

- [ ] `api/app.py` — FastAPI app; REST CRUD for cameras/calibration; WebSocket for the
  live annotated MJPEG/base64 stream
- [ ] `api/routes/` — cameras, sessions, events, analytics, export
- [ ] `db.py` + `models/` — SQLModel schema for `Camera`, `Session`, `VehicleEvent`
- [ ] `analytics/` — DuckDB queries: speed histogram, class breakdown, hourly trends
- [ ] Frontend: Vite+React+TS+Tailwind+shadcn
  - Live stream panel (WebSocket)
  - Real-time speed histogram (Recharts)
  - Vehicle-class donut, per-lane counts table
  - Camera list + "add camera" flow
  - Calibration UI (draw the road rectangle → store 4 points)
- [ ] `docker-compose.yml` — backend + frontend + (optional) Ollama
- [ ] **Definition of done:** `docker compose up` → open `localhost:5173` → see a live
  annotated demo stream with updating charts.

### Phase 4 — Production hardening
**Deliverable:** A deployable, observable, multi-camera system.

- [ ] Multi-camera manager (one pipeline task per camera, supervised lifecycle)
- [ ] Async ingestion that never blocks the event loop; bounded queues with backpressure
- [ ] `Dockerfile` (CPU, multi-stage, non-root user); optional `Dockerfile.gpu`
- [ ] `.devcontainer/devcontainer.json` — one-click VS Code + uv + Docker-in-Docker
- [ ] Structured logging (`structlog`); `/healthz` + `/metricsz` (Prometheus)
- [ ] pytest coverage gate in CI (≥80% on `pipeline/`)
- [ ] Integration test: spin the API + a fake RTSP server, assert events land in SQLite
- [ ] **Definition of done:** Two simulated cameras run for 10 min in compose with no
  memory leak and stable FPS; health endpoint green.

### Phase 5 — "Futuristic" differentiators
**Deliverable:** Features that make the repo memorable in a 60-second recruiter scroll.

- [ ] **Lane segmentation** — YOLO-seg or SegFormer; per-lane speed/counts (closes the
  oldest TODO in the original README)
- [ ] **Anomaly detection** — rule engine (wrong-way, stalled vehicle, pedestrian on
  highway) + optional LLM-generated natural-language incident report via Ollama
- [ ] **Privacy mode** — blur faces and license plates before any frame leaves the box
  (toggle in the UI; on by default for demo)
- [ ] **ONNX/TensorRT export script** — `scripts/export_model.py` shows the
  dev→deploy optimization story even though TensorRT needs a GPU
- [ ] **Polished README** — architecture diagram (Mermaid), demo GIF, feature list,
  quickstart, screenshots, "what I'd do next" section
- [ ] **Landing page** — the frontend's root route doubles as a project landing page
  with the diagram + demo + links
- [ ] **Definition of done:** README renders beautifully on GitHub; the live demo shows
  ≥3 differentiator features simultaneously.

---

## 5. What deliberately stays out of scope

To keep this finishable and focused:

- **Cloud SaaS / multi-tenancy / billing.** This is a portfolio piece, not a product.
- **Real-time alerting to external services** (Slack/email/SMS). A webhook stub is fine.
- **Training custom models.** We use pretrained YOLO weights only.
- **Mobile app.** The dashboard is responsive web; that's enough.
- **Caltrans streams.** Dead and out of our control. We ship a royalty-free sample and
  accept user-provided RTSP/file/YouTube.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CPU FPS too low to feel "real-time" | YOLOv11n@640 on a modern laptop CPU hits ~15–25 FPS via ONNX Runtime; OpenVINO lifts this further. We show real FPS in the UI rather than pretending. |
| ByteTrack ID switches on occlusion | Acceptable for a portfolio piece; surface it honestly in metrics. Upgrading to BoT-SORT is a one-line config change via `supervision`. |
| Perspective-transform accuracy | Calibration UI makes the assumptions visible. We never claim lab-grade accuracy; we show the method is sound. |
| Frontend scope creep | A fixed, small component list (§3). No auth, no multi-tenant, no settings page beyond camera management. |
| Local LLM weight download in CI | Ollama is optional and compose-only; CI never starts it. |

---

## 7. Success criteria

The rewrite succeeds for its portfolio purpose if, in ~60 seconds, a visitor can:

1. `git clone` + `docker compose up` and see a live dashboard.
2. Read the README and understand the architecture from the Mermaid diagram.
3. See real mph numbers, vehicle classes, and per-lane counts updating in the UI.
4. Open `src/speed_detector/pipeline/` and find typed, tested, readable code.
5. Notice at least one "futuristic" feature (anomaly report, privacy blur, lane seg).

If all five land, this project is no longer "a 2019 OpenCV script" — it's a 2026
real-time vision platform.
