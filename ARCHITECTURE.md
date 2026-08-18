# Architecture

> Technical blueprint for the rewritten speed-detector. Companion to [PLAN.md](./PLAN.md).
> This document is the source of truth for structure and data flow; code that diverges
> from this document should update it.

---

## 1. System context

```
                 ┌─────────────────────────────────────────────┐
   Video in ───▶ │  speed_detector (this repo)                 │ ───▶ Browser
  (RTSP/file/    │                                             │       dashboard
   USB/YouTube)  │  ingest → detect → track → calibrate → speed │
                 │            │                                 │
                 │            ▼                                 │
                 │   SQLModel(SQLite) + DuckDB(analytics)       │
                 │            │                                 │
                 │            ▼ (optional)                      │
                 │        Ollama LLM (incident reports)         │
                 └─────────────────────────────────────────────┘
```

Everything runs on a single laptop (CPU). No cloud dependency. The browser dashboard
talks to a local FastAPI backend over REST + WebSocket.

---

## 2. High-level data flow

```
VideoSource ──frame──▶ Detector ──Detection[]──▶ Tracker ──Track[]──▶
                                                                 │
                                                                 ▼
   Annotator ◀── speed, class, id ── SpeedEstimator ◀── Calibrator (px→m)
       │
       ├──▶ Annotated frame ──▶ WebSocket (browser)
       └──▶ VehicleEvent ─────▶ SQLite ──▶ DuckDB ──▶ Analytics API
```

**Invariant:** the core pipeline (`source → detector → tracker → calibrator → speed`)
is pure-Python, synchronous on a per-frame basis, and has **no web, DB, or LLM
dependency**. Every downstream consumer (API, analytics, LLM) is a plugin that reads
the typed events the core emits. This keeps the core testable and swappable.

---

## 3. Key data types (typed boundary)

```python
# speed_detector/types.py
from dataclasses import dataclass
from enum import IntEnum


class VehicleClass(IntEnum):
    CAR = 0
    TRUCK = 1
    BUS = 2
    MOTORCYCLE = 3
    BICYCLE = 4


@dataclass(frozen=True, slots=True)
class BBox:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    confidence: float
    cls: VehicleClass


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    bbox: BBox
    confidence: float
    cls: VehicleClass
    frame_index: int


@dataclass(frozen=True, slots=True)
class SpeedReading:
    track_id: int
    cls: VehicleClass
    speed_mph: float
    speed_kph: float
    world_position_m: tuple[float, float]  # after perspective transform
    frame_index: int
```

Frozen `slots=True` dataclasses are used throughout the core for cache efficiency and
to make the data flow impossible to mutate by accident — a property the legacy code
lacked.

---

## 4. Package layout

```
speed-detector/
├── pyproject.toml              # uv-managed; ruff/mypy/pytest config
├── PLAN.md
├── ARCHITECTURE.md             # this file
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile                  # CPU multi-stage
├── .github/workflows/ci.yml
├── .devcontainer/
│   └── devcontainer.json
├── legacy/                     # old main.py / vehicle_counter.py / threadedcam.py
│   └── README.md               # explains why they're archived
├── scripts/
│   ├── download_sample_video.py
│   └── export_model.py          # YOLO → ONNX (and optional TensorRT) export
├── src/
│   └── speed_detector/
│       ├── __init__.py
│       ├── __main__.py          # python -m speed_detector
│       ├── cli.py               # Typer entry point
│       ├── config.py            # pydantic-settings, reads .env
│       ├── types.py             # §3 above
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── source.py        # VideoSource: RTSP/file/USB/YouTube, async queue
│       │   ├── detector.py      # UltralyticsDetector | ONNXDetector (Protocol)
│       │   ├── tracker.py       # ByteTrack wrapper
│       │   ├── calibrator.py    # perspective transform, 4-point per camera
│       │   ├── speed.py         # ground-plane velocity, EMA smoothing
│       │   └── pipeline.py      # orchestrator; emits events
│       ├── annotator.py         # draw boxes/trails/labels/lane polygons
│       ├── events.py            # VehicleEvent → SQLite writer (background thread)
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py           # FastAPI app factory
│       │   ├── deps.py          # DI: pipeline, db, settings
│       │   ├── routes/
│       │   │   ├── cameras.py
│       │   │   ├── sessions.py
│       │   │   ├── events.py
│       │   │   ├── analytics.py
│       │   │   └── export.py
│       │   └── ws.py            # WebSocket: annotated MJPEG stream + metrics
│       ├── models/
│       │   ├── __init__.py
│       │   ├── camera.py         # SQLModel: Camera, Calibration
│       │   ├── session.py
│       │   └── event.py          # VehicleEvent
│       ├── db.py                # engine, session factory, migrations (Alembic-lite)
│       ├── analytics/
│       │   ├── __init__.py
│       │   └── queries.py       # DuckDB over exported SQLite (or parquet)
│       ├── llm/
│       │   └── incident_report.py   # Ollama client; optional, behind a flag
│       └── utils/
│           ├── geometry.py       # homography helpers, lane masks
│           ├── logging.py        # structlog config
│           └── metrics.py        # Prometheus counters/histograms
├── web/                         # Vite + React + TS frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── LiveStream.tsx      # WebSocket MJPEG consumer
│       │   ├── SpeedHistogram.tsx
│       │   ├── ClassDonut.tsx
│       │   ├── LaneTable.tsx
│       │   ├── CameraList.tsx
│       │   └── Calibrator.tsx      # draw 4 points on a still frame
│       ├── hooks/
│       │   └── useStream.ts
│       ├── api/
│       │   └── client.ts
│       └── types/
│           └── schema.ts          # generated from FastAPI OpenAPI
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── synthetic_video.py     # generate deterministic test frames
│   ├── unit/
│   │   ├── test_calibrator.py
│   │   ├── test_speed.py
│   │   └── test_tracker.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_api.py
└── samples/
    └── README.md                   # how the bundled sample video is licensed
```

**Conventions:**
- `src/` layout → the package is never accidentally importable from the repo root,
  forcing real installation (`uv sync` / `pip install -e .`).
- `legacy/` isolates the old code so a reader can diff approaches but the new code
  never imports it.
- `web/` is a separate npm workspace; the backend serves its built `dist/` in
  production, and Vite proxies `/api` in dev.

---

## 5. Component contracts

### 5.1 VideoSource (`pipeline/source.py`)
```python
class VideoSource(Protocol):
    fps: float

    @property
    def frame_size(self) -> tuple[int, int]: ...
    def next_frame(self) -> np.ndarray | None: ...  # None = EOF/error
    def close(self) -> None: ...
```
- Implementations: `FileSource`, `RTSPSource`, `WebcamSource`, `YouTubeSource`
  (shells out to `yt-dlp` for the stream URL).
- A background thread fills a bounded `queue.Queue(maxsize=2)`; `next_frame()` never
  blocks the GIL-heavy CV path for long, and old frames are dropped on overflow
  (replaces the legacy `time.sleep(1/fps)` hack that caused the "abrupt jumps" the
  old README complained about).

### 5.2 Detector (`pipeline/detector.py`)
```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
```
- `UltralyticsDetector` — wraps `ultralytics.YOLO("yolo11n.pt")`; used in dev. Returns
  typed `Detection`s; filters to COCO vehicle classes {car, truck, bus, motorcycle}.
- `ONNXDetector` — loads `models/yolo11n.onnx` via `onnxruntime.InferenceSession`;
  implements letterbox/preprocess/postprocess. This is the "production" path and the
  one used in Docker to avoid pulling PyTorch into the image.
- Selection via `config.DETECTOR_BACKEND ∈ {"ultralytics","onnx"}`.

### 5.3 Tracker (`pipeline/tracker.py`)
ByteTrack via `supervision.ByteTrack` (cleaner API than raw ultralytics `.track()`,
and lets us swap to `BoT-SORT` with one line). Maintains `dict[int, Track]` and emits
enter/exit events to the `events.py` writer.

### 5.4 Calibrator (`pipeline/calibrator.py`)
```python
class Calibrator:
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray): ...
    @property
    def homography(self) -> np.ndarray: ...
    def to_world(self, pixel: tuple[float, float]) -> tuple[float, float]: ...
    def world_distance(self, a, b) -> float: ...  # meters
```
- 4-point perspective transform stored per camera in `Camera.calibration`.
- `src_points` are pixels clicked in the calibration UI; `dst_points` are real-world
  meters (the user measures a known stretch of road and types its length).
- Eliminates the legacy "average mode" vs "distance mode" split — there's now one
  correct method grounded in geometry.

### 5.5 SpeedEstimator (`pipeline/speed.py`)
- Converts a `Track`'s pixel trail to world coordinates via the `Calibrator`.
- Velocity = Δworld_distance / Δtime (time from `frame_index / fps`).
- EMA smoothing over the last N visible frames to suppress per-frame jitter.
- Outputs `SpeedReading` in both mph and kph.

### 5.6 Pipeline (`pipeline/pipeline.py`)
```python
class Pipeline:
    def __init__(self, source, detector, tracker, calibrator, on_event): ...
    def run(self) -> Iterator[AnnotatedFrame]: ...  # generator; cooperative
    def stop(self) -> None: ...
```
- One `Pipeline` per camera.
- `run()` is a generator so the caller (the API/CLI) controls backpressure and can
  interleave with WebSocket sends without threads for the hot path.
- `on_event: Callable[[VehicleEvent], None]` is the seam for the DB writer and LLM.

### 5.7 API (`api/`)
- `POST /cameras` — register a source + calibration
- `POST /cameras/{id}/start` — spawns a `Pipeline` task in a supervisor
- `GET /cameras/{id}/stream` — WebSocket; server pushes annotated JPEG frames +
  a small JSON metrics envelope at the source FPS (capped for browser sanity)
- `GET /sessions/{id}/events` — paginated `VehicleEvent`s
- `GET /analytics/speed-histogram?session=…` — DuckDB-backed
- `GET /export/session.csv` — CSV download

### 5.8 Frontend
Single-page React app, three primary views:
1. **Live** — stream panel + 3 live charts (histogram, donut, lane table)
2. **Cameras** — list + add-camera + calibration canvas
3. **History** — session list + per-session analytics + CSV export

Routing via `react-router`; data via `@tanstack/react-query`; charts via `recharts`.
The OpenAPI schema is the single source of truth for `web/src/types/`.

---

## 6. Threading & async model

Two worlds, cleanly separated:

- **CV core** runs in a dedicated thread per camera (OpenCV + ONNX Runtime release the
  GIL during native calls, so this is genuinely parallel). The pipeline `run()`
  generator yields `(annotated_frame, metrics)` into a thread-safe queue.
- **API layer** is `asyncio` (FastAPI). A small bridge reads the queue and broadcasts
  to connected WebSocket clients. No `asyncio` inside the CV core.

This split avoids the classic "I put OpenCV in an async function and my event loop
starves" bug, and keeps the core testable from plain `pytest` (no event loop needed).

```
   ┌──────────────┐   queue   ┌────────────────┐   WebSocket   ┌────────┐
   │ CV thread    │ ────────▶ │ asyncio bridge │ ─────────────▶ │ browser│
   │ (per camera) │           │ (FastAPI)      │               └────────┘
   └──────────────┘           └────────────────┘
```

---

## 7. Configuration

`config.Settings` (pydantic-settings) reads from `.env`:

```dotenv
# .env.example
DETECTOR_BACKEND=onnx            # onnx | ultralytics
ONNX_MODEL_PATH=models/yolo11n.onnx
YOLO_CONFIDENCE=0.35
YOLO_IOU=0.7
TRACKER=bytetrack                # bytetrack | botsort
DB_URL=sqlite:///data/speed.db
STREAM_MAX_FPS=15                # cap pushed to browsers
LOG_LEVEL=INFO
OLLAMA_ENABLED=false             # set true to enable LLM incident reports
OLLAMA_MODEL=qwen2.5:3b
```

No magic constants in code. The legacy `settings.json` road-config format is replaced
by per-camera rows in SQLite, editable via the UI.

---

## 8. Testing strategy

| Layer | Tool | What we assert |
|---|---|---|
| Geometry (`calibrator`, `speed`) | pytest, pure unit | A known 4-point transform + known Δt yields exactly the expected m/s. No video needed. |
| Tracker | pytest + `supervision` fakes | Synthetic `Detection` streams produce expected `Track` IDs and lifetimes. |
| Pipeline | pytest + synthetic video | `synthetic_video.py` renders deterministic moving rectangles; assert events + speeds. |
| API | `httpx.AsyncClient` + `pytest-asyncio` | REST endpoints round-trip; WebSocket pushes at least one frame. |
| E2E | `docker compose` + a fake RTSP server (`mediamtx`) | 2 cameras, 60s, assert SQLite has events and `/healthz` is green. |

Coverage gate: ≥80% on `src/speed_detector/pipeline/` and `src/speed_detector/api/`.
The `web/` frontend gets `vitest` smoke tests for the two most important components.

---

## 9. Deployment artifacts

### Dockerfile (CPU, multi-stage)
- **Stage 1 (builder):** python:3.12-slim, install uv, `uv sync --frozen`, build `web/`
  with Node, export YOLO to ONNX if only PyTorch weights are vendored.
- **Stage 2 (runtime):** python:3.12-slim, copy `site-packages` + `web/dist` +
  `models/`, install `onnxruntime`. Non-root user. ~600MB. No PyTorch in the final
  image.

### docker-compose.yml
- `backend` — the FastAPI app + CV threads
- `frontend` — only in dev (Vite proxy); in prod the backend serves `web/dist`
- `ollama` (optional, `profiles: [llm]`) — pulled only with `--profile llm`
- `mediamtx` (optional, `profiles: [test]`) — fake RTSP for E2E

### CI (`.github/workflows/ci.yml`)
Jobs: `lint` (ruff + mypy), `test` (pytest + coverage), `build` (docker build, no
push), `frontend` (vitest + tsc). All run on every PR. Image publish to GHCR only on
tagged releases — and only after explicit approval, not on every push.

---

## 10. Performance budget (CPU laptop)

Target on a mid-range 2024+ laptop CPU, 640×640 input:

| Stage | Budget | Notes |
|---|---|---|
| Source decode | ~3 ms | OpenCV `VideoCapture` |
| Preprocess + ONNX infer | ~25–35 ms | `yolo11n` ONNX; OpenVINO can halve this |
| Tracker (ByteTrack) | ~1–2 ms | pure numpy |
| Calibrate + speed + annotate | ~3 ms | |
| **Total per frame** | **~35–45 ms** | **~22–28 FPS** |

If real hardware underperforms, we degrade gracefully: cap `STREAM_MAX_FPS` for the
browser push, but keep the CV core running at full rate for event logging. The UI shows
the true CV FPS so we never fake the number.

---

## 11. Open questions (resolve before the corresponding phase)

1. **Sample video licensing.** Need a royalty-free traffic clip with a permissive
   license to bundle in `samples/`. Candidate: Pexels/Pixabay traffic footage. Verify
   the license explicitly in Phase 2.
2. **Calibration ground-truth.** For the portfolio demo, do we want a second sample
   video with a known road length and a calibration that yields ~correct mph, so the
   numbers look credible? Recommend yes.
3. **Ollama model size.** `qwen2.5:3b` (~2GB) vs `llama3.2:3b` vs `qwen2.5:1.5b`
   (~1GB). Decide based on RAM budget on the target laptop during Phase 5.
4. **Map view.** `leaflet` is in the stack list but adds scope. Likely defer to a
   "nice-to-have" after Phase 5 unless multi-camera is a stated selling point.

These are flagged so they don't become silent scope creep; each gets a decision before
its phase begins.
