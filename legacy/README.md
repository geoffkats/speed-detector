# Legacy implementation (2019)

The original 2019 OpenCV/KNN speed detector is preserved here for reference:

- `main.py` — original entry point: KNN background subtraction, contour
  detection, manual centroid tracking, and the "distance mode" vs "average mode"
  speed heuristics.
- `vehicle_counter.py` — hand-rolled `Vehicle` and `VehicleCounter` classes
  (the "ghost tracks" TODO the author noted in comments lives here).
- `threadedcam.py` — an abandoned threaded-camera attempt, never wired into
  `main.py`.
- `settings.json` — per-road config for the (now long-dead) Caltrans `.m3u8`
  streams.
- `demo.gif` — the original demo recording.
- `__init__.py` — vestigial empty file from the old flat layout.

**This code is not imported by the rewritten `speed_detector` package.** It
exists only so the before/after comparison stays visible. See
[`../PLAN.md`](../PLAN.md) for why it was rewritten and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the new structure.
