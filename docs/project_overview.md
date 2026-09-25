# Project overview

The project is a small command-line reference for AOI-style image inspection.
`aoi_demo` enumerates images, decodes them with OpenCV, records acquisition
metrics, invokes a score provider, applies the configured frozen threshold and
quality gates, then writes per-image and batch reports. The EfficientAD-S path
uses ONNX Runtime and preserves a spatial anomaly map for optional overlays.
Errors are recorded per image instead of silently producing a PASS decision.

The demo shows software engineering and validation workflow. It is not a camera,
PLC, factory MES, or qualified defect disposition system.
