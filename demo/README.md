# C++ inspection demo

`aoi_demo` is a C++17/OpenCV/ONNX Runtime command-line inspector. Its `ScoreProvider`
interface supports CSV reference scores, a generic ONNX provider, and the
EfficientAD-S ONNX anomaly-map provider. The pipeline emits reports and, when
enabled, heatmap overlays. See the root [README](../README.md) for verified
metrics, build/run/test commands, and limitations.

The synthetic CSV sample verifies software flow only. The EfficientAD-S model,
third-party SDKs, and dataset are not redistributed.
