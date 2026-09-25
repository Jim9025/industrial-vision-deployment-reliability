# Reproducibility

The public checkout builds against separately installed OpenCV and ONNX Runtime
SDKs. Run the root README's CMake and CTest commands. Without the separately
held model and frozen fixtures, CTest runs the core `aoi_tests` only. The CSV
synthetic example exercises the command-line flow; its scores were assigned
solely to create PASS and NG branches.

`results/parity_summary.json`, `benchmark_summary.json`, and
`batch_summary.json` are selected fields from verified local result artifacts.
They are published as evidence summaries, not as independently reproducible
public benchmarks. The local parity check compared an independent Python
reference with C++/ONNX for the same saved model and frozen threshold; it did
not measure defect-detection accuracy. No retraining or thesis experiment is
part of the public build.
