# Test scope

The original local checkout recorded three passing CTest entries: core `aoi_tests`,
`onnx_frozen_parity`, and `efficientad_frozen_parity`. The latter two require local
models and frozen fixtures, which are deliberately excluded from this public repo.
A clean public clone runs `aoi_tests`; its result should be checked locally after
building, rather than inferred from the original three-test record.
