# Tasks: Add Fourier Shape Descriptors

## 1. Core Implementation

- [ ] 1.1 Create `sleap_roots/fourier.py` module with docstring header
- [ ] 1.2 Implement `get_arc_length_parameterization()` - resample polyline to uniform arc length
- [ ] 1.3 Implement `get_tangent_angles()` - compute tangent angle θ(s) along curve
- [ ] 1.4 Implement `get_fourier_coefficients()` - FFT of tangent angle function
- [ ] 1.5 Implement `get_curvature()` - compute curvature κ(s) = dθ/ds
- [ ] 1.6 Implement `get_fourier_descriptors()` - main API returning fixed-length embedding
- [ ] 1.7 Add normalization options (rotation-invariant, scale-invariant)

## 2. Integration

- [ ] 2.1 Add exports to `sleap_roots/__init__.py`
- [ ] 2.2 Create `TraitDef` entries for Fourier descriptors in pipeline (optional)
- [ ] 2.3 Add `fourier_embedding` as optional computed trait in pipelines

## 3. Testing

- [ ] 3.1 Create `tests/test_fourier.py` with unit tests
- [ ] 3.2 Test arc-length parameterization correctness
- [ ] 3.3 Test tangent angle computation on known shapes (straight line, semicircle)
- [ ] 3.4 Test Fourier coefficient properties (conjugate symmetry, Parseval's theorem)
- [ ] 3.5 Test invariance properties (translation, rotation, scale)
- [ ] 3.6 Test handling of edge cases (NaN values, short roots, single-point roots)
- [ ] 3.7 Integration test with real root data from test fixtures

## 4. Documentation

- [ ] 4.1 Add Google-style docstrings to all public functions
- [ ] 4.2 Add module-level docstring explaining the methodology
- [ ] 4.3 Update `docs/api/traits/` with Fourier descriptor documentation
- [ ] 4.4 Add usage examples in docstrings

## 5. Validation

- [ ] 5.1 Run Black formatting check
- [ ] 5.2 Run pydocstyle check
- [ ] 5.3 Run full test suite
- [ ] 5.4 Verify no regressions in existing traits
