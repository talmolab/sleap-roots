# Design: Fourier Shape Descriptors for Root Skeletons

## Context

sleap-roots extracts root skeleton coordinates as ordered (x, y) point sequences from SLEAP pose estimation. Current trait modules compute interpretable metrics (lengths, angles, hull properties), but researchers also need:
- Fixed-length embeddings for ML pipelines
- Shape representations that capture patterns beyond hand-crafted traits
- Compact encodings for clustering and similarity search

This design describes a lightweight Fourier descriptor module that produces shape embeddings from root polylines using established morphometric techniques.

## Goals / Non-Goals

**Goals:**
- Produce fixed-length shape embeddings from variable-length root polylines
- Work with open curves (root skeletons are not closed contours)
- Zero additional dependencies (NumPy only)
- Maintain consistency with existing module patterns
- Support both per-root and aggregated embeddings

**Non-Goals:**
- Neural network-based encoding (Poly2Vec, Geo2Vec) - requires training
- 3D shape analysis - roots are 2D projections
- Real-time processing optimization - batch processing is acceptable
- Replacing existing interpretable traits - this is complementary

## Decisions

### Decision 1: Tangent Angle Parameterization

**What:** Represent each root as θ(s), the tangent angle as a function of normalized arc length.

**Why:**
- Standard approach for open biological curves (C. elegans eigenworms)
- Naturally invariant to translation
- Rotation invariance achieved by subtracting mean angle
- Curvature is directly derivable: κ(s) = dθ/ds

**Alternatives considered:**
- Elliptic Fourier Descriptors: Requires closed contours; would need to artificially close root polylines
- Raw (x, y) Fourier: Not rotation-invariant without additional normalization
- Curvature-only: Loses directional information

### Decision 2: Fixed Harmonic Count

**What:** Return first N Fourier coefficients (default N=10, configurable).

**Why:**
- Low frequencies capture overall shape (straight vs curved)
- High frequencies capture local variations (wiggles)
- 10 harmonics is sufficient for ~98% shape reconstruction in similar biological applications
- Fixed length enables direct use as ML features

**Output format:**
```python
# For N harmonics, output is 2N floats:
# [magnitude_1, ..., magnitude_N, phase_1, ..., phase_N]
# or alternatively:
# [real_1, ..., real_N, imag_1, ..., imag_N]
```

### Decision 3: Arc-Length Resampling

**What:** Resample each root polyline to M uniformly-spaced points along arc length before computing tangent angles.

**Why:**
- SLEAP predictions have varying node counts and spacing
- Uniform sampling ensures consistent Fourier analysis
- Default M=100 provides smooth curves without excessive computation

**Implementation:**
```python
def get_arc_length_parameterization(points: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Resample polyline to uniform arc-length spacing."""
    # Compute cumulative arc length
    # Interpolate to uniform spacing
    # Return resampled (n_samples, 2) array
```

### Decision 4: Module-Level Functions (Not Class)

**What:** Implement as standalone functions following existing sleap-roots patterns.

**Why:**
- Consistent with `lengths.py`, `angle.py`, `tips.py` module structure
- Functions are stateless and composable
- Easier to integrate into pipeline `TraitDef` system

### Decision 5: NaN Handling

**What:** Filter NaN nodes before processing; return NaN-filled result for invalid roots.

**Why:**
- SLEAP predictions may have missing nodes
- Consistent with existing trait modules' NaN handling
- Downstream analysis can filter/impute as needed

## API Design

```python
# sleap_roots/fourier.py

def get_arc_length_parameterization(
    points: np.ndarray,
    n_samples: int = 100
) -> np.ndarray:
    """Resample polyline to uniform arc-length spacing.

    Args:
        points: Root skeleton points of shape (n_nodes, 2).
        n_samples: Number of uniformly-spaced samples to generate.

    Returns:
        Resampled points of shape (n_samples, 2).
    """

def get_tangent_angles(
    points: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute tangent angles along a polyline.

    Args:
        points: Polyline points of shape (n_points, 2).
        normalize: If True, subtract mean angle for rotation invariance.

    Returns:
        Tangent angles of shape (n_points - 1,) in radians.
    """

def get_curvature(points: np.ndarray) -> np.ndarray:
    """Compute curvature along a polyline.

    Args:
        points: Polyline points of shape (n_points, 2).

    Returns:
        Curvature values of shape (n_points - 2,).
    """

def get_fourier_coefficients(
    signal: np.ndarray,
    n_harmonics: int = 10
) -> np.ndarray:
    """Compute Fourier coefficients of a 1D signal.

    Args:
        signal: 1D signal array.
        n_harmonics: Number of harmonic coefficients to return.

    Returns:
        Array of shape (2 * n_harmonics,) containing
        [magnitudes..., phases...] or [real..., imag...].
    """

def get_fourier_descriptors(
    points: np.ndarray,
    n_samples: int = 100,
    n_harmonics: int = 10,
    include_curvature: bool = False
) -> np.ndarray:
    """Compute Fourier shape descriptors for a root polyline.

    This is the main API for generating fixed-length shape embeddings.

    Args:
        points: Root skeleton points of shape (n_nodes, 2).
        n_samples: Arc-length resampling resolution.
        n_harmonics: Number of Fourier harmonics to compute.
        include_curvature: If True, append curvature-based descriptors.

    Returns:
        Shape descriptor array of fixed length.
    """
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Fourier descriptors may not outperform existing traits | Position as complementary; validate on real phenotyping tasks |
| Computational cost for large batches | Profile and optimize; vectorize where possible |
| Parameter tuning (n_samples, n_harmonics) | Provide sensible defaults; document trade-offs |
| Interpretability loss vs hand-crafted traits | Document that this is for ML, not direct biological interpretation |

## Open Questions

1. **Output format**: Should coefficients be (magnitude, phase) or (real, imag)? Magnitude/phase may be more interpretable.

2. **Pipeline integration**: Should Fourier descriptors be first-class traits in pipelines, or a separate utility? Initial implementation as standalone functions; pipeline integration can follow if adoption is positive.

3. **Aggregation across roots**: For plants with multiple laterals, how should embeddings be aggregated? Options: per-root, mean pooling, concatenation. Defer to user for now.
