# Proposal: Add Fourier Shape Descriptors for Root Encoding

## Why

Current trait extraction provides ~40 interpretable morphological traits (lengths, angles, convex hull metrics, etc.), but these are hand-crafted measures that may miss predictive shape patterns not explicitly defined. Researchers need:

1. **ML-ready shape embeddings** for classification, clustering, and prediction tasks where existing traits may leave signal on the table
2. **Unsupervised phenotype discovery** to find root shape groups that don't map to predefined traits
3. **Compact shape representations** that capture the "full" morphology in a fixed-length vector

Fourier-based shape descriptors are well-established in biological morphometrics:
- Elliptic Fourier Descriptors (EFDs) are standard for cell, leaf, and organ shape analysis
- Tangent angle parameterization is used for worm/nematode pose encoding (C. elegans eigenworms)
- Curvature Scale Space (CSS) is an MPEG-7 standardized shape descriptor

However, most methods target **closed contours**. Root skeletons are **open polylines**, requiring an adapted approach similar to the tangent angle method used in C. elegans research.

## What Changes

- Add new `sleap_roots/fourier.py` module implementing Fourier shape descriptors for open polylines
- Implement arc-length parameterized tangent angle representation (proven approach from worm morphology)
- Compute Fourier coefficients of the tangent angle function to produce fixed-length embeddings
- Add curvature-based descriptors as complementary features
- Integrate as optional "traits" in existing pipeline architecture
- No external dependencies beyond NumPy (already required)

## Impact

**Affected specs:**
- `trait-computation` (new capability) - Fourier shape descriptor requirements

**Affected code:**
- `sleap_roots/fourier.py` - New module for Fourier descriptors
- `sleap_roots/__init__.py` - Export new functions
- `sleap_roots/trait_pipelines.py` - Optional integration into pipelines
- `tests/test_fourier.py` - Unit tests for new module

**Non-breaking:** This change adds new optional functionality; existing traits and pipelines are unchanged.

## Research Background

### Tangent Angle Parameterization (Primary Approach)

From C. elegans research (Stephens et al., PLOS Computational Biology):
- Parameterize curve by arc length `s` (normalized 0 to 1)
- Compute tangent angle `θ(s)` at each point
- Apply Fourier transform to `θ(s)` to get coefficients
- Low-frequency components capture overall shape; high-frequency capture local wiggles

This is ideal for root skeletons because:
- Works on open curves (not just closed contours)
- Invariant to position and overall orientation
- Naturally ordered from base to tip (like head to tail in worms)
- Curvature `κ(s) = dθ/ds` is directly derivable

### Curvature Scale Space (Complementary)

From MPEG-7 standardization:
- Compute curvature along the curve
- Multi-scale analysis captures both global and local shape features
- Robust to noise and small perturbations

### Why Not EFDs?

Classic Elliptic Fourier Descriptors require closed contours. While we could artificially close root polylines, this introduces artifacts. The tangent angle approach is mathematically cleaner for open curves.

## References

1. Stephens, G.J., et al. "Dimensionality and Dynamics in the Behavior of C. elegans." PLOS Computational Biology, 2008. DOI: 10.1371/journal.pcbi.1000028
2. Kuhl, F.P. & Giardina, C.R. "Elliptic Fourier features of a closed contour." Computer Graphics and Image Processing, 1982.
3. Mokhtarian, F. & Mackworth, A.K. "A theory of multiscale, curvature-based shape representation for planar curves." IEEE TPAMI, 1992.
4. Kriegel, F.L., et al. "Cell shape characterization and classification with discrete Fourier transforms." Cytometry Part A, 2018.
