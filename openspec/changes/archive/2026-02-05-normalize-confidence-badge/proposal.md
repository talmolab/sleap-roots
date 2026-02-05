# Normalize Confidence Badge Display

## Why

SLEAP prediction scores are raw model outputs (log-likelihoods), not probabilities.
Values regularly exceed 1.0, making the badge display confusing and the color thresholds
(>0.8 green, >0.5 yellow) meaningless. Users cannot interpret what the number means or
compare scores across scans.

Additionally, the badge uses a green/yellow/red color scheme that is inconsistent with
the viridis colormap used in the confidence overlay, creating two visual languages
for the same concept.

## What Changes

- Normalize confidence scores to 0-1 range across all scans in the dataset
  - Uses global min/max so scans remain comparable to each other
- Use the same viridis colormap for the badge background color as the root overlays
  - Removes arbitrary green/yellow/red thresholds
  - Perceptually uniform, colorblind-friendly, one visual language
- Add a label ("Score:") and tooltip explaining what the score represents
- Update both the overview badge and frame-level stats display

## Impact

- Affected specs: `html-prediction-viewer` (MODIFIED confidence display requirement)
- Affected code: `sleap_roots/viewer/generator.py`, `sleap_roots/viewer/templates/viewer.html`

## Design

### Normalization Strategy

Two-pass approach in `generate()`:

1. **First pass**: render frames and collect all raw confidence scores
2. **After rendering**: compute global min/max across all scans
3. **Normalize**: convert all raw scores to 0-1 using `normalize_confidence()`
4. **Update template data**: store normalized values for badge and stats

This preserves relative ordering between scans (a scan with higher raw scores
will still have a higher normalized score) while keeping everything in 0-1.

### Viridis Badge Color

The badge background color is computed from the normalized score using the viridis
colormap. The generator pre-computes the RGB hex color and passes it to the template
so no JavaScript colormap library is needed.

```python
import matplotlib.pyplot as plt
cmap = plt.get_cmap("viridis")
rgba = cmap(normalized_score)
hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
```

### Label and Tooltip

The badge displays as `Score: 0.82` with a title attribute:
"Normalized prediction confidence (0=low, 1=high)"