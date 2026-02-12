# Prediction Viewer

Generate an interactive HTML viewer to visually validate SLEAP predictions before computing traits.

## Quick Start

```bash
sleap-roots viewer /path/to/predictions --output viewer.html
```

Open `viewer.html` in any browser to review predictions with keyboard navigation.

## When to Use

The prediction viewer is useful for:

- **Model validation**: Check if predictions from a model trained on one species generalize to another
- **Quality control**: Quickly review prediction accuracy across many scans
- **Debugging**: Identify problematic frames or scans before batch processing
- **Sharing**: Create portable HTML reports for collaborators

## Output Modes

The viewer supports three output modes optimized for different use cases:

| Mode | Flag | Best For | Output |
|------|------|----------|--------|
| Client-render | *(default)* | Fast QC, local use | HTML + external images |
| Pre-rendered | `--render` | Sharing, offline | HTML + `viewer_images/` |
| Embedded | `--embed` | Single file sharing | Self-contained HTML |

### Client-Render Mode (Default)

Fastest generation. The HTML references external image files and draws prediction overlays in the browser using Canvas.

```bash
sleap-roots viewer predictions/ --output viewer.html
```

### Pre-Rendered Mode

Generates matplotlib-rendered overlay images saved to disk. Best for sharing when recipients may not have access to source images.

```bash
sleap-roots viewer predictions/ --output viewer.html --render
```

Creates `viewer.html` plus a `viewer_images/` directory.

### Embedded Mode

Embeds all images as base64 in a single HTML file. Largest file size but completely self-contained.

```bash
sleap-roots viewer predictions/ --output viewer.html --embed
```

## Image Sources

The viewer automatically detects image sources from the `.slp` prediction files:

| Source Type | Description | Auto-Detected? |
|-------------|-------------|----------------|
| Image directories | Folder of individual image files (.png, .jpg, etc.) | Yes |
| H5 video files | HDF5 files containing image stacks | Yes |

### When to Use `--images`

Use the `--images` option when:

- **Source images have moved**: The paths stored in `.slp` files no longer exist
- **Images are on a different drive**: The viewer can't resolve relative paths
- **Using a different image directory**: You want to overlay predictions on different images

```bash
# Images moved to a new location
sleap-roots viewer predictions/ --images /new/path/to/images

# Using images from a network drive
sleap-roots viewer predictions/ --images //server/share/images
```

The viewer will attempt to match image filenames from the `.slp` files to files in the `--images` directory.

## CLI Reference

```bash
sleap-roots viewer PREDICTIONS_DIR [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PREDICTIONS_DIR` | Directory containing `.slp` prediction files |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--images PATH` | *(auto-detect)* | Directory containing source images |
| `--output, -o PATH` | `viewer.html` | Output HTML file path |
| `--max-frames N` | `10` | Maximum frames per scan (0 = all) |
| `--no-limit` | `false` | Disable 1000 total frame limit |
| `--render` | `false` | Pre-render overlays to disk |
| `--embed` | `false` | Embed images as base64 |
| `--format FORMAT` | `jpeg` | Image format for `--render` (jpeg/png) |
| `--quality N` | `85` | JPEG quality 1-100 for `--render` |
| `--zip` | `false` | Create ZIP archive for sharing |
| `--timepoint PATTERN` | *(none)* | Filter scans by timepoint pattern (e.g., `Day0*`) |

### ZIP Archive Portability

The `--zip` option creates a portable archive, but behavior depends on your source data:

| Source Type | Mode | Result |
|-------------|------|--------|
| Image directories | Client-render (default) | ZIP includes HTML + images folder |
| H5 video files | `--render --zip` | ZIP includes HTML + rendered images |
| H5 video files | `--embed --zip` | ZIP includes self-contained HTML |
| H5 video files | `--zip` alone | ⚠️ Not portable (references H5 file) |

For H5 video sources, always use `--render --zip` or `--embed --zip` to create a shareable archive.

## Keyboard Navigation

### Overview Grid

| Key | Action |
|-----|--------|
| `←` `→` `↑` `↓` | Navigate between scan cards |
| `Enter` | Open selected scan |
| `C` | Toggle view mode (root type / confidence) |

### Frame View

| Key | Action |
|-----|--------|
| `←` `→` | Previous / next frame |
| `Esc` | Return to overview |
| `C` | Toggle view mode |

## View Modes

### Root Type View

Colors predictions by root type:

- **Blue**: Primary roots
- **Orange**: Lateral roots
- **Green**: Crown roots

### Confidence View

Colors predictions by confidence score using a viridis colormap (purple = low, yellow = high). Scores are normalized per-frame.

## Examples

### Basic Usage

Generate a viewer for pipeline output:

```bash
sleap-roots viewer experiment/predictions --output qc_viewer.html
```

### With Explicit Images Directory

When images are in a separate location:

```bash
sleap-roots viewer experiment/predictions \
    --images experiment/images \
    --output viewer.html
```

### All Frames with ZIP for Sharing

```bash
sleap-roots viewer predictions/ \
    --output full_viewer.html \
    --max-frames 0 \
    --no-limit \
    --embed \
    --zip
```

### High-Quality Pre-Rendered for Publication

```bash
sleap-roots viewer predictions/ \
    --output figures/viewer.html \
    --render \
    --format png \
    --quality 95
```

### Multi-Timepoint Experiments

For experiments that track plants across multiple days, use `--timepoint` to filter:

```bash
# View only Day0 scans
sleap-roots viewer predictions/ --images images/ --timepoint "Day0*"

# View multiple timepoints (OR logic)
sleap-roots viewer predictions/ --images images/ \
    --timepoint "Day0*" \
    --timepoint "Day3*"
```

The viewer automatically groups scans by timepoint (parent directory) and displays plant names prominently on scan cards.

## Multi-Timepoint Features

When working with experiments that scan the same plants across multiple days:

### Gallery Grouping

Scans are automatically grouped by timepoint (parent directory name). Each group shows a collapsible section with the timepoint name and scan count.

### Plant Name Display

Scan cards display the plant name (QR code / genotype) prominently instead of internal scan IDs. Hover over the plant name to see the scan ID in a tooltip. This makes it easy to identify the same plant across different timepoints.

### Timepoint Filtering

Use `--timepoint` to generate viewers for specific timepoints. Patterns are case-insensitive and support glob wildcards:

- `--timepoint "Day0*"` - Match Day0, Day0_2025-11-27, etc.
- `--timepoint "day*"` - Case-insensitive matching
- Multiple `--timepoint` flags use OR logic

## Tips

- **Start with defaults**: The default client-render mode with 10 frames per scan is fast and sufficient for most QC tasks
- **Use `--zip` for sharing**: Creates a portable archive that includes all required files
- **H5 videos need render mode**: For H5 video sources, use `--render --zip` to create portable archives
- **Toggle overlays**: In client-render mode, use the "Show Predictions" checkbox to compare raw images with overlays
- **Check confidence scores**: The confidence badge on each scan card shows normalized mean prediction quality
- **Filter by timepoint**: For large multi-timepoint datasets, use `--timepoint` to generate focused viewers