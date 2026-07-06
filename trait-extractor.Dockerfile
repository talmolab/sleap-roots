# syntax=docker/dockerfile:1
#
# Service image for the `trait_extractor` batch driver. Ships the `sleap-roots` library +
# `sleap-roots-contracts` (the slim `extractor` extra) and the flat top-level `trait_extractor/`
# service package (excluded from the PyPI wheel, so COPY'd in rather than pip-installed).
#
# Distinct image identity: ghcr.io/talmolab/sleap-roots-trait-extractor (NOT the library's
# ghcr.io/talmolab/sleap-roots). Entry: `python -m trait_extractor <input_dir> <output_dir>`.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System libs. `build-essential` (gcc + libstdc++ + libgomp1) covers the scientific stack's
# runtime needs on the slim base; matplotlib runs headless via the Agg backend (no Tk/GL), and
# sleap_roots imports no OpenCV. Kept minimal — extend only if an in-image `import sleap_roots`
# fails to resolve a shared object.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency metadata + lockfile first so the (heavy) install layer caches across source
# edits. README.md is copied because the project declares a `dynamic` readme.
COPY pyproject.toml uv.lock README.md ./
# The `sleap-roots` project is built/installed by `uv sync` (setuptools resolves its dynamic
# version from `sleap_roots.__version__`), so its source MUST be present in the context.
COPY sleap_roots ./sleap_roots
# The service package is not pip-installable (excluded from the wheel); it is reached via
# PYTHONPATH below.
COPY trait_extractor ./trait_extractor

# Install the library + contracts (slim `extractor` extra) from the frozen lockfile, WITHOUT
# the dev/docs group. `--python 3.12` pins the interpreter: uv against a <3.11 interpreter would
# exit 0 while silently dropping `sleap-roots-contracts` (its `python_version >= '3.11'` marker).
RUN uv sync --frozen --no-dev --extra extractor --python 3.12

# Run from the baked venv (no runtime resolution), with the repo root on the path so the copied
# `trait_extractor` package imports (the editable install exposes only `sleap_roots`).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# Headless matplotlib: force the Agg backend and a writable cache dir (the slim base has no
# writable HOME and no Tk), so importing sleap_roots' plotting paths works in the container.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Bake the build commit so emitted envelopes carry a non-empty `provenance.traits_code_sha`
# (the idempotency key hashes it; the CLI has no flag to inject it, so a "" default would let a
# traits-code change silently collide with a prior result under A4's dedup). Placed AFTER the
# install/COPY layers so a per-commit SHA does not bust the dependency cache. A runtime env value
# passed by the Argo template overrides this. The container digest is NOT baked (unknown at build
# time, and not a key input) — it stays a runtime-supplied SRT_TRAITS_CONTAINER_DIGEST.
ARG SRT_TRAITS_CODE_SHA=""
ENV SRT_TRAITS_CODE_SHA=${SRT_TRAITS_CODE_SHA}

# The container entry command: `docker run <image> <input_dir> <output_dir>`. Exec form so the
# batch driver's process exit code propagates to the container (read by Argo).
ENTRYPOINT ["python", "-m", "trait_extractor"]
