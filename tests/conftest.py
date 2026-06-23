import matplotlib

# Force the headless Agg backend process-wide before any module imports pyplot,
# so the circumnutation plotting tests (and save_plots) never open a GUI window
# or fail on a dev machine with an interactive backend — matching headless CI
# regardless of test-collection order.
matplotlib.use("Agg")

from tests.fixtures.data import *  # noqa: E402
