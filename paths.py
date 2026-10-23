"""Canonical locations for every artefact this project reads or writes.

Before this existed, roughly eighty files sat at the repository root and their
locations were hardcoded as bare filenames across the dashboard, the tests, the
figure generators and the build scripts. Moving anything meant finding every
reference by hand.

Everything that needs a path now asks here. Moving a directory is a one-line
change in this file.

One deliberate exception: `hpc/` is not described here. Those scripts are copied
to the cluster and run from their own working directory, reading and writing
bare filenames with no knowledge of this repository's layout. That is the
contract with SLURM and this module must not try to change it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- results: everything produced by a training, sweep or attack run ---
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"          # training_metrics_*.json, scalability_metrics.json
ATTACKS = RESULTS / "attacks"          # mia_results*.json
CHECKPOINTS = RESULTS / "checkpoints"  # fraud_detector_*.pth
DEMO_CASES = RESULTS / "demo_cases.json"

# --- the dissertation and its build products ---
THESIS = ROOT / "thesis"
THESIS_SOURCE = THESIS / "thesis_source.md"
FIGURES = THESIS / "figures"           # fig_*.png / .jpg, referenced by the thesis
LATEX = THESIS / "latex"               # preamble.tex + the generated thesis.tex

# --- supporting material ---
DOCS = ROOT / "docs"
SRC = ROOT / "src"                     # model_arch, load_model, predict
HPC = ROOT / "hpc"                     # cluster bundle, self-contained
TESTS = ROOT / "tests"

# Where a bare artefact filename lives, by naming convention. The dashboard and
# the figure generators refer to results by name rather than by path, so the
# lookup happens once here instead of at every call site.
_BY_PREFIX = (
    ("mia_results", ATTACKS),
    ("training_metrics", METRICS),
    ("scalability_metrics", METRICS),
    ("fraud_detector", CHECKPOINTS),
    ("demo_cases", RESULTS),
)


def artefact(name):
    """Resolve a bare artefact filename to its full path.

    Accepts a path that is already resolved and returns it unchanged, so callers
    can pass either. Unknown names fall back to the repository root, which keeps
    the behaviour of a caller that has its own convention.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    for prefix, directory in _BY_PREFIX:
        if candidate.name.startswith(prefix):
            return directory / candidate.name
    return ROOT / candidate.name


def metrics_glob(pattern):
    """Sorted metrics files matching a bare glob such as 'training_metrics_nm*.json'."""
    return sorted(METRICS.glob(pattern))
