"""Shared pytest fixtures and import shims.

The HPC modules import `cudf`, which only exists on a GPU node with RAPIDS
installed. Everything in those modules that is worth testing -- the windowing
arithmetic, the split logic, the ordering rules -- is pure PyTorch or pure
Python and does not touch cuDF at all. Rather than skip those tests everywhere
except the cluster, cuDF is stubbed at import time so the modules load and their
real logic can be exercised on any machine, including CI.

Any test that genuinely needs cuDF behaviour is marked `requires_gpu` and
skipped here.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make both the repo root and hpc/ importable. hpc/ scripts import their
# siblings as top-level modules ("from data_loader import ..."), which is how
# they run on the cluster.
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "hpc"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Stub cuDF before any hpc module is imported. MagicMock is sufficient: the
# tested code paths never call into it.
if "cudf" not in sys.modules:
    sys.modules["cudf"] = MagicMock()

# Opacus is imported by the training scripts but not needed for the logic under
# test; stub it only if genuinely absent so a real install is preferred.
try:
    import opacus  # noqa: F401
except ImportError:  # pragma: no cover - depends on the machine
    opacus_stub = types.ModuleType("opacus")
    opacus_stub.PrivacyEngine = MagicMock()
    validators = types.ModuleType("opacus.validators")
    validators.ModuleValidator = MagicMock()
    sys.modules["opacus"] = opacus_stub
    sys.modules["opacus.validators"] = validators


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gpu: needs a CUDA GPU with RAPIDS/cuDF")
    config.addinivalue_line("markers", "requires_artifacts: needs trained checkpoints or metrics JSON")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def active_checkpoint(repo_root):
    path = repo_root / "results" / "checkpoints" / "fraud_detector_oversampled.pth"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return path


@pytest.fixture(scope="session")
def active_metrics(repo_root):
    import json
    path = repo_root / "results" / "metrics" / "training_metrics_oversampled.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text())


@pytest.fixture
def model():
    """A freshly initialised FraudDetector in eval mode."""
    import torch
    from model_arch import FraudDetector
    torch.manual_seed(0)
    m = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)
    m.eval()
    return m
