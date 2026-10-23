"""Architecture invariants.

Every test here guards a failure this project has actually suffered, or a
silent-drift risk the structure still carries. None of them are generic
"does it run" checks.
"""

import ast
from pathlib import Path

import pytest
import torch


def _code_fingerprint(path):
    """AST dump with all docstrings stripped, so comments and prose differ
    freely while any real code change shows up."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return ast.dump(tree)


def test_root_and_hpc_model_arch_have_identical_code(repo_root):
    """The architecture is defined twice -- root for the dashboard, hpc/ for
    training -- with nothing enforcing they match. If they drift, a checkpoint
    trained against one loads into the other with `strict=False` silently
    filling the difference with random weights, and predictions become
    meaningless with no error raised anywhere.

    Docstrings and comments are deliberately allowed to differ; only executable
    code is compared.
    """
    root = _code_fingerprint(repo_root / "src" / "model_arch.py")
    hpc = _code_fingerprint(repo_root / "hpc" / "model_arch.py")
    assert root == hpc, (
        "model_arch.py and hpc/model_arch.py have diverged in CODE. A checkpoint "
        "trained on one will not load correctly into the other."
    )


def test_encoder_is_deterministic_in_eval_mode(model):
    """The defect documented in thesis_source.md Section 3.6.1.

    The encoder samples from a learned distribution during training as a
    regulariser. That sampling must be gated behind `self.training`, or it stays
    active during evaluation -- which is exactly what happened: identical input
    returned fraud probabilities from 1.79% to 7.13% across calls, and because
    the same path ran in the test-set loop, every reported metric was a single
    non-reproducible draw.
    """
    x = torch.randn(1, 5, 165)
    model.eval()
    with torch.no_grad():
        outputs = [model(x) for _ in range(8)]
    for i, out in enumerate(outputs[1:], start=1):
        assert torch.equal(outputs[0], out), (
            f"eval() call {i} differs from call 0 -- the encoder is sampling "
            f"stochastically during evaluation. This is the Section 3.6.1 bug."
        )


def test_encoder_is_stochastic_in_train_mode(model):
    """The other half of the gate. If train() were also deterministic the
    regulariser would silently not be doing anything, so this asserts the gate
    works in both directions rather than that sampling was simply removed.
    """
    x = torch.randn(1, 5, 165)
    model.train()
    torch.manual_seed(1)
    a = model(x)
    torch.manual_seed(2)
    b = model(x)
    assert not torch.equal(a, b), (
        "train() produced identical output under different RNG state -- the "
        "encoder's stochastic sampling appears to have been removed entirely, "
        "not just gated."
    )


def test_output_shape_and_valid_probabilities(model):
    """Guards the contract MODEL_CARD.md and load_model.py both document:
    (batch, 5, 165) in, (batch, 2) logits out, ordered [Safe, Fraud]."""
    for batch in (1, 4):
        x = torch.randn(batch, 5, 165)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (batch, 2), f"expected ({batch}, 2), got {tuple(logits.shape)}"
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()


def test_rejects_wrong_feature_count(model):
    """165 features is fixed by the Elliptic dataset. Passing a different width
    must fail loudly rather than silently broadcasting or truncating."""
    with pytest.raises(Exception):
        with torch.no_grad():
            model(torch.randn(1, 5, 100))


@pytest.mark.requires_artifacts
def test_checkpoint_loads_with_no_missing_weights(active_checkpoint):
    """A checkpoint is loaded with strict=False, because Opacus writes extra
    DPGRU parameter aliases alongside the standard nn.GRU names. That is benign,
    but the same flag would also hide genuinely MISSING weights, leaving layers
    at random initialisation while the dashboard reports confident predictions.

    Missing keys must be empty. Unexpected keys are expected and allowed.
    """
    from model_arch import FraudDetector

    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)
    state = torch.load(active_checkpoint, map_location="cpu", weights_only=True)
    cleaned = {k.replace("_module.", "") if k.startswith("_module.") else k: v
               for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    assert not missing, (
        f"checkpoint is missing weights for {missing}; those layers would run on "
        f"random initialisation and predictions would be meaningless"
    )
    # Documents the expected Opacus aliasing rather than asserting it away.
    for key in unexpected:
        assert "gru" in key.lower(), (
            f"unexpected key {key!r} is not an Opacus GRU alias -- the checkpoint "
            f"may not match this architecture"
        )
