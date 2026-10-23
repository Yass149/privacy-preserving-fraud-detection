"""Minimal, self-contained example of loading this project's trained fraud
detector and scoring a transaction window. This is the file to hand to someone
who wants to *use* the model rather than reproduce the research.

Requires only `torch` and `model_arch.py` (in this same folder) -- no Streamlit,
no Opacus, no cuDF, no GPU.

    python load_model.py

Read MODEL_CARD.md first: it documents the input format, the privacy guarantee,
and -- importantly -- the limitations that make this a triage aid rather than an
automated decision-maker.
"""

import os
import sys

import torch

import sys as _sys
from pathlib import Path as _Path

# This module now lives in src/. Checkpoints live in results/checkpoints/, and
# paths.py at the repository root is the single place that knows that.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as _paths

from model_arch import FraudDetector

DEFAULT_CHECKPOINT = str(_paths.CHECKPOINTS / "fraud_detector_oversampled.pth")

# Fixed by the Elliptic dataset and this project's architecture; a checkpoint
# will not load into a model built with different values.
INPUT_DIM = 165    # features per transaction (time_step column already removed)
WINDOW_SIZE = 5    # transactions per window, chronologically ordered, oldest first
LATENT_DIM = 32
HIDDEN_DIM = 64


def load_fraud_detector(checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
    """Load the trained model, ready for inference.

    Handles two details that trip people up:

    1. Opacus prefixes every parameter name with `_module.` during private
       training. Those prefixes are stripped here so the weights fit the plain
       architecture.
    2. The checkpoint also carries Opacus's internal DPGRU parameter aliases
       alongside the standard nn.GRU names. Those extras are inert duplicates,
       so `strict=False` is correct here -- but any *missing* key is a real
       problem and raises rather than silently producing a half-random model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train it with hpc/train.py "
            f"(see README.md), or copy an existing checkpoint next to this script."
        )

    model = FraudDetector(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)

    # weights_only=True refuses to execute arbitrary pickled code -- important
    # because checkpoints get emailed and copied between machines.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cleaned = {k[len("_module."):] if k.startswith("_module.") else k: v
               for k, v in state_dict.items()}

    missing, _unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is missing weights for {missing}. "
            f"Those layers would run on random initialisation, so predictions would be "
            f"meaningless. Refusing to load."
        )

    model.to(device)
    model.eval()   # also switches the encoder to its deterministic path
    return model


@torch.no_grad()
def predict_fraud_probability(model, window):
    """Score one or more transaction windows.

    Args:
        model:  a model returned by load_fraud_detector()
        window: float tensor shaped (5, 165) for a single window, or
                (batch, 5, 165) for several. Transactions must be in
                chronological order, oldest first.

    Returns:
        A 1-D tensor of P(Fraud), one probability per window.
    """
    if window.dim() == 2:
        window = window.unsqueeze(0)
    if window.shape[1:] != (WINDOW_SIZE, INPUT_DIM):
        raise ValueError(
            f"Expected each window shaped ({WINDOW_SIZE}, {INPUT_DIM}), "
            f"got {tuple(window.shape[1:])}."
        )

    logits = model(window.float())
    return torch.softmax(logits, dim=1)[:, 1]


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT

    print(f"Loading model from '{checkpoint}'...")
    model = load_fraud_detector(checkpoint)
    print("Loaded. Encoder is in deterministic eval mode.\n")

    # Random stand-in data purely to demonstrate the call signature and shapes.
    # Real use means real Elliptic feature vectors -- see MODEL_CARD.md.
    demo_batch = torch.randn(3, WINDOW_SIZE, INPUT_DIM)
    probabilities = predict_fraud_probability(model, demo_batch)

    print(f"Scored {len(probabilities)} example windows (random demo data):")
    for i, p in enumerate(probabilities.tolist()):
        print(f"  window {i}: P(fraud) = {p:.4f}")

    print("\nDeterminism check (same input scored twice):")
    again = predict_fraud_probability(model, demo_batch)
    print(f"  identical both times: {torch.equal(probabilities, again)}")
