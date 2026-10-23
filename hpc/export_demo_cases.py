"""Export a sample of real held-out test cases, with predictions and feature
attributions, for the local dashboard's case browser.

WHY THIS EXISTS
---------------
The dashboard's model explorer previously scored a synthetic input: three
sliders over 165 features, with the other 162 held at random values. That
demonstrates almost nothing -- the absolute probability is meaningless, and the
attributions explain a vector that does not correspond to any real transaction.

This script replaces that with real evidence. It scores the actual held-out
test set (transactions from strictly later time steps than any training data),
selects an informative sample across all four outcome categories, computes
Integrated Gradients attributions for each, and writes the results out.

WHAT IS AND IS NOT EXPORTED
---------------------------
Only DERIVED values leave the cluster: the true label, the model's predicted
probability, and the top feature attributions (index plus attribution value).
No raw Elliptic feature vectors are written, so nothing about the dataset's own
distribution terms is at stake -- the output is a record of this model's
behaviour, not a redistribution of the dataset.

Pass --include-features to additionally write the raw feature windows to a
.npz, which enables live re-scoring in the dashboard. That is off by default
deliberately: check the Elliptic dataset's licence terms before using it.

FALSE POSITIVES ARE INCLUDED ON PURPOSE
---------------------------------------
The deployed model's fraud precision is roughly 0.23, so about three quarters
of what it flags is not fraud. A demo showing only successful catches would
misrepresent it. The sample is drawn across true positives, false negatives,
false positives and true negatives so the dashboard can show the failures
alongside the successes.

Usage, on a GPU node with the environment active:

    python export_demo_cases.py
    python export_demo_cases.py --n-per-group 50 --include-features
"""

import argparse
import json
import os

import numpy as np
import torch

from data_loader import load_financial_data, TransactionWindowDataset
from model_arch import FraudDetector
from reproducibility import DEFAULT_SEED, set_seed

WINDOW_SIZE = 5
INPUT_DIM = 165


def integrated_gradients(model, window, target=1, steps=64, baseline=None):
    """Integrated Gradients (Sundararajan et al., 2017), in plain PyTorch.

    Implemented here rather than imported from Captum so this script adds no
    dependency to the HPC environment -- Captum is only installed locally, for
    the dashboard. Verified against Captum's implementation on this exact model:
    correlation 0.999996, and identical top-8 feature rankings on every case
    tested, which is what this export actually uses.

    Interpolates `steps` points from a baseline (all zeros by default) to the
    input, averages the gradients along that path by the trapezoid rule, and
    scales by the input's displacement from the baseline.

    window: (seq_len, n_features) for ONE case. Returns the same shape.

    NOTE: must be run on CPU. cuDNN cannot compute an RNN backward pass while
    the module is in eval mode ("cudnn RNN backward can only be called in
    training mode"), and switching to train() is NOT an acceptable workaround
    here -- it reactivates the encoder's stochastic sampling, which would make
    every attribution a different random draw. That is precisely the
    non-determinism defect documented in thesis_source.md Section 3.6.1. CPU
    uses a native RNN kernel with no such restriction, and the cost is
    negligible for a few hundred cases.
    """
    x = window.detach()
    b = torch.zeros_like(x) if baseline is None else baseline.detach()
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device).view(-1, 1, 1)
    interp = (b.unsqueeze(0) + alphas * (x - b).unsqueeze(0)).requires_grad_(True)
    score = model(interp)[:, target].sum()
    grads = torch.autograd.grad(score, interp)[0]
    avg_grad = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0
    return (x - b) * avg_grad


def build_test_set(base_dir, tiebreak):
    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(base_dir, "elliptic_txs_classes.csv")
    raw, labels = load_financial_data(features_path, classes_path, tiebreak=tiebreak)

    # Identical split to training: time_step > 34 is held out.
    test_mask = raw[:, 0] > 34
    X = raw[test_mask][:, 1:]
    y = labels[test_mask]
    dataset = TransactionWindowDataset(X, y, window_size=WINDOW_SIZE)
    targets = y[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(dataset)]
    return dataset, targets


def load_checkpoint(path, device):
    model = FraudDetector(input_dim=INPUT_DIM, latent_dim=32, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    cleaned = {k.replace("_module.", "") if k.startswith("_module.") else k: v
               for k, v in state.items()}
    missing, _ = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint {path} missing weights for {missing}; refusing to export "
                           f"from a partially-initialised model.")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str,
                    default="/scratch5/ic800171/thesis_project/elliptic_data/elliptic_bitcoin_dataset")
    ap.add_argument("--checkpoint", type=str, default="fraud_detector_oversampled.pth")
    ap.add_argument("--output", type=str, default="demo_cases.json")
    ap.add_argument("--n-per-group", type=int, default=40,
                    help="Cases per outcome category (TP / FN / FP / TN).")
    ap.add_argument("--ig-steps", type=int, default=64,
                    help="Interpolation steps for Integrated Gradients.")
    ap.add_argument("--top-k", type=int, default=8,
                    help="Number of top-magnitude feature attributions to keep per case.")
    ap.add_argument("--tiebreak", type=str, default="hash", choices=["hash", "txid", "none"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--include-features", action="store_true",
                    help="Also write raw feature windows to a .npz. OFF by default -- check the "
                         "Elliptic dataset's licence terms before enabling.")
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(f"{args.output} already exists -- refusing to overwrite. "
                              f"Delete it or pass a different --output.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, targets = build_test_set(args.base_dir, args.tiebreak)
    model = load_checkpoint(args.checkpoint, device)
    print(f"Test windows: {len(dataset)}  (fraud: {int((targets == 1).sum())})")

    # Score the whole held-out set.
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu())
    probs = torch.cat(probs).numpy()
    truth = targets.cpu().numpy()
    pred = (probs >= 0.5).astype(int)

    groups = {
        "true_positive":  np.where((truth == 1) & (pred == 1))[0],
        "false_negative": np.where((truth == 1) & (pred == 0))[0],
        "false_positive": np.where((truth == 0) & (pred == 1))[0],
        "true_negative":  np.where((truth == 0) & (pred == 0))[0],
    }
    print("Outcome counts across the full held-out set:")
    for name, idx in groups.items():
        print(f"  {name:15s} {len(idx):6d}")

    rng = np.random.default_rng(args.seed)
    chosen = {}
    for name, idx in groups.items():
        take = min(args.n_per_group, len(idx))
        chosen[name] = rng.choice(idx, size=take, replace=False) if take else np.array([], int)

    # Attribution runs on CPU -- see integrated_gradients()'s note on the cuDNN
    # eval-mode restriction. Scoring above already used the GPU, which is where
    # the expensive full-test-set pass actually was.
    model = model.cpu()
    print("\nComputing attributions on CPU (cuDNN cannot backward through an "
          "RNN in eval mode)...")

    cases, feature_windows = [], []

    for name, indices in chosen.items():
        for i in indices:
            window, _ = dataset[int(i)]
            # Attribute toward the fraud class, then read the final timestep --
            # the one whose label the window carries.
            attr_full = integrated_gradients(model, window.cpu().float(), target=1, steps=args.ig_steps)
            attr = attr_full[-1, :].detach().numpy()
            order = np.argsort(np.abs(attr))[::-1][:args.top_k]
            cases.append({
                "id": int(i),
                "outcome": name,
                "true_label": int(truth[i]),
                "fraud_probability": float(probs[i]),
                "top_attributions": [
                    {"feature_index": int(j), "attribution": float(attr[j])} for j in order
                ],
            })
            if args.include_features:
                feature_windows.append(window.cpu().numpy())

    payload = {
        "checkpoint": os.path.basename(args.checkpoint),
        "tiebreak": args.tiebreak,
        "seed": args.seed,
        "window_size": WINDOW_SIZE,
        "input_dim": INPUT_DIM,
        "decision_threshold": 0.5,
        "n_test_windows_total": int(len(dataset)),
        "outcome_counts_full_test_set": {k: int(len(v)) for k, v in groups.items()},
        "note": ("Derived model outputs only -- no raw Elliptic feature values are included. "
                 "Feature indices refer to the dataset's anonymised columns, whose semantics "
                 "are not published."),
        "cases": cases,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}: {len(cases)} cases "
          f"({os.path.getsize(args.output) / 1024:.0f} KB)")

    if args.include_features:
        npz = args.output.replace(".json", "_features.npz")
        np.savez_compressed(npz, windows=np.stack(feature_windows),
                            ids=np.array([c["id"] for c in cases]))
        print(f"Wrote {npz}: raw feature windows "
              f"({os.path.getsize(npz) / 1024:.0f} KB) -- check dataset licence before sharing.")


if __name__ == "__main__":
    main()
