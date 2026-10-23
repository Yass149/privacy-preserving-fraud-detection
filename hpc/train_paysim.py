"""
PaySim training pipeline -- tests whether the findings from the Elliptic
experiments (train.py / train_baseline.py) generalize to a second, very
differently-structured dataset (Objective 1 of the spec form: PaySim was
promised as a second validation dataset, and until now only its data
pipeline and scalability had been exercised, never a trained model).

Key differences from the Elliptic pipeline, and why:
- PaySim transactions are independent events, not graph nodes with
  precomputed neighbourhood features, so there is no analogous "5-step
  temporal motif" to construct. window_size=1 is used deliberately --
  forcing the Elliptic-style window onto this dataset would be an
  unjustified architectural choice, not a real methodological parallel.
- input_dim is derived from the data at runtime rather than hardcoded,
  since PaySim's one-hot-encoded feature count differs from Elliptic's 165.
- PaySim's fraud rate is far more extreme (~0.13% vs ~6.5% for Elliptic),
  so the minority_oversample_factor that worked for Elliptic (5x) is very
  unlikely to be enough here -- pass a much larger value.

One script, three configurations via flags, matching train.py/
train_baseline.py's metrics JSON schema so visualize.py-style tooling can
treat these runs the same way:
  baseline:        --no_dp
  DP, naive:        --dp  (default weight=10x, minority_oversample_factor=1.0)
  DP, oversampled:   --dp --minority_oversample_factor <N>
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

from model_arch import FraudDetector
from data_loader import TransactionWindowDataset
from paysim_loader import load_paysim_data_with_time
from reproducibility import DEFAULT_SEED, set_seed


def train_paysim(csv_path, use_dp=True, noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5,
                  epochs=5, minority_oversample_factor=1.0, train_quantile=0.7,
                  output_dir=".", tag=None, seed=DEFAULT_SEED):
    print(f"--- Initiating PaySim {'DP-SGD' if use_dp else 'Baseline (No DP)'} Training ---")
    # Seed before any model construction or data shuffling. Matters more here
    # than on Elliptic: paysim_loader now sorts rows chronologically, so batch
    # composition depends on both that ordering and the shuffle RNG.
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_features, labels, steps, feature_cols = load_paysim_data_with_time(csv_path)
    input_dim = raw_features.shape[1]

    # Chronological split (mirrors the Elliptic pipeline's leakage-free
    # design -- PaySim's 'step' field is exactly analogous to time_step).
    # Use a quantile of the observed step values rather than a hardcoded
    # cutoff, since the exact max step present isn't assumed in advance.
    cutoff = torch.quantile(steps.float(), train_quantile).item()
    print(f"Chronological split at step <= {cutoff:.0f} (train) / > {cutoff:.0f} (test)")
    train_mask = steps <= cutoff
    test_mask = steps > cutoff

    WINDOW_SIZE = 1  # see module docstring: no per-entity temporal motif exists in PaySim
    X_train = raw_features[train_mask]
    y_train = labels[train_mask]
    X_test = raw_features[test_mask]
    y_test = labels[test_mask]

    # Unlike Elliptic's 165 features (pre-normalized in the released
    # dataset), PaySim's raw columns are real monetary values spanning huge
    # ranges. Feeding these unnormalized into the encoder causes numerical
    # divergence (observed: NaN training loss). Standardize using TRAIN-set
    # statistics only, computed before oversampling duplicates any rows, to
    # avoid both train/test leakage and oversampling-skewed statistics.
    feat_mean = X_train.mean(dim=0, keepdim=True)
    feat_std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    train_dataset = TransactionWindowDataset(X_train, y_train, window_size=WINDOW_SIZE)
    test_dataset = TransactionWindowDataset(X_test, y_test, window_size=WINDOW_SIZE)

    if minority_oversample_factor > 1.0:
        window_targets = y_train[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(train_dataset)]
        fraud_idx = torch.nonzero(window_targets == 1, as_tuple=True)[0].cpu().tolist()
        safe_idx = torch.nonzero(window_targets == 0, as_tuple=True)[0].cpu().tolist()
        n_repeats = int(round(minority_oversample_factor))
        oversampled_idx = safe_idx + fraud_idx * n_repeats
        minority_share = (len(fraud_idx) * n_repeats) / len(oversampled_idx) if oversampled_idx else 0.0
        print(f"Oversampling minority class {n_repeats}x: {len(fraud_idx)} fraud rows -> "
              f"{len(fraud_idx) * n_repeats}, {len(safe_idx)} safe rows unchanged. "
              f"New train set size: {len(oversampled_idx)} (was {len(train_dataset)}), "
              f"minority share: {minority_share:.2%}")
        train_dataset = Subset(train_dataset, oversampled_idx)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    print(f"Train rows: {len(train_dataset)} | Test rows: {len(test_dataset)} | input_dim: {input_dim}")

    model = FraudDetector(input_dim=input_dim, latent_dim=32, hidden_dim=64).to(device)

    # ModuleValidator.fix() can REPLACE incompatible layers (e.g. swap the
    # GRU for a DP-compatible reimplementation), returning a new model
    # object. It must run before the optimizer is created, or the optimizer
    # ends up holding references to the pre-fix model's parameters --
    # exactly the "Module parameters are different than optimizer
    # Parameters" error Opacus raises if this order is wrong.
    if use_dp:
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    weights = torch.tensor([1.0, 10.0 if minority_oversample_factor <= 1.0 else 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    privacy_engine = None
    if use_dp:
        from opacus import PrivacyEngine
        print("Attaching Opacus Differential Privacy Engine...")
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
        )

    epoch_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        record = {"epoch": epoch + 1, "train_loss": avg_loss}
        if use_dp:
            epsilon = privacy_engine.get_epsilon(delta=delta)
            record["epsilon"] = epsilon
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | epsilon: {epsilon:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
        epoch_history.append(record)

    final_epsilon = privacy_engine.get_epsilon(delta=delta) if use_dp else None

    model.eval()
    all_preds, all_labels_out = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_out.extend(batch_labels.cpu().numpy())

    report_text = classification_report(all_labels_out, all_preds, target_names=['Safe', 'Fraud'], zero_division=0)
    report_dict = classification_report(all_labels_out, all_preds, target_names=['Safe', 'Fraud'], zero_division=0, output_dict=True)
    print("\nFinal PaySim Classification Report:")
    print(report_text)

    run_type = "dp_sgd" if use_dp else "baseline_no_dp"
    # Encode the config that actually distinguishes runs (oversample factor)
    # into the filename itself, not just --tag -- otherwise two runs with
    # the same tag but different minority_oversample_factor silently
    # overwrite each other (this happened: a factor=1.0 run's output was
    # clobbered by a later factor=200 run, both tagged "v1").
    config_suffix = f"os{minority_oversample_factor:g}"
    name_parts = [f"paysim_{'dp' if use_dp else 'baseline'}", config_suffix] + ([tag] if tag else [])
    default_name = "_".join(name_parts)
    weights_path = os.path.join(output_dir, f"fraud_detector_{default_name}.pth")
    metrics_path = os.path.join(output_dir, f"training_metrics_{default_name}.json")

    # Defense in depth on top of the config-in-filename fix above.
    for path in (weights_path, metrics_path):
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists -- refusing to overwrite it silently. "
                f"Use a different --tag, or delete the file first if this is intentional."
            )

    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    metrics = {
        "run_type": run_type,
        "dataset": "paysim",
        "seed": seed,
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "window_size": WINDOW_SIZE,
        "train_quantile": train_quantile,
        "chronological_cutoff_step": cutoff,
        "noise_multiplier": noise_multiplier if use_dp else None,
        "max_grad_norm": max_grad_norm if use_dp else None,
        "delta": delta if use_dp else None,
        "final_epsilon": final_epsilon,
        "minority_oversample_factor": minority_oversample_factor,
        "class_weights": weights.cpu().tolist(),
        "epoch_history": epoch_history,
        "train_rows": len(train_dataset),
        "test_rows": len(test_dataset),
        "classification_report": report_dict,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str,
                         default="/scratch5/ic800171/thesis_project/elliptic_data/paysim.csv")
    parser.add_argument("--dp", dest="use_dp", action="store_true", default=True)
    parser.add_argument("--no_dp", dest="use_dp", action="store_false")
    parser.add_argument("--noise_multiplier", type=float, default=1.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minority_oversample_factor", type=float, default=1.0)
    parser.add_argument("--train_quantile", type=float, default=0.7,
                         help="Fraction of the step range used for training (chronological split).")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Random seed for reproducibility. Logged into the metrics JSON so any "
                              "reported run can be reproduced exactly on the same environment.")
    args = parser.parse_args()

    train_paysim(csv_path=args.csv_path, use_dp=args.use_dp, noise_multiplier=args.noise_multiplier,
                 max_grad_norm=args.max_grad_norm, delta=args.delta, epochs=args.epochs,
                 minority_oversample_factor=args.minority_oversample_factor,
                 train_quantile=args.train_quantile, output_dir=args.output_dir, tag=args.tag,
                 seed=args.seed)
