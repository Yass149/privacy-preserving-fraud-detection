"""Train the non-private Elliptic baseline. Runs on the GPU node.

The control condition for every DP result in the dissertation. Identical to
train.py in architecture, data pipeline, chronological split, windowing and
evaluation -- the ONLY difference is that no privacy engine is attached, so any
measured gap is attributable to Differential Privacy rather than to incidental
differences between two scripts.

That equivalence is load-bearing and was once broken: this script formerly built
single-timestep inputs while train.py built genuine 5-step windows, silently
confounding the headline comparison. Both now share TransactionWindowDataset
from data_loader.py. If you change the data construction in one, change it in
the other. See thesis_source.md Section 3.6.2.

Typical use:
    python train_baseline.py --epochs 5 --minority_oversample_factor 1.0
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

from model_arch import FraudDetector
from data_loader import load_financial_data, TransactionWindowDataset
from reproducibility import DEFAULT_SEED, set_seed


def train_baseline(epochs=5, output_dir=".", tag=None, minority_oversample_factor=1.0,
                    seed=DEFAULT_SEED, tiebreak="hash"):
    print("--- Initiating VANILLA (Non-DP) GPU Training ---")
    # Seed before any model construction or data shuffling, so this baseline is
    # reproducible on the same terms as the DP runs it's compared against.
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    base_dir = "/scratch5/ic800171/thesis_project/elliptic_data/elliptic_bitcoin_dataset"
    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(base_dir, "elliptic_txs_classes.csv")

    # Thread the row-ordering mode through explicitly. The baseline is the
    # control for every DP comparison in the thesis, and ordering demonstrably
    # moves it, so which ordering was used must be recorded rather than
    # inherited silently from the loader default.
    raw_features, labels = load_financial_data(features_path, classes_path, tiebreak=tiebreak)

    # 2. Chronological Train/Test Split & Temporal Windowing
    # IMPORTANT: this must match train.py's data construction exactly
    # (same TransactionWindowDataset, same window size) so the DP vs
    # non-DP comparison isolates the effect of DP-SGD alone, rather than
    # also confounding it with a difference in how much temporal context
    # each model receives.
    print("Executing Chronological Split (Train: Steps 1-34, Test: Steps 35-49)...")
    time_steps = raw_features[:, 0]
    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    X_train_raw = raw_features[train_mask][:, 1:]
    y_train_raw = labels[train_mask]

    X_test_raw = raw_features[test_mask][:, 1:]
    y_test_raw = labels[test_mask]

    WINDOW_SIZE = 5
    print(f"Applying Temporal Sliding Window (Size: {WINDOW_SIZE}) to capture crime motifs...")

    train_dataset = TransactionWindowDataset(X_train_raw, y_train_raw, window_size=WINDOW_SIZE)
    test_dataset = TransactionWindowDataset(X_test_raw, y_test_raw, window_size=WINDOW_SIZE)

    # Optional minority oversampling, for isolating oversampling's own effect
    # from DP-SGD's noise/clipping effect (see train.py for the full
    # rationale -- this flag exists here purely to run that comparison).
    if minority_oversample_factor > 1.0:
        window_targets = y_train_raw[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(train_dataset)]
        fraud_idx = torch.nonzero(window_targets == 1, as_tuple=True)[0].cpu().tolist()
        safe_idx = torch.nonzero(window_targets == 0, as_tuple=True)[0].cpu().tolist()
        n_repeats = int(round(minority_oversample_factor))
        oversampled_idx = safe_idx + fraud_idx * n_repeats
        print(f"Oversampling minority class {n_repeats}x: {len(fraud_idx)} fraud windows -> "
              f"{len(fraud_idx) * n_repeats}, {len(safe_idx)} safe windows unchanged. "
              f"New train set size: {len(oversampled_idx)} (was {len(train_dataset)}).")
        train_dataset = Subset(train_dataset, oversampled_idx)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    print(f"Rolling Windows Created -> Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # 3. Initialize Standard Model (NO OPACUS MODIFICATIONS)
    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # If oversampling is active, imbalance is already being handled at the
    # data level -- don't also reweight the loss (keeps the comparison to
    # train.py's oversampled DP run a clean single-variable difference).
    weights = torch.tensor([1.0, 10.0 if minority_oversample_factor <= 1.0 else 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("\n[SUCCESS] Baseline Pipeline is armed. Starting Training Loop...")

    # 4. Training Loop (No Privacy Engine)
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
        print(f"Epoch {epoch+1}/{epochs} | Baseline Train Loss: {avg_loss:.4f}")
        epoch_history.append({"epoch": epoch + 1, "train_loss": avg_loss})

    # 5. Evaluation on Unseen Test Data
    print("\n--- Evaluating Baseline Model on Unseen Test Data ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            # Keep the raw P(Fraud) as well as the argmax label. Recall,
            # precision and F1 all describe ONE operating point (the 0.5
            # threshold), so comparing two models on them compares two
            # arbitrary points on two different curves. Average precision
            # summarises the whole precision-recall curve and is the right
            # basis for a like-for-like comparison under heavy class
            # imbalance -- see thesis_source.md Section 5.3.
            all_scores.extend(
                torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            )
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())


    # Threshold-independent metrics. The classification report above is
    # computed at a single 0.5 cutoff; these summarise performance across ALL
    # thresholds, so two models sitting at different operating points can be
    # compared like for like. Average precision is the appropriate summary
    # under heavy class imbalance (ROC-AUC is optimistic when negatives
    # dominate), so both are recorded and average precision is the one to cite.
    average_precision = float(average_precision_score(all_labels, all_scores))
    roc_auc = float(roc_auc_score(all_labels, all_scores))
    print(f"Average Precision (PR-AUC): {average_precision:.4f} | ROC-AUC: {roc_auc:.4f}")

    report_text = classification_report(all_labels, all_preds, target_names=['Safe', 'Fraud'], zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=['Safe', 'Fraud'], zero_division=0, output_dict=True)
    print("\nFinal Baseline Classification Report:")
    print(report_text)

    # 6. Persist weights + metrics (same convention as train.py)
    if tag:
        weights_path = os.path.join(output_dir, f"fraud_detector_baseline_{tag}.pth")
        metrics_path = os.path.join(output_dir, f"training_metrics_baseline_{tag}.json")
    else:
        weights_path = os.path.join(output_dir, "fraud_detector_baseline.pth")
        metrics_path = os.path.join(output_dir, "training_metrics_baseline.json")

    # Refuse to silently overwrite a previous run's output (see train.py).
    for path in (weights_path, metrics_path):
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists -- refusing to overwrite it silently. "
                f"Use a different --tag, or delete the file first if this is intentional."
            )

    torch.save(model.state_dict(), weights_path)
    print(f"\nBaseline model weights saved to {weights_path}")

    metrics = {
        "run_type": "baseline_no_dp",
        "seed": seed,
        "tiebreak": tiebreak,
        "minority_oversample_factor": minority_oversample_factor,
        "class_weights": weights.cpu().tolist(),
        "epoch_history": epoch_history,
        "window_size": WINDOW_SIZE,
        "train_windows": len(train_dataset),
        "test_windows": len(test_dataset),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "classification_report": report_dict,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--minority_oversample_factor", type=float, default=1.0)
    parser.add_argument("--tiebreak", type=str, default="hash", choices=["hash", "txid", "none"],
                         help="Row-ordering mode for rows tied on time_step. Must match the DP runs "
                              "this baseline is compared against. See hpc/data_loader.py.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Random seed for reproducibility. Logged into the metrics JSON so any "
                              "reported run can be reproduced exactly on the same environment.")
    args = parser.parse_args()

    train_baseline(epochs=args.epochs, output_dir=args.output_dir, tag=args.tag,
                    minority_oversample_factor=args.minority_oversample_factor, seed=args.seed,
                    tiebreak=args.tiebreak)
