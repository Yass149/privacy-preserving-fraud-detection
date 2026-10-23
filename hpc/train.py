"""Train the Elliptic fraud detector under DP-SGD (Opacus). Runs on the GPU node.

This is the script behind the dissertation's central finding. Two ways of
handling the ~10% fraud class are available, and they behave completely
differently under Differential Privacy:

  --minority_oversample_factor 1.0   applies 10x loss reweighting. Under DP-SGD
                                     this COLLAPSES: per-sample gradient
                                     clipping renormalises every gradient to the
                                     same budget, erasing the class weight
                                     before the optimiser sees it. Result: zero
                                     fraud detected.

  --minority_oversample_factor 5.0   duplicates minority windows instead.
                                     Survives clipping, because it changes how
                                     many fraud gradients enter each batch
                                     rather than how large each one is.
                                     Result: recall at parity with the
                                     non-private baseline. Its reported epsilon also falls, but that is an artefact
                                     of the enlarged dataset rather than a privacy gain: see
                                     thesis_source.md Section 5.1 for the group-privacy correction.

Outputs a checkpoint and a metrics JSON recording the full configuration --
including the random seed and the row-ordering mode -- so any reported figure
can be traced to a reproducible run. Refuses to overwrite existing output.

Typical use:
    python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
        --minority_oversample_factor 5.0 --tag oversampled
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

from model_arch import FraudDetector
from data_loader import load_financial_data, TransactionWindowDataset
from reproducibility import DEFAULT_SEED, set_seed


def train_pipeline(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5, epochs=5,
                    output_dir=".", tag=None, minority_oversample_factor=1.0,
                    seed=DEFAULT_SEED, tiebreak="hash", class_weight_minority=None):
    print("--- Initiating Privacy-Preserving GPU Training ---")
    print(f"Config: noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}, delta={delta}")
    # Seed before any model construction or data shuffling, so weight init,
    # DataLoader shuffling and Opacus's DP noise are all reproducible.
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    base_dir = "/scratch5/ic800171/thesis_project/elliptic_data/elliptic_bitcoin_dataset"
    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(base_dir, "elliptic_txs_classes.csv")

    raw_features, labels = load_financial_data(features_path, classes_path, tiebreak=tiebreak)

    # 2. Chronological Train/Test Split & Temporal Windowing
    print("Executing Chronological Split (Train: Steps 1-34, Test: Steps 35-49)...")

    # Extract the time_step column (index 0 of raw_features) to create boolean masks
    time_steps = raw_features[:, 0]

    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    # Drop the time_step column ([:, 1:]); TransactionWindowDataset handles
    # the sequence dimension natively. load_financial_data() has already
    # sorted rows by time_step, so windows below are temporally coherent.
    X_train_raw = raw_features[train_mask][:, 1:]
    y_train_raw = labels[train_mask]

    X_test_raw = raw_features[test_mask][:, 1:]
    y_test_raw = labels[test_mask]

    WINDOW_SIZE = 5
    print(f"Applying Temporal Sliding Window (Size: {WINDOW_SIZE}) to capture crime motifs...")

    train_dataset = TransactionWindowDataset(X_train_raw, y_train_raw, window_size=WINDOW_SIZE)
    test_dataset = TransactionWindowDataset(X_test_raw, y_test_raw, window_size=WINDOW_SIZE)

    # Loss-reweighting (the original approach: weight=[1.0, 10.0]) does NOT
    # survive DP-SGD's gradient clipping: Opacus clips every PER-SAMPLE
    # gradient to the same L2 norm regardless of how the loss was weighted,
    # so a 10x weight on minority-class samples gets renormalized right back
    # down to the same budget as everything else. That cancellation is what
    # collapsed the DP model to an always-predict-majority classifier.
    #
    # Oversampling instead changes HOW MANY minority-class gradients get
    # averaged into each batch, not their individual magnitude -- that
    # signal survives clipping intact. We oversample by duplicating minority
    # windows in the training set (a Subset), so Opacus's uniform Poisson
    # subsampling -- and therefore its epsilon accounting -- still applies
    # correctly to the (now larger, rebalanced) dataset as given to it.
    #
    # Caveat worth stating in the dissertation's limitations: duplicating a
    # window means the same underlying transaction can appear more than once
    # per epoch, so its true per-individual sensitivity is somewhat higher
    # than Opacus's per-row accounting assumes. This is a standard, widely
    # used approximation in DP + imbalanced-data work, not a hidden flaw --
    # but it should be named as an approximation, not presented as exact.
    if minority_oversample_factor > 1.0:
        window_targets = y_train_raw[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(train_dataset)]
        fraud_idx = torch.nonzero(window_targets == 1, as_tuple=True)[0].cpu().tolist()
        safe_idx = torch.nonzero(window_targets == 0, as_tuple=True)[0].cpu().tolist()
        n_repeats = int(round(minority_oversample_factor))
        oversampled_idx = safe_idx + fraud_idx * n_repeats
        minority_share = (len(fraud_idx) * n_repeats) / len(oversampled_idx)
        print(f"Oversampling minority class {n_repeats}x: {len(fraud_idx)} fraud windows -> "
              f"{len(fraud_idx) * n_repeats}, {len(safe_idx)} safe windows unchanged. "
              f"New train set size: {len(oversampled_idx)} (was {len(train_dataset)}), "
              f"minority share: {minority_share:.1%}")
        train_dataset = Subset(train_dataset, oversampled_idx)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    print(f"Rolling Windows Created -> Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # 3. Initialize Model & Fix for Opacus
    # ModuleValidator.fix() can REPLACE incompatible layers (e.g. swap the
    # GRU for a DP-compatible reimplementation), returning a new model
    # object. It must run before the optimizer is created, or the optimizer
    # ends up holding references to the pre-fix model's parameters --
    # exactly the "Module parameters are different than optimizer
    # Parameters" error Opacus raises if this order is wrong.
    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64).to(device)
    model = ModuleValidator.fix(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Imbalance is now handled by oversampling above (when enabled), not loss
    # weighting -- see the comment above for why weighting alone fails here.
    # Default behaviour is unchanged: 10x loss weight when oversampling is off,
    # neutral weight when it is on. class_weight_minority overrides that, which
    # is what makes the "DP-SGD with NO imbalance remedy at all" control
    # expressible -- previously oversample=1.0 forced the 10x weight on, so
    # unweighted DP-SGD could not be run and Section 5.1's claim that clipping
    # reduces training to unweighted SGD could not be tested directly.
    if class_weight_minority is None:
        class_weight_minority = 10.0 if minority_oversample_factor <= 1.0 else 1.0
    weights = torch.tensor([1.0, float(class_weight_minority)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 4. Attach Privacy Engine (ONLY to the training loader)
    print("Attaching Opacus Differential Privacy Engine...")
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    print("\n[SUCCESS] Pipeline is fully armed. Starting Training Loop...")

    # 5. Training Loop
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

        epsilon = privacy_engine.get_epsilon(delta=delta)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Privacy Budget (epsilon): {epsilon:.4f} (delta={delta})")
        epoch_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "epsilon": epsilon})

    final_epsilon = privacy_engine.get_epsilon(delta=delta)

    # 6. Evaluation on Unseen Test Data
    print("\n--- Evaluating Model on Unseen Test Data ---")
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
    print("\nFinal Classification Report:")
    print(report_text)

    # 7. Persist weights + metrics. Default run keeps the existing filenames
    # (drop-in for the local dashboard); tagged sweep runs get their own
    # files so they don't clobber each other or the production checkpoint.
    if tag:
        weights_path = os.path.join(output_dir, f"fraud_detector_{tag}.pth")
        metrics_path = os.path.join(output_dir, f"training_metrics_{tag}.json")
    else:
        weights_path = os.path.join(output_dir, "fraud_detector_private.pth")
        metrics_path = os.path.join(output_dir, "training_metrics.json")

    # Refuse to silently overwrite a previous run's output -- two runs with
    # different configs but the same --tag have clobbered each other before
    # (see train_paysim.py history). Loud failure beats silent data loss.
    for path in (weights_path, metrics_path):
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists -- refusing to overwrite it silently. "
                f"Use a different --tag, or delete the file first if this is intentional."
            )

    torch.save(model.state_dict(), weights_path)
    print(f"\nModel weights saved to {weights_path}")

    metrics = {
        "run_type": "dp_sgd",
        "seed": seed,
        "tiebreak": tiebreak,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "delta": delta,
        "final_epsilon": final_epsilon,
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
    parser.add_argument("--noise_multiplier", type=float, default=1.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default=None,
                         help="Run tag for sweep outputs (e.g. eps_low). Omit for the default production run.")
    parser.add_argument("--minority_oversample_factor", type=float, default=1.0,
                         help="Duplicate minority-class (Fraud) training windows this many times before "
                              "training. 1.0 = off (original loss-reweighting-only behavior). Use e.g. 5.0 "
                              "to fix the clipping-cancels-class-weight collapse -- see the comment above.")
    parser.add_argument("--class_weight_minority", type=float, default=None,
                         help="Loss weight on the minority (Fraud) class. Default: 10.0 when "
                              "oversampling is off, 1.0 when it is on. Set explicitly to 1.0 to "
                              "run DP-SGD with no imbalance remedy at all.")
    parser.add_argument("--tiebreak", type=str, default="hash", choices=["hash", "txid", "none"],
                         help="How to order rows tied on time_step. 'hash' (default) is reproducible "
                              "and decorrelated from txId. 'txid' is reproducible but measurably "
                              "contaminated -- see data_loader.py. 'none' is the old "
                              "non-deterministic behaviour.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Random seed for reproducibility. Logged into the metrics JSON so any "
                              "reported run can be reproduced exactly on the same environment.")
    args = parser.parse_args()

    train_pipeline(noise_multiplier=args.noise_multiplier, max_grad_norm=args.max_grad_norm,
                   delta=args.delta, epochs=args.epochs, output_dir=args.output_dir, tag=args.tag,
                   minority_oversample_factor=args.minority_oversample_factor, seed=args.seed,
                   tiebreak=args.tiebreak, class_weight_minority=args.class_weight_minority)
