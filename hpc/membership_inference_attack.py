"""
Loss-threshold Membership Inference Attack (Yeom et al., 2018), with a
class-stratified breakdown.

The aggregate attack (as originally run) answers: "can an adversary tell
member from non-member, averaged over all transactions?" That number came
back HIGHER for the DP-SGD (oversampled) model than for the non-private
baseline, which is backwards from what a privacy mechanism should do. The
leading hypothesis: 5x-duplicating fraud windows to fix the DP model's
utility collapse (train.py --minority_oversample_factor) means the model
sees each duplicated fraud transaction up to 5x per epoch, and memorizes
those specific examples harder than majority-class examples seen once.

This script tests that hypothesis directly by computing the attack AUC
SEPARATELY within the Fraud-labeled subset and the Safe-labeled subset of
the member/non-member pools. If the hypothesis is right, the Fraud-only
attack AUC should be noticeably higher than the Safe-only attack AUC, and
higher than the aggregate figure -- i.e. the privacy cost of the fix is
concentrated on the very class it was meant to help.

Run on HPC (needs cudf + the real Elliptic CSVs + both trained checkpoints).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from model_arch import FraudDetector
from data_loader import load_financial_data, TransactionWindowDataset

CLASS_NAMES = {0: "Safe", 1: "Fraud"}


def load_checkpoint(path, device):
    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_module.", "") if k.startswith("_module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        # A warning here was not enough. This script computes and writes a
        # privacy result; a partially random-initialised model produces
        # plausible-looking AUCs that are meaningless, and a printed warning
        # scrolls past. Fail loudly instead, matching export_demo_cases.py.
        raise RuntimeError(
            f"{path} is missing weights for {missing}; those layers would run on "
            f"random initialisation, so any attack result computed from this model "
            f"would be meaningless. Refusing to proceed."
        )
    # Opacus writes DPGRU aliases alongside the standard nn.GRU parameter names.
    # Those are inert duplicates; anything else means the checkpoint does not
    # match this architecture.
    unexpected_non_gru = [k for k in unexpected if "gru" not in k.lower()]
    if unexpected_non_gru:
        raise RuntimeError(
            f"{path} carries unexpected non-GRU parameters {unexpected_non_gru}; "
            f"this checkpoint may not match the current architecture."
        )
    model.eval()
    return model


def window_targets_for(dataset, labels_raw, window_size):
    """Vectorized: the label of each window is the label of its last row."""
    n = len(dataset)
    return labels_raw[window_size - 1: window_size - 1 + n]


def per_sample_losses(model, dataset, indices, device, criterion):
    losses = []
    with torch.no_grad():
        for i in indices:
            x, y = dataset[int(i)]
            x = x.unsqueeze(0).to(device)
            y = torch.as_tensor([y]).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
    return np.array(losses)


def sample_indices_for_class(targets, target_class, sample_size, seed):
    class_indices = torch.nonzero(targets == target_class, as_tuple=True)[0].cpu().numpy()
    take = min(sample_size, len(class_indices))
    rng = np.random.default_rng(seed)
    return rng.choice(class_indices, size=take, replace=False), len(class_indices)


def attack_auc_for(member_losses, nonmember_losses):
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(nonmember_losses)])
    return roc_auc_score(labels, scores)


def run_attack(model, train_dataset, test_dataset, train_targets, test_targets,
                device, sample_size, tag):
    criterion = nn.CrossEntropyLoss(reduction="mean")

    result = {"tag": tag}

    # --- Aggregate (matches the original, non-stratified run) ---
    rng_all_train = np.random.default_rng(1)
    rng_all_test = np.random.default_rng(2)
    n_train, n_test = len(train_dataset), len(test_dataset)
    member_idx_all = rng_all_train.choice(n_train, size=min(sample_size * 2, n_train), replace=False)
    nonmember_idx_all = rng_all_test.choice(n_test, size=min(sample_size * 2, n_test), replace=False)
    member_losses_all = per_sample_losses(model, train_dataset, member_idx_all, device, criterion)
    nonmember_losses_all = per_sample_losses(model, test_dataset, nonmember_idx_all, device, criterion)
    agg_auc = attack_auc_for(member_losses_all, nonmember_losses_all)
    result["aggregate"] = {
        "n_members_sampled": int(len(member_losses_all)),
        "n_nonmembers_sampled": int(len(nonmember_losses_all)),
        "mean_member_loss": float(member_losses_all.mean()),
        "mean_nonmember_loss": float(nonmember_losses_all.mean()),
        "attack_auc": float(agg_auc),
    }
    print(f"[{tag}] AGGREGATE Attack AUC: {agg_auc:.4f}")

    # --- Class-stratified: does one class carry disproportionately more leakage? ---
    result["by_class"] = {}
    for class_label in (0, 1):
        class_name = CLASS_NAMES[class_label]
        member_idx, n_member_pool = sample_indices_for_class(train_targets, class_label, sample_size, seed=10 + class_label)
        nonmember_idx, n_nonmember_pool = sample_indices_for_class(test_targets, class_label, sample_size, seed=20 + class_label)

        member_losses = per_sample_losses(model, train_dataset, member_idx, device, criterion)
        nonmember_losses = per_sample_losses(model, test_dataset, nonmember_idx, device, criterion)
        class_auc = attack_auc_for(member_losses, nonmember_losses)

        print(f"[{tag}] {class_name}-only Attack AUC: {class_auc:.4f} "
              f"(members sampled {len(member_losses)}/{n_member_pool} available, "
              f"non-members {len(nonmember_losses)}/{n_nonmember_pool} available)")

        result["by_class"][class_name] = {
            "n_members_sampled": int(len(member_losses)),
            "n_members_available": int(n_member_pool),
            "n_nonmembers_sampled": int(len(nonmember_losses)),
            "n_nonmembers_available": int(n_nonmember_pool),
            "mean_member_loss": float(member_losses.mean()),
            "mean_nonmember_loss": float(nonmember_losses.mean()),
            "attack_auc": float(class_auc),
        }

    fraud_auc = result["by_class"]["Fraud"]["attack_auc"]
    safe_auc = result["by_class"]["Safe"]["attack_auc"]
    print(f"[{tag}] Fraud-vs-Safe leakage gap: {fraud_auc - safe_auc:+.4f} "
          f"({'fraud class leaks MORE' if fraud_auc > safe_auc else 'safe class leaks MORE'})")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, default="fraud_detector_baseline.pth")
    parser.add_argument("--dp_ckpt", type=str, default="fraud_detector_oversampled.pth")
    parser.add_argument("--sample_size", type=int, default=800,
                         help="Max windows sampled PER CLASS for the stratified attack "
                              "(aggregate attack samples 2x this, capped by pool size).")
    parser.add_argument("--output", type=str, default="mia_results_stratified.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = "/scratch5/ic800171/thesis_project/elliptic_data/elliptic_bitcoin_dataset"
    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(base_dir, "elliptic_txs_classes.csv")

    raw_features, labels = load_financial_data(features_path, classes_path)
    time_steps = raw_features[:, 0]
    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    X_train = raw_features[train_mask][:, 1:]
    y_train = labels[train_mask]
    X_test = raw_features[test_mask][:, 1:]
    y_test = labels[test_mask]

    WINDOW_SIZE = 5
    train_dataset = TransactionWindowDataset(X_train, y_train, window_size=WINDOW_SIZE)
    test_dataset = TransactionWindowDataset(X_test, y_test, window_size=WINDOW_SIZE)
    train_targets = window_targets_for(train_dataset, y_train, WINDOW_SIZE)
    test_targets = window_targets_for(test_dataset, y_test, WINDOW_SIZE)

    print(f"Member pool: {len(train_dataset)} ({int((train_targets == 1).sum())} fraud) | "
          f"Non-member pool: {len(test_dataset)} ({int((test_targets == 1).sum())} fraud)")

    results = []

    print("\n=== Attacking BASELINE (non-DP) model ===")
    baseline_model = load_checkpoint(args.baseline_ckpt, device)
    results.append(run_attack(baseline_model, train_dataset, test_dataset, train_targets, test_targets,
                               device, args.sample_size, tag="baseline_no_dp"))

    print("\n=== Attacking DP-SGD (oversampled) model ===")
    dp_model = load_checkpoint(args.dp_ckpt, device)
    results.append(run_attack(dp_model, train_dataset, test_dataset, train_targets, test_targets,
                               device, args.sample_size, tag="dp_sgd_oversampled"))

    if os.path.exists(args.output):
        raise FileExistsError(
            f"{args.output} already exists. Use a different --output, or delete "
            f"the file first if overwriting it is intentional."
        )
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved stratified attack results to {args.output}")


if __name__ == "__main__":
    main()
