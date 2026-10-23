"""Command-line inference example: load the deployed checkpoint and score a
transaction window, printing the fraud probability and the features that drove
it.

For a cleaner, better-documented entry point intended for handing the model to
someone else, see load_model.py in the repository root, and MODEL_CARD.md for
the input/output contract and the model's limitations.

Loads fraud_detector_oversampled.pth -- the DP-SGD-plus-oversampling model that
actually works. Deliberately NOT fraud_detector_private.pth, which is the
collapsed configuration retained only as the historical record of the failure
documented in thesis_source.md Section 4.2; it predicts "Safe" for essentially
everything, so demonstrating it would misrepresent the project.
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients

# Import your custom architecture
import sys as _sys
from pathlib import Path as _Path

# This module now lives in src/. Checkpoints live in results/checkpoints/, and
# paths.py at the repository root is the single place that knows that.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as _paths

from model_arch import FraudDetector

def load_inference_model(model_path):
    print(f"Loading certified private model from {model_path}...")
    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_module."):
            clean_state_dict[key.replace("_module.", "")] = value
        else:
            clean_state_dict[key] = value
            
    # Explicit Loading Verification to catch silent mismatches
    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    if missing_keys:
        print(f"[WARNING] Missing Keys Detected during deserialization:\n{missing_keys}")
        
    model.eval()
    return model

def analyze_transaction(model, transaction_tensor):
    print("\n--- Processing Incoming Temporal Sequence ---")
    
    with torch.no_grad():
        logits = model(transaction_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        fraud_prob = probabilities[0][1].item() * 100
        safe_prob = probabilities[0][0].item() * 100

    print(f"FRAUD PROBABILITY: {fraud_prob:.2f}%")
    print(f"SAFE PROBABILITY:  {safe_prob:.2f}%")
    
    print("\n--- Generating Regulatory AI Explanation ---")
    ig = IntegratedGradients(model)
    
    transaction_tensor.requires_grad_()
    attributions, delta = ig.attribute(transaction_tensor, target=1, return_convergence_delta=True)
    
    # We now isolate the attributions specifically for the most recent transaction [timestep 4]
    attr_latest = attributions[0, -1, :].detach().numpy()
    
    top_indices = np.argsort(attr_latest)[-3:][::-1]
    
    print("Top 3 Suspicious Features Triggering the Alert (Current Timestep):")
    for rank, idx in enumerate(top_indices):
        feature_name = f"Feature_{idx + 1}" 
        impact_score = attr_latest[idx]
        print(f"  {rank + 1}. {feature_name} (Attribution Score: {impact_score:.4f})")
        
    return fraud_prob, top_indices

if __name__ == "__main__":
    # Matches app.py's ACTIVE_CHECKPOINT -- the working DP-SGD + oversampling
    # model. NOT fraud_detector_private.pth, which is the collapsed
    # (0.0009 recall, effectively always-predicts-Safe) run kept on disk only
    # as the historical record of the bug documented in thesis_source.md
    # Section 3.6 -- demoing it here would misrepresent the project the same
    # way app.py's own comment warns against.
    MODEL_PATH = str(_paths.CHECKPOINTS / "fraud_detector_oversampled.pth")
    model = load_inference_model(MODEL_PATH)
    
    # Simulate a rolling window of 5 sequential Bitcoin transactions
    # Shape: (Batch=1, Sequence=5, Features=165)
    print("Intercepting 5-step rolling window of Bitcoin transactions...")
    dummy_sequence = torch.randn(1, 5, 165)
    
    analyze_transaction(model, dummy_sequence)
    