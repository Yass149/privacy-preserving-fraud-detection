"""Manual smoke test: construct FraudDetector, push one random batch through it,
print the output shape.

Be clear about what this is NOT. It contains no assertions and returns no
pass/fail signal -- it prints "successful" as long as the forward pass does not
raise, and would do so even if the model produced the wrong output shape. It
catches import errors and gross wiring mistakes, nothing subtler.

The repository currently has no automated test suite. Adding assert-based tests
-- particularly a checkpoint round-trip load with strict=True, and an
equivalence check between the root and hpc copies of model_arch.py -- would
close the two silent-failure risks most likely to bite this project.
"""

import torch
from model_arch import FraudDetector

def test_pipeline():
    # Simulate a batch of data: 8 nodes, window_size=5 time steps, 165 features (Elliptic size)
    dummy_data = torch.randn(8, 5, 165)

    # Initialize the model
    model = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)
    
    # Run the forward pass
    output = model(dummy_data)
    
    print("Integration Test Successful!")
    print("Input shape:", dummy_data.shape)
    print("Output shape (Batch, Classes):", output.shape)

if __name__ == "__main__":
    test_pipeline()