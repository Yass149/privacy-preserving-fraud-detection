"""The FraudDetector architecture, shared by training (on the HPC cluster) and
inference (in the local dashboard).

Three stages: a FeatureBottleneckEncoder compressing 165 Elliptic features to a
32-dimensional latent vector, a TemporalGatedAggregator (a GRU) consuming the
5-step encoded sequence, and a linear binary classifier.

One detail matters more than its size suggests. The encoder samples from a
learned distribution during training as a mild regulariser, but that sampling is
gated behind `self.training`, so `.eval()` returns the distribution mean
deterministically. Before that gate existed, the deployed model returned a
different fraud probability every time it was called on identical input, and
because the same code path ran during test-set evaluation, every reported metric
was a single non-reproducible draw. See thesis_source.md Section 3.6.1.

This file is duplicated at hpc/model_arch.py because the cluster runs scripts
from that directory. The two copies must stay identical -- a checkpoint trained
against one will not load correctly into the other if they diverge.
"""

import torch
import torch.nn as nn

class FeatureBottleneckEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FeatureBottleneckEncoder, self).__init__()
        # PyTorch Linear layers naturally accept 3D (Batch, Time, Features) inputs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2) 
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        if self.training:
            return self.reparameterize(mu, logvar)
        return mu

class TemporalGatedAggregator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(TemporalGatedAggregator, self).__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        
    def forward(self, x_sequence):
        output, _ = self.gru(x_sequence)
        return output[:, -1, :]

class FraudDetector(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(FraudDetector, self).__init__()
        self.encoder = FeatureBottleneckEncoder(input_dim, latent_dim)
        self.aggregator = TemporalGatedAggregator(latent_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2) 

    def forward(self, x_sequence):
        # 3D sequence passed directly to match the RACC2 training environment
        encoded_seq = self.encoder(x_sequence)
        temporal_features = self.aggregator(encoded_seq)
        return self.classifier(temporal_features)