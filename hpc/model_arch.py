"""The FraudDetector architecture -- HPC copy.

Byte-for-byte equivalent in behaviour to model_arch.py in the repository root,
duplicated here because the cluster runs training scripts from this directory
and imports it as a top-level module. The two copies MUST stay in sync: a
checkpoint trained against this definition will not load correctly into a
diverged root copy, and `strict=False` loading would hide the mismatch rather
than raise.

See the root copy's docstring for the architecture description and for why the
encoder's sampling is gated behind `self.training`.
"""

import torch
import torch.nn as nn

class FeatureBottleneckEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FeatureBottleneckEncoder, self).__init__()
        # PyTorch Linear layers naturally accept 3D (Batch, Time, Features) inputs!
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
        # THE OPACUS FIX: 
        # Pass the 3D sequence directly into the encoder. 
        # No .view() flattening means Opacus tracks the batch size perfectly.
        encoded_seq = self.encoder(x_sequence)
        
        temporal_features = self.aggregator(encoded_seq)
        
        return self.classifier(temporal_features)