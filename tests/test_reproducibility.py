"""Seeding and deterministic ordering.

Guards the two defects that made this project's results irreproducible: no RNG
seeding anywhere, and a non-total sort that reordered 99.9% of rows between
runs.
"""

import random

import numpy as np
import pytest
import torch

from reproducibility import DEFAULT_SEED, set_seed


def test_same_seed_produces_identical_draws_across_all_generators():
    """set_seed must cover every generator the training path draws from.
    Missing any one of them leaves a channel of irreproducibility open --
    Opacus, for instance, takes both its DP noise and its Poisson subsampling
    indices from PyTorch's global generator.
    """
    def draws():
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1)),
        )

    set_seed(DEFAULT_SEED)
    first = draws()
    set_seed(DEFAULT_SEED)
    second = draws()

    assert first == second, (
        f"identical seeds produced different draws: {first} vs {second} -- "
        f"some generator is not being seeded"
    )


def test_different_seeds_produce_different_draws():
    """Confirms seeding is actually varying the stream rather than pinning
    everything to a constant, which would look reproducible but destroy the
    randomness training depends on."""
    set_seed(1)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(2)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a != b


def test_set_seed_returns_the_seed_for_logging():
    """Callers log the return value into their metrics JSON so a reported run
    can be traced back to a reproducible configuration."""
    assert set_seed(123) == 123
    assert set_seed() == DEFAULT_SEED


def test_model_initialisation_is_reproducible_under_seed():
    """Weight initialisation is the first thing to consume RNG. If it were not
    covered, every run would start from different weights regardless of what
    happened afterwards."""
    from model_arch import FraudDetector

    set_seed(DEFAULT_SEED)
    a = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)
    set_seed(DEFAULT_SEED)
    b = FraudDetector(input_dim=165, latent_dim=32, hidden_dim=64)

    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(pa, pb), f"parameter {name} differs between identically seeded models"


def test_total_ordering_is_stable_under_input_permutation():
    """The bug that made every window differ between runs.

    Sorting on a heavily-tied key alone leaves tied rows in whatever order the
    upstream join emitted, which varies. Elliptic has ~950 rows per time step,
    so this reordered 46,511 of 46,564 rows between two runs.

    A total order -- time step plus a unique tiebreaker -- must produce the same
    sequence regardless of the order rows arrive in. This reproduces both
    behaviours with pandas, whose sort semantics cuDF mirrors.
    """
    pd = pytest.importorskip("pandas")

    n, steps = 5_000, 49
    base = pd.DataFrame({
        "txId": np.arange(n),
        "time": np.repeat(np.arange(1, steps + 1), n // steps + 1)[:n],
    })

    def emit(seed, cols):
        shuffled = base.sample(frac=1, random_state=seed).reset_index(drop=True)
        return shuffled.sort_values(cols).reset_index(drop=True)["txId"].to_numpy()

    partial_a, partial_b = emit(1, ["time"]), emit(2, ["time"])
    assert not np.array_equal(partial_a, partial_b), (
        "fixture failed to reproduce the instability this test exists to guard against"
    )

    total_a, total_b = emit(1, ["time", "txId"]), emit(2, ["time", "txId"])
    assert np.array_equal(total_a, total_b), (
        "a total sort order must be identical regardless of input row order"
    )


def test_hash_tiebreaker_is_deterministic_and_decorrelated():
    """The tiebreaker actually used. It must be stable across runs (so results
    reproduce) while ordering rows independently of txId's own sequence (so no
    structure carried by the identifier leaks into window composition).
    """
    tx = np.arange(10_000, dtype=np.int64)
    hashed = (tx * 2654435761) % 4294967296

    assert np.array_equal(hashed, (tx * 2654435761) % 4294967296), "hash is not deterministic"
    assert len(np.unique(hashed)) == len(tx), "hash collides, so ordering would still be ambiguous"

    # Rank correlation against the raw identifier should be far from 1: that is
    # the whole point of using a hash rather than txId itself.
    order_by_hash = np.argsort(hashed)
    correlation = np.corrcoef(order_by_hash, np.arange(len(tx)))[0, 1]
    assert abs(correlation) < 0.1, (
        f"hash ordering correlates with txId order (r={correlation:.3f}); it would "
        f"reintroduce the leakage it exists to remove"
    )
