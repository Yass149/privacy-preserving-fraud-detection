"""Temporal windowing and train/test split correctness.

The dissertation's validity rests on two claims about this code: that a window's
label is the label of its FINAL transaction, and that no window ever mixes
training-era and test-era data. Both are asserted here rather than assumed.
"""

import pytest
import torch

from data_loader import TransactionWindowDataset

WINDOW = 5
FEATURES = 165


def make_data(n_rows, fraud_every=7):
    feats = torch.arange(n_rows * FEATURES, dtype=torch.float32).reshape(n_rows, FEATURES)
    labels = torch.tensor([1 if i % fraud_every == 0 else 0 for i in range(n_rows)])
    return feats, labels


def test_window_count_formula():
    """len = n_rows - window_size + 1. An off-by-one here silently drops or
    invents a window at the boundary and shifts every reported support count."""
    for n in (5, 6, 20, 100):
        feats, labels = make_data(n)
        ds = TransactionWindowDataset(feats, labels, window_size=WINDOW)
        assert len(ds) == n - WINDOW + 1


def test_window_shape_and_label_alignment():
    """The label must come from the LAST row in the window, not the first or
    the middle. Getting this wrong shifts every label by four positions and
    would still train to a plausible-looking accuracy."""
    feats, labels = make_data(50)
    ds = TransactionWindowDataset(feats, labels, window_size=WINDOW)

    for idx in (0, 1, 17, len(ds) - 1):
        window, target = ds[idx]
        assert window.shape == (WINDOW, FEATURES)
        assert torch.equal(window, feats[idx: idx + WINDOW]), "window rows are not consecutive"
        assert target == labels[idx + WINDOW - 1], (
            f"window {idx} carries the label of row {idx + WINDOW - 1}'s neighbour, "
            f"not that row itself"
        )


def test_windows_are_consecutive_and_ordered():
    """Each window must be a contiguous slice in the given row order. If the
    dataset ever reordered or sampled rows internally, the 'chronologically
    ordered context slice' framing in the thesis would be false."""
    feats, labels = make_data(30)
    ds = TransactionWindowDataset(feats, labels, window_size=WINDOW)
    window, _ = ds[10]
    first_row_value = window[0, 0].item()
    for step in range(1, WINDOW):
        expected = first_row_value + step * FEATURES
        assert window[step, 0].item() == expected, "window rows are not consecutive in order"


def test_split_before_windowing_prevents_boundary_leakage():
    """The core anti-leakage property.

    Train and test are split into separate arrays FIRST, then each is windowed
    independently. Because a window is a contiguous slice within one array, no
    window can span the boundary -- a test-era window can never contain a
    training-era transaction, regardless of window size.

    This test reproduces that construction and asserts the property directly.
    """
    n = 60
    feats, labels = make_data(n)
    time_steps = torch.cat([torch.full((34,), 1.0), torch.full((n - 34,), 40.0)])

    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    train_ds = TransactionWindowDataset(feats[train_mask], labels[train_mask], window_size=WINDOW)
    test_ds = TransactionWindowDataset(feats[test_mask], labels[test_mask], window_size=WINDOW)

    train_rows = set(feats[train_mask][:, 0].tolist())
    test_rows = set(feats[test_mask][:, 0].tolist())
    assert not (train_rows & test_rows), "fixture is wrong: splits overlap"

    for idx in range(len(test_ds)):
        window, _ = test_ds[idx]
        for row_marker in window[:, 0].tolist():
            assert row_marker not in train_rows, (
                f"test window {idx} contains a training-era row -- boundary leakage"
            )

    # And the counts must add up: windowing each side separately loses
    # (window_size - 1) windows per side, which is expected and not a bug.
    assert len(train_ds) == train_mask.sum().item() - WINDOW + 1
    assert len(test_ds) == test_mask.sum().item() - WINDOW + 1


def test_window_size_one_is_a_single_row():
    """PaySim uses window_size=1. That is what makes it structurally immune to
    the txId ordering leakage found on Elliptic: a window is one transaction, so
    its composition cannot depend on neighbouring rows at all."""
    feats, labels = make_data(20)
    ds = TransactionWindowDataset(feats, labels, window_size=1)
    assert len(ds) == 20
    for idx in (0, 9, 19):
        window, target = ds[idx]
        assert window.shape == (1, FEATURES)
        assert torch.equal(window[0], feats[idx])
        assert target == labels[idx]


def test_oversampling_preserves_label_alignment():
    """Oversampling duplicates minority window INDICES via a Subset. The
    duplicated entries must still carry their original labels -- if indices and
    labels ever came apart, the fix that recovers fraud recall would be training
    on mislabelled data."""
    from torch.utils.data import Subset

    feats, labels = make_data(60, fraud_every=5)
    ds = TransactionWindowDataset(feats, labels, window_size=WINDOW)
    window_targets = labels[WINDOW - 1: WINDOW - 1 + len(ds)]

    fraud_idx = torch.nonzero(window_targets == 1, as_tuple=True)[0].tolist()
    safe_idx = torch.nonzero(window_targets == 0, as_tuple=True)[0].tolist()
    assert fraud_idx, "fixture produced no fraud windows"

    factor = 5
    oversampled = Subset(ds, safe_idx + fraud_idx * factor)

    assert len(oversampled) == len(safe_idx) + len(fraud_idx) * factor
    for i in range(len(oversampled)):
        _, target = oversampled[i]
        original_index = oversampled.indices[i]
        assert target == window_targets[original_index], "oversampling broke label alignment"

    minority_share = (len(fraud_idx) * factor) / len(oversampled)
    assert 0 < minority_share < 1
