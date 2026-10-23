"""Load the Elliptic Bitcoin dataset into GPU memory and build temporal windows.

Reads the features and labels CSVs with cuDF, joins them on transaction ID,
drops the unlabelled majority, orders the result chronologically, and hands back
tensors already resident in VRAM.

TWO ORDERING BUGS LIVE HERE, BOTH NOW FIXED, BOTH WORTH UNDERSTANDING
---------------------------------------------------------------------
1. cuDF's hash join does not preserve input row order. Since
   TransactionWindowDataset builds windows from *consecutive rows*, an unsorted
   frame produced windows grouping temporally unrelated transactions. Fixed by
   sorting on the time column before windowing.

2. Sorting on the time column ALONE is not enough, and this one was subtle.
   Elliptic has 49 time steps across ~46,500 rows, so roughly 950 rows tie on
   every value, and their relative order was left to whatever the join emitted
   -- which varied between runs. Measured: 46,511 of 46,564 rows changed
   position between two runs, so essentially every window differed.

   Making the order total fixes reproducibility, but raises which tiebreaker is
   legitimate. Ordering by txId turned out to leak: a model whose minority-class
   signal had been destroyed by gradient clipping still reached 0.49 fraud
   recall under it, and exactly 0.000 under an ordering decorrelated from txId,
   on every seed tested. It was reading structure out of the identifier rather
   than the features. The default is therefore a fixed hash of txId -- equally
   reproducible, carrying no information beyond the time step.

See the `tiebreak` parameter below, and thesis_source.md Sections 3.6.3/3.6.6.
"""

import cudf
import torch
import os

def load_financial_data(features_path, classes_path, tiebreak="hash"):
    """Load and chronologically order the Elliptic dataset.

    `tiebreak` controls how rows sharing a time_step are ordered. Elliptic
    provides a time_step but NO ordering within it (~950 transactions share
    each of the 49 steps), so this is an arbitrary modelling choice -- and the
    reported metrics turn out to be sensitive to it, which is itself worth
    knowing.

      "hash" -- THE DEFAULT. Order tied rows by a fixed hash of txId:
                deterministic and reproducible, but decorrelated from txId's
                own sequence, so window composition carries no information
                beyond the time_step the thesis says it carries.

      "txid" -- order tied rows by txId. Also deterministic, but MEASURABLY
                CONTAMINATED and kept only to reproduce the runs that
                exposed the problem. txId is not a neutral key. A controlled
                experiment (run_ordering_experiment.sh, 3 seeds per cell)
                showed that DP-SGD with loss reweighting -- a configuration
                in which gradient clipping has cancelled the class weight,
                leaving essentially no minority-class learning signal --
                nonetheless reaches 0.48 fraud recall under txId ordering,
                while collapsing to exactly 0.0000 under hash ordering on
                every seed tried. A model with no usable signal cannot score
                0.48 legitimately; it is exploiting structure encoded in the
                txId sequence itself. Oversampling, by contrast, scored ~0.74
                under BOTH orderings, confirming it depends on real features
                rather than on that artefact.

      "none" -- sort on time_step alone (the original, pre-fix behaviour).
                NOT reproducible: cuDF's hash join emits tied rows in an
                arbitrary order that varies run to run. Provided only to
                reproduce historical results, never for new ones.
    """
    print(f"Loading features from {features_path}...")
    df_features = cudf.read_csv(features_path, header=None)

    # BULLETPROOF RENAME: Dynamically grab the exact name of the first column
    first_col_name = df_features.columns[0]
    df_features = df_features.rename(columns={first_col_name: 'txId'})

    # The second raw column is time_step. Capture its name now (by identity,
    # not position) so we can sort on it later regardless of how merge()
    # reorders columns.
    time_col_name = df_features.columns[1]

    print(f"Loading classes from {classes_path}...")
    df_classes = cudf.read_csv(classes_path)

    print("Merging datasets on txId in VRAM...")
    df_merged = df_features.merge(df_classes, on='txId', how='inner')

    print("Filtering out 'unknown' labels...")
    df_labeled = df_merged[df_merged['class'] != 'unknown'].copy()

    # Map Elliptic classes to standard binary classification ('1' -> 1, '2' -> 0)
    df_labeled['label'] = df_labeled['class'].map({'1': 1, '2': 0})

    # CRITICAL: cudf.merge() does not guarantee output row order matches
    # either input's order. TransactionWindowDataset builds sliding windows
    # by slicing consecutive rows, so without this sort those windows would
    # group temporally unrelated transactions together.
    #
    # The sort MUST be total, not just by time_step. Elliptic has only 49
    # distinct time steps across ~46.5k rows, so roughly 950 rows share each
    # value. Sorting on time_step alone leaves those ties in whatever order
    # the hash-join happened to emit, which varies between runs -- and since
    # windows are built from consecutive rows, different tie orderings produce
    # different windows, different class counts at the split boundary, and
    # therefore slightly different metrics from an otherwise identical run.
    # This was observed directly: two attack runs on the same checkpoints, with
    # identical seeds, reported test-set Fraud counts of 1082 and 1083.
    #
    # Adding txId (unique per transaction) as a tiebreaker makes the ordering
    # total and therefore deterministic, so a seeded run is now genuinely
    # reproducible end to end rather than reproducible only in its training.
    if tiebreak == "txid":
        print(f"Sorting by ({time_col_name}, txId) for a total, deterministic order...")
        df_labeled = df_labeled.sort_values([time_col_name, 'txId']).reset_index(drop=True)
    elif tiebreak == "hash":
        # A fixed multiplicative hash: deterministic across runs, but ordering
        # tied rows independently of txId's own sequence, so any structure
        # carried by txId cannot leak into the window composition.
        print(f"Sorting by ({time_col_name}, hash(txId)) -- deterministic, txId-decorrelated...")
        df_labeled['_tb'] = (df_labeled['txId'].astype('int64') * 2654435761) % 4294967296
        df_labeled = df_labeled.sort_values([time_col_name, '_tb']).reset_index(drop=True)
        df_labeled = df_labeled.drop(columns=['_tb'])
    elif tiebreak == "none":
        print(f"Sorting by {time_col_name} ONLY -- WARNING: not reproducible, ties are "
              f"ordered arbitrarily by the hash join and vary between runs.")
        df_labeled = df_labeled.sort_values(time_col_name).reset_index(drop=True)
    else:
        raise ValueError(f"tiebreak must be 'txid', 'hash', or 'none'; got {tiebreak!r}")

    print(f"Final labeled dataset size: {len(df_labeled)} transactions.")

    feature_cols = [col for col in df_labeled.columns if col not in ['txId', 'class', 'label']]
    features_array = df_labeled[feature_cols].values
    labels_array = df_labeled['label'].values

    features_tensor = torch.as_tensor(features_array, device='cuda', dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels_array, device='cuda', dtype=torch.long)

    print(f"Features Tensor Shape: {features_tensor.shape}")
    print(f"Labels Tensor Shape: {labels_tensor.shape}")

    return features_tensor, labels_tensor


class TransactionWindowDataset(torch.utils.data.Dataset):
    """
    Creates a sliding temporal window of transactions to fulfill Objective 4.
    Relies on load_financial_data() having sorted rows by time_step first --
    Elliptic nodes are one-shot transactions (not a recurring entity's
    history), so a window is a chronologically ordered slice of the labeled
    graph, not one wallet's trajectory. State that framing explicitly in the
    dissertation rather than implying per-entity history.
    """
    def __init__(self, features, labels, window_size=5):
        self.features = features
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        # The total number of valid windows we can create
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        # Slice the sequence: e.g., transactions [t-4, t-3, t-2, t-1, t]
        window_x = self.features[idx : idx + self.window_size]
        # The target label is the classification of the final transaction in this sequence
        target_y = self.labels[idx + self.window_size - 1]
        return window_x, target_y


if __name__ == "__main__":
    base_dir = "/scratch5/ic800171/thesis_project/elliptic_data/elliptic_bitcoin_dataset"
    f_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    c_path = os.path.join(base_dir, "elliptic_txs_classes.csv")

    features, labels = load_financial_data(f_path, c_path)
