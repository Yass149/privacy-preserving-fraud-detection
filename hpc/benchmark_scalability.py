"""Measure the GPU data pipeline against a CPU equivalent, on the full PaySim
dataset. Runs on the GPU node.

Times two implementations of the same work -- read CSV, drop identifier columns,
one-hot encode the transaction type, transfer to tensors -- one via pandas on
CPU, one via cuDF with a zero-copy transfer to PyTorch, and writes the result to
scalability_metrics.json for visualize_scale.py to plot.

This exists because the speed-up figure in an early draft was never measured; it
was an estimate written into the text. Every number it now produces is timed at
run time on real data. Refuses to overwrite an existing output file.

Note the result is hardware-specific: the reported figure came from an NVIDIA
H100 NVL, and an earlier run on an L40S measured a substantially lower speed-up
on the same code and data. Re-run it on whatever hardware you are reporting.
"""

import argparse
import json
import os
import time

import pandas as pd

from paysim_loader import load_paysim_data


def cpu_baseline(csv_path):
    start = time.time()
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    df = pd.get_dummies(df, columns=['type'])
    feature_cols = [c for c in df.columns if c != 'isFraud']
    _ = df[feature_cols].values.astype('float32')
    _ = df['isFraud'].values.astype('int64')
    elapsed = time.time() - start
    return elapsed, len(df)


def gpu_pipeline(csv_path):
    start = time.time()
    features_tensor, _ = load_paysim_data(csv_path)
    elapsed = time.time() - start
    return elapsed, features_tensor.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measures real CPU (pandas) vs GPU (cuDF) pipeline time on the same PaySim CSV. "
                    "Replaces the placeholder numbers in fig_3_scalability.png with an actual measurement."
    )
    parser.add_argument("--csv_path", type=str,
                         default="/scratch5/ic800171/thesis_project/elliptic_data/paysim.csv")
    parser.add_argument("--output", type=str, default="scalability_metrics.json")
    args = parser.parse_args()

    print("Running CPU (pandas) baseline...")
    cpu_time, cpu_rows = cpu_baseline(args.csv_path)
    print(f"CPU: {cpu_rows} rows in {cpu_time:.2f}s")

    print("Running GPU (cuDF) pipeline...")
    gpu_time, gpu_rows = gpu_pipeline(args.csv_path)
    print(f"GPU: {gpu_rows} rows in {gpu_time:.2f}s")

    if cpu_rows != gpu_rows:
        print(f"[WARNING] Row count mismatch: CPU={cpu_rows} GPU={gpu_rows} -- "
              f"investigate before trusting the speedup number.")

    result = {
        "csv_path": args.csv_path,
        "row_count": int(cpu_rows),
        "cpu_seconds": cpu_time,
        "gpu_seconds": gpu_time,
        "speedup": cpu_time / gpu_time,
    }
    if os.path.exists(args.output):
        raise FileExistsError(
            f"{args.output} already exists. Use a different --output, or delete "
            f"the file first if overwriting it is intentional."
        )
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved to {args.output}")
