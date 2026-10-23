#!/usr/bin/env bash
# Full re-run of every result reported in the dissertation.
#
# RESUMABLE BY DESIGN: each step is skipped if its output metrics file already
# exists, so if the job is interrupted (dropped SSH, SLURM time limit, node
# failure) you can simply run this script again and it continues from where it
# stopped rather than re-doing hours of finished work.
#
# It deliberately does NOT use `set -e`: one failing configuration should not
# prevent the other fourteen from running. Failures are counted and reported in
# the summary at the end, and everything is logged to run_all_<timestamp>.log.
#
# Usage, on a GPU node with the environment active:
#     bash run_all.sh

LOG="run_all_$(date +%Y%m%d_%H%M%S).log"
FAILED=0
COMPLETED=0
SKIPPED=0

# Log everything to both the terminal and the log file.
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo " Full re-run started: $(date)"
echo " Logging to: $LOG"
echo "=========================================================="
echo

run_step() {
    local outfile="$1"; shift
    local label="$1"; shift

    if [ -f "$outfile" ]; then
        echo ">>> SKIP  [$label] -- $outfile already exists"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo
    echo "----------------------------------------------------------"
    echo ">>> RUN   [$label]"
    echo ">>> $*"
    echo "----------------------------------------------------------"
    local start=$SECONDS

    if "$@"; then
        echo ">>> DONE  [$label] in $((SECONDS - start))s"
        COMPLETED=$((COMPLETED + 1))
    else
        echo ">>> FAIL  [$label] (exit $?) -- continuing with remaining steps"
        FAILED=$((FAILED + 1))
    fi
}

# ============ ELLIPTIC: the three headline configurations (Sec 4.2) ============
# Run first: this is the thesis's primary dataset. If time runs out, these are
# the results that matter most.

run_step training_metrics_baseline.json "elliptic baseline (no DP)" \
    python train_baseline.py --epochs 5 --minority_oversample_factor 1.0

run_step training_metrics.json "elliptic DP + loss reweighting (collapses)" \
    python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 --minority_oversample_factor 1.0

run_step training_metrics_oversampled.json "elliptic DP + 5x oversampling (the fix)" \
    python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
        --minority_oversample_factor 5.0 --tag oversampled

# ============ ELLIPTIC: noise_multiplier sweep (Sec 4.4, table 1) ============
# nm=1.1 is intentionally absent -- it is the same configuration as the
# 'oversampled' run above, which the thesis reuses as the sweep's centre point.

for nm in 0.4 0.7 1.5 2.0 3.0; do
    run_step "training_metrics_nm${nm}.json" "elliptic sweep noise_multiplier=${nm}" \
        python train.py --noise_multiplier "$nm" --max_grad_norm 1.0 \
            --minority_oversample_factor 5.0 --tag "nm${nm}"
done

# ============ ELLIPTIC: max_grad_norm sweep (Sec 4.4, table 2) ============
# mgn=1.0 is likewise the 'oversampled' run above.

for mgn in 0.5 2.0 4.0; do
    run_step "training_metrics_mgn${mgn}.json" "elliptic sweep max_grad_norm=${mgn}" \
        python train.py --noise_multiplier 1.1 --max_grad_norm "$mgn" \
            --minority_oversample_factor 5.0 --tag "mgn${mgn}"
done

# ============ PAYSIM: cross-dataset generalisation (Sec 4.5) ============
# Slower than Elliptic -- 6.36M rows, and the 200x oversample expands the
# training set to ~5.2M. Deliberately last so an out-of-time interruption
# costs the least important results.

run_step training_metrics_paysim_baseline_os1_v2.json "paysim baseline (no DP)" \
    python train_paysim.py --no_dp --epochs 5 --minority_oversample_factor 1.0 --tag v2

run_step training_metrics_paysim_dp_os1_v1.json "paysim DP + reweighting (collapses)" \
    python train_paysim.py --dp --noise_multiplier 1.1 --minority_oversample_factor 1.0 --tag v1

run_step training_metrics_paysim_dp_os200_v1.json "paysim DP + 200x oversampling (recovers)" \
    python train_paysim.py --dp --noise_multiplier 1.1 --minority_oversample_factor 200.0 --tag v1

# ============ PRIVACY AUDIT + SCALABILITY BENCHMARK ============
# These depend on the Elliptic checkpoints above existing.

run_step mia_results_stratified.json "membership inference (class-stratified)" \
    python membership_inference_attack.py \
        --baseline_ckpt fraud_detector_baseline.pth \
        --dp_ckpt fraud_detector_oversampled.pth \
        --output mia_results_stratified.json

run_step mia_results.json "membership inference (aggregate, for the dashboard)" \
    python membership_inference_attack.py \
        --baseline_ckpt fraud_detector_baseline.pth \
        --dp_ckpt fraud_detector_oversampled.pth \
        --output mia_results.json

run_step scalability_metrics.json "cuDF vs pandas scalability benchmark" \
    python benchmark_scalability.py

# ============ SUMMARY ============
echo
echo "=========================================================="
echo " Finished: $(date)"
echo " Completed: $COMPLETED   Skipped (already done): $SKIPPED   Failed: $FAILED"
echo "=========================================================="

if [ "$FAILED" -gt 0 ]; then
    echo
    echo "Some steps failed. Search the log for '>>> FAIL' to find them:"
    echo "    grep -n '>>> FAIL' $LOG"
    echo "Re-running this script will retry ONLY the failed/missing steps."
fi

echo
echo "Seed check -- every run below must show seed=42:"
python - <<'PY'
import glob, json
rows = sorted(glob.glob("training_metrics*.json"))
if not rows:
    print("  (no metrics files found)")
for f in rows:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  {f:52s} UNREADABLE: {e}")
        continue
    fraud = d.get("classification_report", {}).get("Fraud", {})
    eps = d.get("final_epsilon")
    eps_s = f"{eps:.4f}" if isinstance(eps, (int, float)) else "n/a"
    rec = fraud.get("recall")
    rec_s = f"{rec:.4f}" if isinstance(rec, (int, float)) else "n/a"
    print(f"  {f:52s} seed={d.get('seed')}  eps={eps_s:>8s}  fraud_recall={rec_s}")
PY
