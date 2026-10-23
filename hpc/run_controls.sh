#!/usr/bin/env bash
# The three missing control experiments, plus multi-seed repeats of the three
# headline configurations.
#
# WHY THESE EXIST
# The examining committee's second-largest criticism: the headline causal claim
# rests on a 2-of-4 design, and the two decisive controls were never run.
#
#   1. DP-SGD UNWEIGHTED. Section 5.1 asserts that under clipping "training
#      proceeds, in effect, as unweighted SGD". Unweighted DP-SGD was never
#      actually run, so the equivalence is asserted rather than shown. If this
#      also collapses to ~0.000, the mechanism is demonstrated. If it does NOT,
#      the 10x weight was doing something and the explanation needs revising.
#
#   2. NON-PRIVATE OVERSAMPLED. The headline comparison currently changes TWO
#      variables at once (DP on/off AND reweighting -> oversampling), so the
#      difference cannot be attributed to DP alone -- despite Section 1.4
#      claiming exactly that. This is the missing cell that isolates DP.
#
#   3. max_grad_norm SWEPT ON THE REWEIGHTED CONFIG. The decisive test of the
#      thesis's own mechanism. If clipping cancels the 10x weight, then raising
#      max_grad_norm should progressively RESTORE recall. The existing sweep was
#      run only on the oversampled config, where it cannot test this.
#
#   4. MULTI-SEED HEADLINES. Every table in Chapter 4 is a single run at seed
#      42, and across the three seeds that do exist for the oversampled config,
#      seed 42 is the most favourable on Macro F1. Two extra seeds per headline
#      configuration allow mean and spread to be reported instead.
#
# These runs also capture average precision and ROC-AUC, which the training
# scripts now compute -- the threshold-independent metrics the committee asked
# for. No existing result is overwritten: every run here uses its own --tag.
#
# Resumable: a step whose metrics file already exists is skipped, so an
# interrupted run continues rather than restarting.
#
#     bash run_controls.sh

LOG="controls_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

COMPLETED=0; SKIPPED=0; FAILED=0

run_step() {
    local outfile="$1"; shift
    local label="$1"; shift
    if [ -f "$outfile" ]; then
        echo ">>> SKIP  [$label] -- $outfile exists"
        SKIPPED=$((SKIPPED + 1)); return 0
    fi
    echo
    echo "----------------------------------------------------------"
    echo ">>> RUN   [$label]"
    echo ">>> $*"
    echo "----------------------------------------------------------"
    local start=$SECONDS
    if "$@"; then
        echo ">>> DONE  [$label] in $((SECONDS - start))s"; COMPLETED=$((COMPLETED + 1))
    else
        echo ">>> FAIL  [$label] (exit $?) -- continuing"; FAILED=$((FAILED + 1))
    fi
}

echo "=========================================================="
echo " Control experiments started: $(date)"
echo "=========================================================="

# ---- CONTROL 1: DP-SGD with NO imbalance remedy at all ----
# oversample=1.0 keeps loss weighting ON in train.py, so weighting must be
# neutralised explicitly. --class_weight_minority 1.0 does that.
run_step training_metrics_ctrl_dp_unweighted.json "CONTROL 1: DP-SGD, unweighted, no oversampling" \
    python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
        --minority_oversample_factor 1.0 --class_weight_minority 1.0 \
        --tag ctrl_dp_unweighted

# ---- CONTROL 2: non-private WITH oversampling (isolates DP) ----
run_step training_metrics_baseline_ctrl_oversampled.json "CONTROL 2: non-private + 5x oversampling" \
    python train_baseline.py --epochs 5 --minority_oversample_factor 5.0 \
        --tag ctrl_oversampled

# ---- CONTROL 3: clip-norm sweep ON THE REWEIGHTED config ----
for mgn in 1.0 10.0 100.0; do
    run_step "training_metrics_ctrl_rw_mgn${mgn}.json" "CONTROL 3: reweighted, max_grad_norm=${mgn}" \
        python train.py --noise_multiplier 1.1 --max_grad_norm "$mgn" \
            --minority_oversample_factor 1.0 --tag "ctrl_rw_mgn${mgn}"
done

# ---- MULTI-SEED HEADLINES (seed 42 already exists for all three) ----
for seed in 1 2; do
    run_step "training_metrics_baseline_seed${seed}.json" "SEED ${seed}: non-private baseline" \
        python train_baseline.py --epochs 5 --minority_oversample_factor 1.0 \
            --seed "$seed" --tag "seed${seed}"

    run_step "training_metrics_seed${seed}_reweighted.json" "SEED ${seed}: DP + 10x reweighting" \
        python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
            --minority_oversample_factor 1.0 --seed "$seed" --tag "seed${seed}_reweighted"

    run_step "training_metrics_seed${seed}_oversampled.json" "SEED ${seed}: DP + 5x oversampling" \
        python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
            --minority_oversample_factor 5.0 --seed "$seed" --tag "seed${seed}_oversampled"
done

echo
echo "=========================================================="
echo " Finished: $(date)"
echo " Completed: $COMPLETED   Skipped: $SKIPPED   Failed: $FAILED"
echo "=========================================================="
echo
echo "RESULTS"
python - <<'PY'
import glob, json, os

def row(path, label):
    if not os.path.exists(path):
        return f"  {label:44s} (not produced)"
    d = json.load(open(path))
    fr = d.get("classification_report", {}).get("Fraud", {})
    ap = d.get("average_precision")
    eps = d.get("final_epsilon")
    return (f"  {label:44s} recall={fr.get('recall', float('nan')):.4f} "
            f"prec={fr.get('precision', float('nan')):.4f} "
            f"AP={ap if ap is None else round(ap, 4)} "
            f"eps={eps if eps is None else round(eps, 4)}")

print("\nCONTROLS")
print(row("training_metrics_ctrl_dp_unweighted.json", "DP-SGD unweighted (no remedy)"))
print(row("training_metrics_baseline_ctrl_oversampled.json", "Non-private + 5x oversampling"))
for m in ("1.0", "10.0", "100.0"):
    print(row(f"training_metrics_ctrl_rw_mgn{m}.json", f"Reweighted, max_grad_norm={m}"))

print("\nMULTI-SEED HEADLINES")
for name, files in [
    ("baseline",    ["training_metrics_baseline.json", "training_metrics_baseline_seed1.json", "training_metrics_baseline_seed2.json"]),
    ("reweighted",  ["training_metrics.json", "training_metrics_seed1_reweighted.json", "training_metrics_seed2_reweighted.json"]),
    ("oversampled", ["training_metrics_oversampled.json", "training_metrics_seed1_oversampled.json", "training_metrics_seed2_oversampled.json"]),
]:
    vals = []
    for f in files:
        if os.path.exists(f):
            vals.append(json.load(open(f))["classification_report"]["Fraud"]["recall"])
    if len(vals) > 1:
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        print(f"  {name:14s} n={len(vals)}  recall mean={mean:.4f}  min={min(vals):.4f}  "
              f"max={max(vals):.4f}  spread={spread:.4f}")
    else:
        print(f"  {name:14s} only {len(vals)} run available")

print("""
HOW TO READ THE CONTROLS

  Control 1 collapses to ~0.000
      -> unweighted DP-SGD fails exactly like the reweighted config, so the
         10x weight was indeed doing nothing. Mechanism DEMONSTRATED.
  Control 1 performs meaningfully better than 0.000
      -> the weight WAS having an effect and Section 5.1's explanation needs
         revising. Report this; it is a real finding either way.

  Control 2 close to the non-private baseline
      -> oversampling alone costs little without DP, so the utility gap in the
         headline comparison is attributable to DP. Claim SUPPORTED.
  Control 2 much better than the DP oversampled run
      -> that gap is the true cost of DP, isolated for the first time.

  Control 3 recall RISES with max_grad_norm
      -> loosening the clip restores the weight's effect. This is the single
         most direct confirmation of the thesis's mechanism.
  Control 3 stays flat near 0.000
      -> clipping is not the operative cause and the mechanism needs rethinking.
""")
PY
