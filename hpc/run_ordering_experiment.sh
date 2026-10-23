#!/usr/bin/env bash
# Decisive experiment: is the "DP-SGD + loss reweighting collapses" finding real,
# or an artefact of how rows tied on time_step happen to be ordered?
#
# BACKGROUND
# Elliptic supplies a time_step but no ordering *within* it (~950 transactions
# share each of 49 steps), so within-step order is an arbitrary modelling
# choice. The reweighted configuration has produced wildly different results
# under different orderings:
#     hash-join order (non-reproducible):  recall 0.0009, then 0.0000
#     txId order       (reproducible):     recall 0.4866
# while the oversampled configuration barely moved (0.7461 -> 0.7396).
#
# TWO COMPETING EXPLANATIONS
#   (A) The reweighted config is knife-edge unstable. Clipping cancels the loss
#       weight, leaving a weak minority signal, so the outcome swings on fine
#       details of the trajectory. Ordering is just one such detail.
#   (B) txId ordering leaks structure. If txId correlates with class or
#       insertion time, sorting by it clusters similar transactions into the
#       same window, making the task easier and lifting a weak model above the
#       collapse threshold.
#
# WHAT THIS TESTS
#   * tiebreak=hash orders tied rows by a fixed hash of txId: just as
#     reproducible, but decorrelated from txId's own sequence. If results match
#     txid, txId carries no special structure -> explanation (A). If the model
#     collapses under hash but not txid, txId is doing the work -> (B).
#   * Three seeds per cell measure run-to-run variance, separating "unstable"
#     from "stably different".
#
# Resumable: finished cells are skipped. ~10 runs, roughly 30 minutes.
#     bash run_ordering_experiment.sh

LOG="ordering_experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo " Ordering experiment started: $(date)"
echo "=========================================================="

run_cell() {
    local tag="$1"; shift
    if [ -f "training_metrics_${tag}.json" ]; then
        echo ">>> SKIP  $tag (already done)"
        return 0
    fi
    echo
    echo ">>> RUN   $tag"
    if "$@" --tag "$tag" > /dev/null 2>&1; then
        echo ">>> DONE  $tag"
    else
        echo ">>> FAIL  $tag -- re-run manually to see the error"
    fi
}

# seed 42 / txid already exists from the main run for both configs, under the
# names training_metrics.json and training_metrics_oversampled.json -- the
# summary below reads those directly rather than duplicating the work.

for seed in 42 1 2; do
    for tb in txid hash; do
        # Skip the two cells the main run already produced.
        if [ "$seed" = "42" ] && [ "$tb" = "txid" ]; then continue; fi

        run_cell "exp_reweight_${tb}_s${seed}" \
            python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
                --minority_oversample_factor 1.0 --seed "$seed" --tiebreak "$tb"

        run_cell "exp_oversamp_${tb}_s${seed}" \
            python train.py --noise_multiplier 1.1 --max_grad_norm 1.0 \
                --minority_oversample_factor 5.0 --seed "$seed" --tiebreak "$tb"
    done
done

echo
echo "=========================================================="
echo " RESULTS"
echo "=========================================================="
python - <<'PY'
import json, os

def get(fname):
    if not os.path.exists(fname):
        return None
    d = json.load(open(fname))
    r = d["classification_report"]
    return (r["Fraud"]["recall"], r["Fraud"]["precision"], r["macro avg"]["f1-score"])

def cell(config, tb, seed):
    if seed == 42 and tb == "txid":
        return get("training_metrics.json" if config == "reweight"
                   else "training_metrics_oversampled.json")
    return get(f"training_metrics_exp_{config}_{tb}_s{seed}.json")

for config, label in [("reweight", "DP + 10x LOSS REWEIGHTING  (thesis says: collapses)"),
                      ("oversamp", "DP + 5x OVERSAMPLING       (thesis says: the fix)")]:
    print(f"\n{label}")
    print(f"  {'tiebreak':10s} {'seed':>5s} {'recall':>8s} {'precision':>10s} {'macroF1':>9s}")
    recalls = {}
    for tb in ("txid", "hash"):
        for seed in (42, 1, 2):
            v = cell(config, tb, seed)
            if v is None:
                print(f"  {tb:10s} {seed:5d}        --         --        --")
                continue
            rec, pre, mf = v
            recalls.setdefault(tb, []).append(rec)
            print(f"  {tb:10s} {seed:5d} {rec:8.4f} {pre:10.4f} {mf:9.4f}")
    for tb, vals in recalls.items():
        if len(vals) > 1:
            print(f"  -> {tb}: recall spread {min(vals):.4f} to {max(vals):.4f} "
                  f"(range {max(vals) - min(vals):.4f})")

print("""
HOW TO READ THIS
  * reweight collapses (~0.00) under hash but not txid
        -> txId ordering was doing the work. Switch to hash; the collapse
           finding stands, and the thesis needs only a methodology note.
  * reweight is ~0.49 under BOTH tiebreaks, tight across seeds
        -> the collapse was an artefact of non-reproducible ordering. The
           headline finding must be reframed: reweighting DEGRADES detection
           (0.74 -> 0.49), it does not eliminate it.
  * reweight swings widely ACROSS SEEDS within a tiebreak
        -> the configuration is unstable. That is itself the finding, and a
           stronger one: reweighting under DP-SGD is unreliable, whereas
           oversampling (compare its spread) is stable.
""")
PY
