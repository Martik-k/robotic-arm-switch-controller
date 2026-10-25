#!/usr/bin/env bash
# Robustness test: trained models + expert with noise on the button position -> results/robustness.csv
# Usage:  bash scripts/run_noise_sweep.sh
#         SIGMAS="0.01 0.02" bash scripts/run_noise_sweep.sh     -> other sigmas (meters)
#         ENV=lever-pull-v3 bash scripts/run_noise_sweep.sh      -> another task
#         VARIANT=ppo-bias2cm BASELINES="" bash scripts/run_noise_sweep.sh   -> other runs, no expert
#         KINDS=jitter bash scripts/run_noise_sweep.sh           -> only one noise type

cd "$(dirname "$0")/.."
ENV=${ENV:-button-press-v3}
VARIANT=${VARIANT:-ppo}
RUNS=${RUNS:-runs/${VARIANT}_${ENV}_seed*}
BASELINES=${BASELINES-"--expert"}
SIGMAS=${SIGMAS:-"0.005 0.01 0.02 0.05"}
KINDS=${KINDS:-"jitter bias"}

# sigma 0 once, both curves start from it
python -m scripts.evaluate $BASELINES --env "$ENV" --run $RUNS --noise-jitter 0 "$@"

for sigma in $SIGMAS; do
    echo "=== sigma $sigma ==="
    # one noise type at a time
    for kind in $KINDS; do
        python -m scripts.evaluate $BASELINES --env "$ENV" --run $RUNS --noise-$kind "$sigma" "$@"
    done
done
