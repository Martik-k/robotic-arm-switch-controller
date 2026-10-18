#!/usr/bin/env bash
# Train the same config with several seeds, one after another.
# Usage:  bash scripts/run_seeds.sh                    -> seeds 0 1 2 3 4
#         SEEDS="0 1 2" bash scripts/run_seeds.sh      -> other seeds
#         bash scripts/run_seeds.sh --n-envs 4         -> extra args go to train.py

cd "$(dirname "$0")/.."             # run from the repo root, so `-m scripts.train` works
SEEDS=${SEEDS:-"0 1 2 3 4"}
failed=()

for seed in $SEEDS; do
    echo "=== seed $seed ==="
    # if one seed crashes we still run the rest
    python -m scripts.train --seed "$seed" "$@" || failed+=("$seed")
done

if [ ${#failed[@]} -eq 0 ]; then
    echo "all seeds finished: $SEEDS"
else
    echo "failed seeds: ${failed[*]}"
    exit 1
fi
