#!/usr/bin/env bash
set -e
MODELS=("dcgan64" "ddpm64")
SPLITS=("iid" "noniid_light" "noniid_heavy")
METHODS=("fedavg" "fedprox" "fedcafe")
SEEDS=(0 1 2)
mkdir -p results
for m in "${MODELS[@]}"; do
  for s in "${SPLITS[@]}"; do
    for a in "${METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python3 fedgen/runner.py --config fedgen/configs/${m}_${s}.yaml --algo ${a} --seed ${seed}
      done
    done
  done
done
python3 fedgen/fairness/metrics.py --summarize runs --out results/summary.csv
echo "[OK] results/summary.csv"
