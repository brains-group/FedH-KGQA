#!/usr/bin/env bash
# WebQSP baselines: entity-embedding setting chosen on dev (frozen, or fine-tuned with lr 1e-4).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
D=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
C="--kge-epochs 60 --kge-rounds 20 --kge-local-epochs 3 --kge-neg 4096 --qa-rounds 50 --qa-local-epochs 3"
for V in frozen:--tune-ent=0 elr1e-4:--ent-lr=1e-4; do
  N=${V%%:*}; A=${V#*:}
  mkdir -p results/logs/baselines_wq_$N
  for M in embedkgqa fede fedr; do
    python experiments/b_kgqa.py --data-dir $D --method $M --out results/baselines_wq_$N/webqsp $C $A \
        2>&1 | tee results/logs/baselines_wq_$N/webqsp_$M.log
  done
done
echo Done.
