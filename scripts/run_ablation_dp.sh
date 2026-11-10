#!/usr/bin/env bash
# Ablation 5: DP training (QA-pair-level DP-SGD per client, epsilon composed over all rounds, delta 1e-5,
# clip 1.0) for FedH-KGQA (local labels + discovery B = 5) on MetaQA 3-hop and PQ-3H, seed 0.
# The no-DP points are the main-table runs. In DP runs the "client loss" printed per round is the mean
# per-example gradient norm before clipping.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_ablation_dp.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
WORKERS=${WORKERS:-8}
LOG=results/logs/ablation_dp; mkdir -p "$LOG"
B="--encoder hf --model-name roberta-base --workers $WORKERS --sparsify-rel 0.1 --sparsify-top 3 --out results/ablation_dp \
   --setting federated --labels local --discover 5 --dp-delta 1e-5 --dp-clip 1.0 --lr 2e-3"
M3="--data-dir data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr --rounds 5 --local-epochs 2 --dp-batch 1024"
PQ="--data-dir data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta --rounds 20 --local-epochs 5 \
    --discover-start 4 --discover-every 4 --dp-batch 256"
for E in ${EPSILONS:-8 4 2 1}; do
  python experiments/p1_reasoner.py $B $M3 --dp-epsilon $E --tag metaqa3_dp_eps$E 2>&1 | tee "$LOG/metaqa3_dp_eps$E.log"
  python experiments/p1_reasoner.py $B $PQ --dp-epsilon $E --tag pq3_dp_eps$E 2>&1 | tee "$LOG/pq3_dp_eps$E.log"
done
echo "Done. Share: tar czf ablation_dp.tar.gz $LOG results/ablation_dp/*/metrics.json"
