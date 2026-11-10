#!/usr/bin/env bash
# P1 reasoner pilot: MetaQA 2-hop, 3 clients, rho = 0.5, exact relay (raw IDs), no DP.
# Four runs: Centralized, FedH-KGQA (relay-discovered labels), federated with local labels only, Isolated.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_p1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs; mkdir -p "$LOG"
D=${D:-data/processed/metaqa/2hop/C3_rho0.5_balanced_s0}
WORKERS=${WORKERS:-8}
export PYTHONUNBUFFERED=1
COMMON="--data-dir $D --encoder hf --model-name ${MODEL:-roberta-base} --rounds ${ROUNDS:-5} --workers $WORKERS"

python experiments/p1_reasoner.py $COMMON --setting centralized --labels gold 2>&1 | tee "$LOG/p1_centralized.log"
python experiments/p1_reasoner.py $COMMON --setting federated --labels relay 2>&1 | tee "$LOG/p1_fed_relay.log"
python experiments/p1_reasoner.py $COMMON --setting federated --labels local 2>&1 | tee "$LOG/p1_fed_local.log"
python experiments/p1_reasoner.py $COMMON --setting isolated 2>&1 | tee "$LOG/p1_isolated.log"
echo "Done. Share results/logs/p1_*.log and results/p1/*/metrics.json"
