#!/usr/bin/env bash
# P3 (part 1): MetaQA 1-hop and 3-hop with the settings chosen after P1 and P2:
# local labels, pull relay, distribution-matched padding.
# Per hop count: partition (C = 3, rho = 0.5), Centralized, FedH-KGQA (federated, local labels),
# Isolated, Isolated training + relay execution (ablation), then the sketch relay on the
# FedH-KGQA model's probabilities.
# Usage: CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_p3_metaqa.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
META=${META:-data/raw/MetaQA}
HOPS=${HOPS:-"3 1"}
COMMON="--encoder hf --model-name ${MODEL:-roberta-base} --rounds ${ROUNDS:-5} --workers $WORKERS"

for H in $HOPS; do
  D=data/processed/metaqa/${H}hop/C3_rho0.5_balanced_s0
  [ -f $D/stats.json ] || python experiments/p1_prepare_data.py --metaqa-dir "$META" --hops $H --clients 3 \
      --rho 0.5 --seed 0 --workers "$WORKERS" 2>&1 | tee "$LOG/p1_data_${H}hop_C3_rho0.5.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --setting centralized --labels gold \
      2>&1 | tee "$LOG/p3_${H}hop_centralized.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --setting federated --labels local \
      2>&1 | tee "$LOG/p3_${H}hop_fed_local.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --setting isolated --labels local --exec local \
      2>&1 | tee "$LOG/p3_${H}hop_isolated.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --setting isolated --labels local --exec relay \
      2>&1 | tee "$LOG/p3_${H}hop_isolated_train_relay_exec.log"
  RUN=results/p1/${H}hop_C3_rho0.5_balanced_s0_federated_local_relay_roberta-base_s0
  python experiments/p2_sketch_relay.py --data-dir $D --probs $RUN/test_probs.npz \
      --out results/p2/${H}hop_C3_rho0.5 --modes pull,push --pads matched:empirical --B 65536,262144 \
      --workers "$WORKERS" 2>&1 | tee "$LOG/p3_${H}hop_relay.log"
done
echo "Done. Share results/logs/p3_*.log, results/logs/p1_data_*hop_C3_rho0.5.log, results/p1/*/metrics.json, results/p2/*hop_C3_rho0.5/summary.json"
