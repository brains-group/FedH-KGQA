#!/usr/bin/env bash
# P3 (part 2): PathQuestion (PQ-2H, PQ-3H, PQL-2H, PQL-3H), C = 3, rho = 0.5.
# Converts the IRN release to the MetaQA layout, then per variant: partition, Centralized,
# FedH-KGQA (federated, local labels), Isolated, Isolated training + relay execution, and the
# sketch relay (pull and push, distribution-matched padding) on the FedH-KGQA model.
# The datasets are small (about 1K to 5K questions), so training runs more rounds.
# Usage: CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_p3_pathquestion.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
VARIANTS=${VARIANTS:-"PQ-2H PQ-3H PQL-2H PQL-3H"}
COMMON="--encoder hf --model-name ${MODEL:-roberta-base} --rounds ${ROUNDS:-20} --workers $WORKERS"

python experiments/convert_pathquestion.py --irn-dir data/raw/IRN/PathQuestion --out data/raw/PathQuestion \
    2>&1 | tee "$LOG/pq_convert.log"

for V in $VARIANTS; do
  H=${V##*-}; H=${H%H}
  D=data/processed/pathquestion/$V/${H}hop/C3_rho0.5_balanced_s0
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/PathQuestion/$V --out data/processed/pathquestion/$V \
      --hops $H --clients 3 --rho 0.5 --seed 0 --workers "$WORKERS" 2>&1 | tee "$LOG/pq_data_${V}.log"
  OUT=results/p1_pq/$V
  python experiments/p1_reasoner.py --data-dir $D $COMMON --out $OUT --setting centralized --labels gold \
      2>&1 | tee "$LOG/pq_${V}_centralized.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --out $OUT --setting federated --labels local \
      2>&1 | tee "$LOG/pq_${V}_fed_local.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --out $OUT --setting isolated --labels local --exec local \
      2>&1 | tee "$LOG/pq_${V}_isolated.log"
  python experiments/p1_reasoner.py --data-dir $D $COMMON --out $OUT --setting isolated --labels local --exec relay \
      2>&1 | tee "$LOG/pq_${V}_isolated_train_relay_exec.log"
  python experiments/p2_sketch_relay.py --data-dir $D \
      --probs $OUT/${H}hop_C3_rho0.5_balanced_s0_federated_local_relay_roberta-base_s0/test_probs.npz \
      --out results/p2_pq/$V --modes pull,push --pads matched:empirical --B 65536 \
      --calib-questions 40 --traffic-questions 150 --workers "$WORKERS" 2>&1 | tee "$LOG/pq_${V}_relay.log"
done
echo "Done. Share results/logs/pq_*.log, results/p1_pq/*/*/metrics.json, results/p2_pq/*/summary.json"
