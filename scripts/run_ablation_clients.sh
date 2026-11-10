#!/usr/bin/env bash
# Ablation: number of clients C = 5 and 10 (C = 3 is the main setting) on MetaQA 3-hop and PQ-3H.
# Builds the partitions (target rho = 0.5; the reachable rho is reported in stats.json), then runs
# FedH-KGQA (local labels + discovery B = 5) and Isolated with the main-table settings, seed 0.
# Usage: CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_ablation_clients.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/ablation_clients; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
SP="--sparsify-rel 0.1 --sparsify-top 3"
B="--encoder hf --model-name roberta-base --workers $WORKERS $SP"
for C in ${CLIENTS:-5 10}; do
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/MetaQA --hops 3 --clients $C --rho 0.5 --seed 0 \
      --workers "$WORKERS" --no-return 2>&1 | tee "$LOG/data_metaqa3_C$C.log"
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/PathQuestion/PQ-3H --out data/processed/pathquestion/PQ-3H \
      --hops 3 --clients $C --rho 0.5 --seed 0 --workers "$WORKERS" --topic-answer 2>&1 | tee "$LOG/data_pq3_C$C.log"
  D=data/processed/metaqa/3hop/C${C}_rho0.5_balanced_s0_nr
  C1="$B --data-dir $D --out results/ablation_clients/metaqa3 --rounds 5"
  python experiments/p1_reasoner.py $C1 --setting federated --labels local --discover 5 2>&1 | tee "$LOG/metaqa3_C${C}_fed_disc5.log"
  python experiments/p1_reasoner.py $C1 --setting isolated --labels local --exec local 2>&1 | tee "$LOG/metaqa3_C${C}_isolated.log"
  D=data/processed/pathquestion/PQ-3H/3hop/C${C}_rho0.5_balanced_s0_ta
  C2="$B --data-dir $D --out results/ablation_clients/PQ-3H --rounds 20 --local-epochs 5 --discover-start 4 --discover-every 4"
  python experiments/p1_reasoner.py $C2 --setting federated --labels local --discover 5 2>&1 | tee "$LOG/pq3_C${C}_fed_disc5.log"
  python experiments/p1_reasoner.py $C2 --setting isolated --labels local --exec local 2>&1 | tee "$LOG/pq3_C${C}_isolated.log"
done
echo "Done. Share: tar czf ablation_clients.tar.gz $LOG results/ablation_clients/*/*/metrics.json data/processed/metaqa/3hop/C*_rho0.5_balanced_s0_nr/stats.json data/processed/pathquestion/PQ-3H/3hop/C*_ta/stats.json"
