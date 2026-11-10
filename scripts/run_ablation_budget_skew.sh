#!/usr/bin/env bash
# Ablations 3 and 4, seed 0, main-table settings (C = 3, rho = 0.5).
#   budget  label-discovery budget curve: Hits@1 against relay queries per training question for
#           local labels only, discovery B = 5, B = 20 and exhaustive search (every first hop the owner
#           sees, then every relation; finds the gold path). Local-only and B = 5 are the main-table runs;
#           WebQSP B = 20 is the v3 run. This adds the missing points.
#   skew    question-type skew: each question type (gold relation path) is trained at only 1 or 2 of the
#           3 clients; FedH-KGQA vs Isolated vs Isolated training + relay execution. Test set unchanged.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablation_budget_skew.sh budget
#        CUDA_VISIBLE_DEVICES=1 bash scripts/run_ablation_budget_skew.sh skew
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PART=${1:-budget}
WORKERS=${WORKERS:-8}
LOG=results/logs/ablation_$PART; mkdir -p "$LOG"
OUT=results/ablation_$PART
B="--encoder hf --model-name roberta-base --workers $WORKERS --sparsify-rel 0.1 --sparsify-top 3 --out $OUT"
M3="--data-dir data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr --rounds 5"
PQ="--data-dir data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta --rounds 20 --local-epochs 5 --discover-start 4 --discover-every 4"
WQ="--data-dir data/processed/webqsp/2hop/C3_rho0.5_balanced_s0 --rounds 50 --local-epochs 3 --k-max 2 --rel-head id --local-full-f1 0.5 --discover-min-f1 0.5 --discover-start 4 --discover-every 4"
F="--setting federated"

if [ "$PART" = budget ]; then
  python experiments/p1_reasoner.py $B $M3 $F --labels local --discover 20 --tag metaqa3_disc20 2>&1 | tee "$LOG/metaqa3_disc20.log"
  python experiments/p1_reasoner.py $B $M3 $F --labels relay --tag metaqa3_exhaustive 2>&1 | tee "$LOG/metaqa3_exhaustive.log"
  python experiments/p1_reasoner.py $B $PQ $F --labels local --discover 20 --tag pq3_disc20 2>&1 | tee "$LOG/pq3_disc20.log"
  python experiments/p1_reasoner.py $B $PQ $F --labels relay --tag pq3_exhaustive 2>&1 | tee "$LOG/pq3_exhaustive.log"
  python experiments/p1_reasoner.py $B $WQ $F --labels relay --tag webqsp_exhaustive 2>&1 | tee "$LOG/webqsp_exhaustive.log"
else
  for K in 1 2; do
    for D in M3 PQ; do
      N=$([ $D = M3 ] && echo metaqa3 || echo pq3); A=${!D}
      python experiments/p1_reasoner.py $B $A $F --labels local --discover 5 --type-skew $K --tag ${N}_skew${K}_fed_disc5 \
          2>&1 | tee "$LOG/${N}_skew${K}_fed_disc5.log"
      python experiments/p1_reasoner.py $B $A --setting isolated --labels local --exec local --type-skew $K \
          --tag ${N}_skew${K}_isolated 2>&1 | tee "$LOG/${N}_skew${K}_isolated.log"
      python experiments/p1_reasoner.py $B $A --setting isolated --labels local --exec relay --type-skew $K \
          --tag ${N}_skew${K}_isolated_relay_exec 2>&1 | tee "$LOG/${N}_skew${K}_isolated_relay_exec.log"
    done
  done
fi
echo "Done ($PART). Share: tar czf ablation_$PART.tar.gz $LOG $OUT/*/metrics.json"
