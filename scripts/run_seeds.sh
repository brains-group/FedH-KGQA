#!/usr/bin/env bash
# Seeds 1 and 2 for Centralized and FedH-KGQA (local labels + discovery B = 5) on the four
# main-table datasets, with the same settings as the seed-0 runs. The partition stays at seed 0;
# the seed changes model initialisation, data order and discovery.
# Usage: CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/seeds; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
SP="--sparsify-rel 0.1 --sparsify-top 3"
B="--encoder hf --model-name roberta-base --workers $WORKERS $SP"

for S in ${SEEDS:-1 2}; do
  # MetaQA 2-hop and 3-hop: 5 rounds, discovery every 2 rounds
  for H in 2 3; do
    D=data/processed/metaqa/${H}hop/C3_rho0.5_balanced_s0; [ "$H" = 3 ] && D=${D}_nr
    C="$B --data-dir $D --out results/seeds/metaqa$H --rounds 5 --seed $S"
    python experiments/p1_reasoner.py $C --setting centralized --labels gold 2>&1 | tee "$LOG/metaqa${H}_centralized_s$S.log"
    python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 2>&1 | tee "$LOG/metaqa${H}_fed_disc5_s$S.log"
  done
  # PQ-3H: 20 rounds of 5 local epochs, discovery every 4 rounds from round 4
  D=data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta
  C="$B --data-dir $D --out results/seeds/PQ-3H --rounds 20 --local-epochs 5 --discover-start 4 --discover-every 4 --seed $S"
  python experiments/p1_reasoner.py $C --setting centralized --labels gold 2>&1 | tee "$LOG/pq3_centralized_s$S.log"
  python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 2>&1 | tee "$LOG/pq3_fed_disc5_s$S.log"
  # WebQSP: 50 rounds of 3 local epochs, F1 >= 0.5 labels, K_max = 2
  D=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
  C="$B --data-dir $D --out results/seeds/webqsp --rounds 50 --local-epochs 3 --k-max 2 --rel-head id \
     --local-full-f1 0.5 --discover-min-f1 0.5 --discover-start 4 --discover-every 4 --seed $S"
  python experiments/p1_reasoner.py $C --setting centralized --labels gold --tag centralized_id_s$S \
      2>&1 | tee "$LOG/webqsp_centralized_s$S.log"
  python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 --tag fed_id_disc5_s$S \
      2>&1 | tee "$LOG/webqsp_fed_disc5_s$S.log"
done
echo "Done. Share: tar czf seeds_results.tar.gz results/logs/seeds results/seeds/*/*/metrics.json"
