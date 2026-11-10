#!/usr/bin/env bash
# Ablation: cross-client ratio rho = 0.25 and 0.75 (0.5 is the main setting) on MetaQA 3-hop and PQ-3H, C = 3.
# Builds the partitions (the reached rho is in stats.json and in the [test]/[train] lines of the data logs),
# then runs FedH-KGQA (local labels + discovery B = 5), Isolated training + relay execution, and Isolated,
# with the main-table settings, seed 0.
# Usage: CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_ablation_rho.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/ablation_rho; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
SP="--sparsify-rel 0.1 --sparsify-top 3"
B="--encoder hf --model-name roberta-base --workers $WORKERS $SP"
for R in ${RHOS:-0.25 0.75}; do
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/MetaQA --hops 3 --clients 3 --rho $R --seed 0 \
      --workers "$WORKERS" --no-return 2>&1 | tee "$LOG/data_metaqa3_rho$R.log"
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/PathQuestion/PQ-3H --out data/processed/pathquestion/PQ-3H \
      --hops 3 --clients 3 --rho $R --seed 0 --workers "$WORKERS" --topic-answer 2>&1 | tee "$LOG/data_pq3_rho$R.log"
  for X in "metaqa3 data/processed/metaqa/3hop/C3_rho${R}_balanced_s0_nr --rounds 5" \
           "pq3 data/processed/pathquestion/PQ-3H/3hop/C3_rho${R}_balanced_s0_ta --rounds 20 --local-epochs 5 --discover-start 4 --discover-every 4"; do
    set -- $X
    N=$1; D=$2; shift 2
    C="$B --data-dir $D --out results/ablation_rho/$N $*"
    python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 --tag ${N}_rho${R}_fed_disc5 \
        2>&1 | tee "$LOG/${N}_rho${R}_fed_disc5.log"
    python experiments/p1_reasoner.py $C --setting isolated --labels local --exec relay --tag ${N}_rho${R}_isolated_relay_exec \
        2>&1 | tee "$LOG/${N}_rho${R}_isolated_relay_exec.log"
    python experiments/p1_reasoner.py $C --setting isolated --labels local --exec local --tag ${N}_rho${R}_isolated \
        2>&1 | tee "$LOG/${N}_rho${R}_isolated.log"
  done
done
echo "Done. Share: tar czf ablation_rho.tar.gz $LOG results/ablation_rho/*/*/metrics.json data/processed/metaqa/3hop/C3_rho0.*_nr/stats.json data/processed/pathquestion/PQ-3H/3hop/C3_rho0.*_ta/stats.json"
