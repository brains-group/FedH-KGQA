#!/usr/bin/env bash
# P4: privacy attacks and cost on the main-table FedH-KGQA models (seed 0).
#   relay   entity existence + traffic analysis + per-query cost on the pull sketch relay (CPU only)
#   mi      QA-pair membership inference on the released reasoner, no DP and DP (needs a GPU, ~10 min)
#   cost    communication and runtime table (CPU, after relay)
# Usage: WORKERS=8 bash scripts/run_p4_attacks.sh relay
#        CUDA_VISIBLE_DEVICES=1 bash scripts/run_p4_attacks.sh mi
#        bash scripts/run_p4_attacks.sh cost
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PART=${1:-relay}
WORKERS=${WORKERS:-8}
LOG=results/logs/p4; mkdir -p "$LOG"

M2D=data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
M3D=data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr
PQD=data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta
WQD=${WQD:-data/processed/webqsp/2hop/C3_rho0.5_balanced_s0}
M2=results/p3v2/metaqa2/2hop_C3_rho0.5_balanced_s0
M3=results/p3v2/metaqa3/3hop_C3_rho0.5_balanced_s0_nr
PQ=results/p3v2/PQ-3H/3hop_C3_rho0.5_balanced_s0_ta
WQ=${WQ:-results/p3_webqsp_v3}
FED=federated_local+disc5_relay_roberta-base_s0
CEN=centralized_gold_relay_roberta-base_s0

if [ "$PART" = relay ]; then
  for X in "metaqa2 $M2D ${M2}_$FED" "metaqa3 $M3D ${M3}_$FED" "PQ-3H $PQD ${PQ}_$FED" "webqsp $WQD $WQ/fed_id_disc5"; do
    set -- $X
    python experiments/p4_relay_attacks.py --data-dir $2 --probs $3/test_probs.npz --out results/p4/relay/$1 \
        --n-questions 1000 --workers "$WORKERS" 2>&1 | tee "$LOG/relay_$1.log"
  done
elif [ "$PART" = mi ]; then
  DP=results/ablation_dp
  python experiments/p4_membership.py --out results/p4/membership --runs \
      ${M2}_$FED ${M2}_$CEN ${M3}_$FED ${M3}_$CEN ${PQ}_$FED ${PQ}_$CEN $WQ/fed_id_disc5 $WQ/centralized_id \
      $DP/metaqa2_dp_eps8 $DP/metaqa2_dp_eps4 $DP/metaqa2_dp_eps2_g0 $DP/metaqa2_dp_eps1_g0 \
      $DP/metaqa3_dp_eps8 $DP/metaqa3_dp_eps4 $DP/metaqa3_dp_eps2 $DP/metaqa3_dp_eps1 \
      $DP/pq3_dp_eps1000 $DP/pq3_dp_eps8 $DP/pq3_dp_eps4 $DP/pq3_dp_eps2 $DP/pq3_dp_eps1 \
      2>&1 | tee "$LOG/membership.log"
else
  python experiments/p4_cost.py --out results/p4/cost.json \
      --row "MetaQA 2-hop" ${M2}_$FED ${M2}_$CEN results/p4/relay/metaqa2 \
      --row "MetaQA 3-hop" ${M3}_$FED ${M3}_$CEN results/p4/relay/metaqa3 \
      --row "PQ-3H" ${PQ}_$FED ${PQ}_$CEN results/p4/relay/PQ-3H \
      --row "WebQSP" $WQ/fed_id_disc5 $WQ/centralized_id results/p4/relay/webqsp 2>&1 | tee "$LOG/cost.log"
fi
echo "Done ($PART). Share: tar czf p4_$PART.tar.gz $LOG results/p4"
