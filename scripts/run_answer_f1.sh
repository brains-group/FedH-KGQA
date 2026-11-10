#!/usr/bin/env bash
# Answer-set F1 for our four settings on every dataset (CPU only, no retraining; a few minutes).
# Usage: bash scripts/run_answer_f1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/p6; mkdir -p "$LOG"
FED=federated_local+disc5_relay_roberta-base_s0
CEN=centralized_gold_relay_roberta-base_s0
ISO=isolated_local_local_roberta-base_s0
ISR=isolated_local_relay_roberta-base_s0
f1 () {  # $1 name, $2 file tag, then run dirs
  python experiments/p6_answer_f1.py --dataset "$1" --runs "${@:3}" --out results/p6/f1_$2.json
}
{
  P=results/p3v2/metaqa2/2hop_C3_rho0.5_balanced_s0
  f1 "MetaQA 2-hop" metaqa2 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR
  P=results/p3v2/metaqa3/3hop_C3_rho0.5_balanced_s0_nr
  f1 "MetaQA 3-hop" metaqa3 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR
  P=results/p3v2/PQ-3H/3hop_C3_rho0.5_balanced_s0_ta
  f1 "PQ-3H" pq3 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR
  W=results/webqsp_m2
  f1 "WebQSP" webqsp $W/fed_id_disc5_s0 $W/centralized_id_s0 $W/isolated_id $W/isolated_id_relay_exec
} 2>&1 | tee "$LOG/answer_f1.log"
echo "Done. Share: tar czf p6_f1.tar.gz $LOG results/p6"
