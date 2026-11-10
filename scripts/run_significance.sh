#!/usr/bin/env bash
# Paired bootstrap of FedH-KGQA (seed 0) against every other main-table method, per dataset (CPU, minutes).
# WebQSP: our settings from the fully masked rerun (results/webqsp_m2); baselines from results/baselines_v2.
# Usage: bash scripts/run_significance.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/p5; mkdir -p "$LOG"
FED=federated_local+disc5_relay_roberta-base_s0
CEN=centralized_gold_relay_roberta-base_s0
ISO=isolated_local_local_roberta-base_s0
ISR=isolated_local_relay_roberta-base_s0
sig () {  # $1 name, $2 file tag, $3 ours, then other run dirs
  python experiments/p5_significance.py --dataset "$1" --ours "$3" --others "${@:4}" --out results/p5/sig_$2.json
}
{
  P=results/p3v2/metaqa2/2hop_C3_rho0.5_balanced_s0
  sig "MetaQA 2-hop" metaqa2 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR results/baselines_v2/metaqa2/*
  P=results/p3v2/metaqa3/3hop_C3_rho0.5_balanced_s0_nr
  sig "MetaQA 3-hop" metaqa3 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR results/baselines_v2/metaqa3/*
  P=results/p3v2/PQ-3H/3hop_C3_rho0.5_balanced_s0_ta
  sig "PQ-3H" pq3 ${P}_$FED ${P}_$CEN ${P}_$ISO ${P}_$ISR results/baselines_v2/PQ-3H/*
  W=results/webqsp_m2
  sig "WebQSP" webqsp $W/fed_id_disc5_s0 $W/centralized_id_s0 $W/isolated_id $W/isolated_id_relay_exec \
      results/baselines_v2/webqsp/*
} 2>&1 | tee "$LOG/significance.log"
echo "Done. Share: tar czf p5_sig.tar.gz $LOG results/p5"
