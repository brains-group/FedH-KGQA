#!/usr/bin/env bash
# P2: the trained reasoner through the identifier-hiding relay (sketches + padding, no noise).
# Uses the test probabilities saved by the P1 federated run; builds the C = 5 and C = 10
# partitions and checks the relay there too (load grows with N * L).
# Usage: WORKERS=8 bash scripts/run_p2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
META=${META:-data/raw/MetaQA}
D=data/processed/metaqa/2hop
RUN=${RUN:-results/p1/2hop_C3_rho0.5_balanced_s0_federated_local_relay_roberta-base_s0}
PROBS=$RUN/test_probs.npz

if [ ! -f "$PROBS" ]; then
  echo "No $PROBS: re-running the P1 federated run to save the test probabilities (needs the GPU)"
  python experiments/p1_reasoner.py --data-dir $D/C3_rho0.5_balanced_s0 --setting federated --labels local \
      --encoder hf --model-name roberta-base --rounds 5 --workers "$WORKERS" 2>&1 | tee "$LOG/p1_fed_relay_rerun.log"
fi

python experiments/p2_sketch_relay.py --data-dir $D/C3_rho0.5_balanced_s0 --probs "$PROBS" \
    --out results/p2/C3_rho0.5 --workers "$WORKERS" 2>&1 | tee "$LOG/p2_C3.log"

for C in 5 10; do
  [ -d $D/C${C}_rho0.5_balanced_s0 ] || python experiments/p1_prepare_data.py --metaqa-dir "$META" --hops 2 \
      --clients $C --rho 0.5 --seed 0 --workers "$WORKERS" 2>&1 | tee "$LOG/p1_data_2hop_C${C}_rho0.5.log"
  python experiments/p2_sketch_relay.py --data-dir $D/C${C}_rho0.5_balanced_s0 --probs "$PROBS" \
      --out results/p2/C${C}_rho0.5 --B 262144 --mass 0.25 --workers "$WORKERS" 2>&1 | tee "$LOG/p2_C${C}.log"
done
echo "Done. Share results/logs/p2_*.log, results/p2/*/summary.json and the new stats.json files"
