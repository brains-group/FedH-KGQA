#!/usr/bin/env bash
# WebQSP v2: fixes from the first WebQSP run.
#  - federated runs were still improving at round 20: 50 rounds of 3 local epochs
#  - discovery accepted only exact answer matches, but only 75% of WebQSP gold paths are exact:
#    accept the best path when its answer F1 >= 0.5; same threshold for a client's own path
#  - the id head beat the name head centrally (0.672 vs 0.639): run federated with both
# Reuses data/processed/webqsp from run_p3_webqsp.sh.
# Usage: CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_p3_webqsp_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/webqsp_v2; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
ENC=${ENCODER:-hf}; MODEL=${MODEL:-roberta-base}
D=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
OUT=results/p3_webqsp_v2
C="--data-dir $D --encoder $ENC --model-name $MODEL --workers $WORKERS --out $OUT --sparsify-rel 0.1 --sparsify-top 3 \
   --k-max 2 --local-full-f1 0.5 --discover-min-f1 0.5 ${EXTRA:-}"
F="--setting federated --labels local --rounds ${ROUNDS:-50} --local-epochs ${LOCAL_EPOCHS:-3}"
DISC="--discover-start 4 --discover-every 4"

python experiments/p1_reasoner.py $C $F --rel-head id --tag fed_id_local 2>&1 | tee "$LOG/fed_id_local.log"
python experiments/p1_reasoner.py $C $F --rel-head id --discover 5 $DISC --tag fed_id_disc5 2>&1 | tee "$LOG/fed_id_disc5.log"
python experiments/p1_reasoner.py $C $F --rel-head id --discover 20 $DISC --tag fed_id_disc20 2>&1 | tee "$LOG/fed_id_disc20.log"
python experiments/p1_reasoner.py $C $F --rel-head text --discover 5 $DISC --tag fed_text_disc5 2>&1 | tee "$LOG/fed_text_disc5.log"
I="--setting isolated --labels local --rounds ${ISO_ROUNDS:-20} --local-epochs 5 --rel-head id"
python experiments/p1_reasoner.py $C $I --exec local --tag isolated_id 2>&1 | tee "$LOG/isolated_id.log"
python experiments/p1_reasoner.py $C $I --exec relay --tag isolated_id_relay_exec 2>&1 | tee "$LOG/isolated_id_relay_exec.log"
python experiments/p2_sketch_relay.py --data-dir $D --sparsify-rel 0.1 --sparsify-top 3 \
    --probs $OUT/fed_id_disc20/test_probs.npz --out $OUT/relay --modes pull --pads matched:empirical \
    --B 262144 --tau 0.05 --calib-questions 200 --traffic-questions 300 --workers "$WORKERS" 2>&1 | tee "$LOG/relay.log"
echo "Done. Share: tar czf webqsp_v2_results.tar.gz results/logs/webqsp_v2 results/p3_webqsp_v2/*/metrics.json results/p3_webqsp_v2/relay/summary.json"
