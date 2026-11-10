#!/usr/bin/env bash
# WebQSP v3: same settings as v2, with one fix to the local labels.
# In v1/v2 a client searched its own view only at the hop count found on the pooled KG, which a
# client cannot know. Now it searches every length up to K_max = 2 and keeps the best (shorter wins
# ties). First-hop candidates are also no longer gated by the (pooled) cross-client flag.
# Rebuilds the WebQSP partition (gold-label cache is reused), then reruns every setting.
# Usage: CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_p3_webqsp_v3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/webqsp_v3; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
python experiments/p1_prepare_data.py --metaqa-dir data/raw/WebQSP --out data/processed/webqsp --hops 2 --mixed-hops \
    --clients 3 --rho 0.5 --seed 0 --max-frontier ${MAXF:-20000} --workers "$WORKERS" 2>&1 | tee "$LOG/data.log"
D=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
OUT=results/p3_webqsp_v3
C="--data-dir $D --encoder hf --model-name roberta-base --workers $WORKERS --out $OUT --sparsify-rel 0.1 --sparsify-top 3 \
   --k-max 2 --local-full-f1 0.5 --discover-min-f1 0.5 --rel-head id"
F="--setting federated --labels local --rounds 50 --local-epochs 3"
DISC="--discover-start 4 --discover-every 4"
I="--setting isolated --labels local --rounds 20 --local-epochs 5"

python experiments/p1_reasoner.py $C --setting centralized --labels gold --rounds 50 --local-epochs 3 --tag centralized_id \
    2>&1 | tee "$LOG/centralized_id.log"
python experiments/p1_reasoner.py $C $F --tag fed_id_local 2>&1 | tee "$LOG/fed_id_local.log"
python experiments/p1_reasoner.py $C $F --discover 5 $DISC --tag fed_id_disc5 2>&1 | tee "$LOG/fed_id_disc5.log"
python experiments/p1_reasoner.py $C $F --discover 20 $DISC --tag fed_id_disc20 2>&1 | tee "$LOG/fed_id_disc20.log"
python experiments/p1_reasoner.py $C $I --exec local --tag isolated_id 2>&1 | tee "$LOG/isolated_id.log"
python experiments/p1_reasoner.py $C $I --exec relay --tag isolated_id_relay_exec 2>&1 | tee "$LOG/isolated_id_relay_exec.log"
python experiments/p2_sketch_relay.py --data-dir $D --sparsify-rel 0.1 --sparsify-top 3 \
    --probs $OUT/fed_id_disc5/test_probs.npz --out $OUT/relay --modes pull --pads matched:empirical \
    --B 262144 --tau 0.05 --calib-questions 200 --traffic-questions 300 --workers "$WORKERS" 2>&1 | tee "$LOG/relay.log"
echo "Done. Share: tar czf webqsp_v3_results.tar.gz results/logs/webqsp_v3 results/p3_webqsp_v3/*/metrics.json results/p3_webqsp_v3/relay/summary.json"
