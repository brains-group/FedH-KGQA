#!/usr/bin/env bash
# P3 (part 3): WebQSP from the RoG release (rmanluo/RoG-webqsp), C = 3, rho = 0.5.
# 1 to 2 hops per question (mixed), relations scored by their names (--rel-head text),
# relation sparsification, model-guided discovery at two budgets (B = 5 and 20).
# Usage: CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_p3_webqsp.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/webqsp; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
ENC=${ENCODER:-hf}; MODEL=${MODEL:-roberta-base}
NAME=$([ "$ENC" = bow ] && echo bow || echo "${MODEL##*/}")
RAW=${RAW:-data/raw/RoG-webqsp}
OUT=results/p3_webqsp
SP="--sparsify-rel 0.1 --sparsify-top 3"

[ -f data/raw/WebQSP/kb.txt ] || python experiments/convert_webqsp.py --rog-dir "$RAW" --out data/raw/WebQSP \
    2>&1 | tee "$LOG/convert.log"
python experiments/p1_prepare_data.py --metaqa-dir data/raw/WebQSP --out data/processed/webqsp --hops 2 --mixed-hops \
    --clients 3 --rho 0.5 --seed 0 --max-frontier ${MAXF:-20000} --workers "$WORKERS" 2>&1 | tee "$LOG/data.log"
D=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
python experiments/diag_gold.py --data-dir $D --split train --n 2000 --json "$LOG/diag.json" 2>&1 | tee "$LOG/diag.log"

C="--data-dir $D --encoder $ENC --model-name $MODEL --workers $WORKERS --out $OUT $SP --rounds ${ROUNDS:-20} \
   --local-epochs ${LOCAL_EPOCHS:-5} --k-max 2 ${EXTRA:-}"
python experiments/p1_reasoner.py $C --setting centralized --labels gold --rel-head id 2>&1 | tee "$LOG/centralized_idhead.log"
python experiments/p1_reasoner.py $C --setting centralized --labels gold --rel-head text 2>&1 | tee "$LOG/centralized.log"
python experiments/p1_reasoner.py $C --setting federated --labels local --rel-head text 2>&1 | tee "$LOG/fed_local.log"
python experiments/p1_reasoner.py $C --setting federated --labels local --rel-head text --discover 5 \
    --discover-start 4 --discover-every 4 2>&1 | tee "$LOG/fed_local_disc5.log"
python experiments/p1_reasoner.py $C --setting federated --labels local --rel-head text --discover 20 \
    --discover-start 2 --discover-every 2 2>&1 | tee "$LOG/fed_local_disc20.log"
python experiments/p1_reasoner.py $C --setting isolated --labels local --exec local --rel-head text 2>&1 | tee "$LOG/isolated.log"
python experiments/p1_reasoner.py $C --setting isolated --labels local --exec relay --rel-head text \
    2>&1 | tee "$LOG/isolated_train_relay_exec.log"
python experiments/p2_sketch_relay.py --data-dir $D $SP \
    --probs $OUT/2hop_C3_rho0.5_balanced_s0_federated_local+disc20_relay_${NAME}-text_s0/test_probs.npz \
    --out $OUT/relay --modes pull --pads matched:empirical --B 65536,262144 --tau 0.05,0.01 \
    --calib-questions 200 --traffic-questions 300 --workers "$WORKERS" 2>&1 | tee "$LOG/relay.log"
echo "Done. Share results/logs/webqsp/, results/p3_webqsp/*/metrics.json, results/p3_webqsp/relay/summary.json, data/processed/webqsp/2hop/*/stats.json, data/raw/WebQSP/convert_stats.json"
