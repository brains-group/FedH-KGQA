#!/usr/bin/env bash
# MetaQA 2-hop with the final settings used on MetaQA 3-hop and PathQuestion (P3 v2):
# relation sparsification at execution, model-guided discovery (B = 5) as the main FedH-KGQA run,
# and the pull relay with matched padding. Reuses the P1 partition (C = 3, rho = 0.5).
# Usage: CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_metaqa2_final.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=results/logs/v2; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
D=data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
OUT=results/p3v2/metaqa2
SP="--sparsify-rel 0.1 --sparsify-top 3"
C="--data-dir $D --encoder hf --model-name roberta-base --workers $WORKERS --out $OUT $SP --rounds 5"
P=metaqa2

python experiments/p1_reasoner.py $C --setting centralized --labels gold 2>&1 | tee "$LOG/${P}_centralized.log"
python experiments/p1_reasoner.py $C --setting federated --labels local 2>&1 | tee "$LOG/${P}_fed_local.log"
python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 2>&1 | tee "$LOG/${P}_fed_local_disc5.log"
python experiments/p1_reasoner.py $C --setting isolated --labels local --exec local 2>&1 | tee "$LOG/${P}_isolated.log"
python experiments/p1_reasoner.py $C --setting isolated --labels local --exec relay 2>&1 | tee "$LOG/${P}_isolated_train_relay_exec.log"
python experiments/p2_sketch_relay.py --data-dir $D $SP \
    --probs $OUT/2hop_C3_rho0.5_balanced_s0_federated_local+disc5_relay_roberta-base_s0/test_probs.npz \
    --out $OUT/relay --modes pull --pads matched:empirical --B 65536 --tau 0.05 \
    --calib-questions 500 --traffic-questions 300 --workers "$WORKERS" 2>&1 | tee "$LOG/${P}_relay.log"
echo "Done. Share: tar czf metaqa2_final_results.tar.gz results/logs/v2/metaqa2_* results/p3v2/metaqa2/*/metrics.json results/p3v2/metaqa2/relay/summary.json"
