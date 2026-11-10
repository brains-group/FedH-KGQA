#!/usr/bin/env bash
# P3 v2: reruns MetaQA 3-hop and PathQuestion with the fixes found in P3:
#   - execution rules per dataset: PathQuestion allows the topic entity as an answer (--topic-answer);
#     MetaQA 3-hop forbids paths back through the topic entity (--no-return) if the diagnostic shows
#     that it matches the answer sets better
#   - relation sparsification at execution (per hop keep p >= 0.1 * max, at most 3, renormalise)
#   - model-guided label discovery (top-5 predicted paths executed through the relay) as a federated variant
#   - more local epochs on the small PathQuestion sets
# Usage (two GPUs, two tmux sessions):
#   CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_p3v2.sh metaqa
#   CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_p3v2.sh pq
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PART=${1:-metaqa}
LOG=results/logs/v2; mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
SP="--sparsify-rel 0.1 --sparsify-top 3"
EXTRA=${EXTRA:-}

run_all () {  # $1 data dir, $2 output dir, $3 log prefix, $4 hop count, then extra training args
  local D=$1 OUT=$2 P=$3 H=$4; shift 4
  local ENC=${ENCODER:-hf} NAME=${MODEL:-roberta-base}
  [ "$ENC" = bow ] && NAME=bow
  local C="--data-dir $D --encoder $ENC --model-name ${MODEL:-roberta-base} --workers $WORKERS --out $OUT $SP $EXTRA $*"
  python experiments/p1_reasoner.py $C --setting centralized --labels gold 2>&1 | tee "$LOG/${P}_centralized.log"
  python experiments/p1_reasoner.py $C --setting federated --labels local 2>&1 | tee "$LOG/${P}_fed_local.log"
  python experiments/p1_reasoner.py $C --setting federated --labels local --discover 5 2>&1 | tee "$LOG/${P}_fed_local_disc5.log"
  python experiments/p1_reasoner.py $C --setting isolated --labels local --exec local 2>&1 | tee "$LOG/${P}_isolated.log"
  python experiments/p1_reasoner.py $C --setting isolated --labels local --exec relay 2>&1 | tee "$LOG/${P}_isolated_train_relay_exec.log"
  local TAG; TAG=$(basename "$D")
  python experiments/p2_sketch_relay.py --data-dir $D $SP \
      --probs $OUT/${H}hop_${TAG}_federated_local+disc5_relay_${NAME}_s0/test_probs.npz \
      --out $OUT/relay --modes pull --pads matched:empirical --B 65536 --tau 0.05,0.01 \
      --calib-questions ${CALIB:-500} --traffic-questions 300 --workers "$WORKERS" 2>&1 | tee "$LOG/${P}_relay.log"
}

if [ "$PART" = metaqa ]; then
  D0=data/processed/metaqa/3hop/C3_rho0.5_balanced_s0
  python experiments/diag_gold.py --data-dir $D0 --split train --n 3000 --json "$LOG/metaqa3_diag.json" \
      2>&1 | tee "$LOG/metaqa3_diag.log"
  if python -c "import json,sys; r=json.load(open('$LOG/metaqa3_diag.json'))['rules']; sys.exit(0 if r['no_return']['oracle_hits1'] > r['set']['oracle_hits1'] + 0.02 else 1)"; then
    echo "[p3v2] no_return matches MetaQA 3-hop answers better: rebuilding the partition with --no-return"
    python experiments/p1_prepare_data.py --metaqa-dir ${META:-data/raw/MetaQA} --hops 3 --clients 3 --rho 0.5 \
        --seed 0 --workers "$WORKERS" --no-return 2>&1 | tee "$LOG/metaqa3_data_nr.log"
    D=${D0}_nr
  else
    echo "[p3v2] no_return does not help: keeping the plain partition"
    D=$D0
  fi
  run_all $D results/p3v2/metaqa3 metaqa3 3 --rounds 5
else
  python experiments/convert_pathquestion.py --irn-dir data/raw/IRN/PathQuestion --out data/raw/PathQuestion \
      2>&1 | tee "$LOG/pq_convert.log"
  for V in ${VARIANTS:-PQ-2H PQ-3H PQL-2H PQL-3H}; do
    H=${V##*-}; H=${H%H}
    python experiments/p1_prepare_data.py --metaqa-dir data/raw/PathQuestion/$V --out data/processed/pathquestion/$V \
        --hops $H --clients 3 --rho 0.5 --seed 0 --workers "$WORKERS" --topic-answer 2>&1 | tee "$LOG/pq_data_${V}.log"
    CALIB=40 run_all data/processed/pathquestion/$V/${H}hop/C3_rho0.5_balanced_s0_ta results/p3v2/$V pq_$V $H \
        --rounds 20 --local-epochs 5 --discover-start 4 --discover-every 4
  done
fi
echo "Done ($PART). Share results/logs/v2/ and results/p3v2/*/*/metrics.json and results/p3v2/*/relay/summary.json"
