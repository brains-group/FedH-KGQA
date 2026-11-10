#!/usr/bin/env bash
# WebQSP with the topic mention masked in every question (fix_webqsp_mask.py), so clients never
# see the topic name. Same partition, labels and settings as WebQSP v3; only question_masked changes.
#   fix   build the masked copy of the partition and report how many mentions were fixed (CPU, ~1 min)
#   a     Centralized seeds 0-2, Isolated, Isolated training + relay execution
#   b     FedH-KGQA (local labels + discovery B = 5) seeds 0-2
# Usage: bash scripts/run_webqsp_masked.sh fix
#        CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_webqsp_masked.sh a
#        CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_webqsp_masked.sh b
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PART=${1:-fix}
WORKERS=${WORKERS:-8}
LOG=results/logs/webqsp_m2; mkdir -p "$LOG"
D0=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
D=${D0}_m2
OUT=results/webqsp_m2
C="--data-dir $D --encoder hf --model-name roberta-base --workers $WORKERS --out $OUT --sparsify-rel 0.1 --sparsify-top 3 \
   --k-max 2 --local-full-f1 0.5 --discover-min-f1 0.5 --rel-head id"
F="--setting federated --labels local --rounds 50 --local-epochs 3 --discover 5 --discover-start 4 --discover-every 4"
I="--setting isolated --labels local --rounds 20 --local-epochs 5"

if [ "$PART" = fix ]; then
  python experiments/fix_webqsp_mask.py --rog-dir ${RAW:-data/raw/RoG-webqsp} --part $D0 --out $D 2>&1 | tee "$LOG/fix.log"
  echo "Share: the [train]/[dev]/[test] lines above and: head -40 $D/mask_examples.txt; grep -A40 unresolved $D/mask_examples.txt"
  exit 0
fi
[ -f $D/test.jsonl ] || { echo "run 'bash scripts/run_webqsp_masked.sh fix' first"; exit 1; }
if [ "$PART" = a ]; then
  for S in 0 1 2; do
    python experiments/p1_reasoner.py $C --setting centralized --labels gold --rounds 50 --local-epochs 3 --seed $S \
        --tag centralized_id_s$S 2>&1 | tee "$LOG/centralized_s$S.log"
  done
  python experiments/p1_reasoner.py $C $I --exec local --tag isolated_id 2>&1 | tee "$LOG/isolated.log"
  python experiments/p1_reasoner.py $C $I --exec relay --tag isolated_id_relay_exec 2>&1 | tee "$LOG/isolated_relay_exec.log"
else
  for S in 0 1 2; do
    python experiments/p1_reasoner.py $C $F --seed $S --tag fed_id_disc5_s$S 2>&1 | tee "$LOG/fed_disc5_s$S.log"
  done
fi
echo "Done ($PART). Share: grep -h '\[test\]' $LOG/*.log; tar czf webqsp_m2_$PART.tar.gz $LOG $OUT/*/metrics.json"
