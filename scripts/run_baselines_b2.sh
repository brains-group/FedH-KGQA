#!/usr/bin/env bash
# Baseline batch 2: TransferNet (centralized, answers-only training) and FL-KG-QA (FedR-style
# federated TransE + per-client KEQA-style QA, as in Gunti et al. 2025), same partitions and test sets, seed 0.
# Part b also finishes the WebQSP dev grid of batch 1 (FedE and FedR with entity lr 1e-4).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_baselines_b2.sh a    # MetaQA 2-hop and 3-hop
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_baselines_b2.sh b    # PQ-3H and WebQSP
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PART=${1:-a}
LOG=results/logs/baselines_v2; mkdir -p "$LOG"
OUT=results/baselines_v2

tn () {  # TransferNet: $1 name, $2 data dir, then extra args
  python experiments/b_transfernet.py --data-dir "$2" --out "$OUT/$1" "${@:3}" 2>&1 | tee "$LOG/${1}_transfernet.log"
}
kg () {  # b_kgqa method: $1 name, $2 method, $3 out root, $4 log dir, $5 data dir, then extra args
  mkdir -p "$4"
  python experiments/b_kgqa.py --data-dir "$5" --method "$2" --out "$3/$1" "${@:6}" 2>&1 | tee "$4/${1}_$2.log"
}

M2=data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
M3=data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr
PQ=data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta
WQ=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
KM="--kge-epochs 100 --kge-rounds 20 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 1"
KP="--kge-epochs 200 --kge-rounds 40 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 5"
KW="--kge-epochs 60 --kge-rounds 20 --kge-local-epochs 3 --kge-neg 4096 --qa-rounds 50 --qa-local-epochs 3 --k-max 2"

if [ "$PART" = a ]; then
  tn metaqa2 $M2 --epochs 10 --batch-size 64
  tn metaqa3 $M3 --epochs 10 --batch-size 64
  kg metaqa2 flkgqa $OUT "$LOG" $M2 $KM
  kg metaqa3 flkgqa $OUT "$LOG" $M3 $KM
else
  tn PQ-3H $PQ --epochs 100 --batch-size 32
  kg PQ-3H flkgqa $OUT "$LOG" $PQ $KP
  tn webqsp $WQ --epochs 50 --batch-size 8 --eval-batch-size 8 --k-max 2
  kg webqsp flkgqa $OUT "$LOG" $WQ $KW --local-full-f1 0.5
  for M in fede fedr; do
    kg webqsp $M results/baselines_wq_elr1e-4 results/logs/baselines_wq_elr1e-4 $WQ $KW --ent-lr=1e-4
  done
fi
echo "Done ($PART)."
