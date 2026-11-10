#!/usr/bin/env bash
# Baseline batch 1: EmbedKGQA (centralized), FedE and FedR (federated ComplEx + EmbedKGQA QA head),
# on the same partitions and test sets as FedH-KGQA (C = 3, rho = 0.5, seed 0).
# KGE: full softmax (1-vs-all) on MetaQA and PathQuestion; sampled softmax on WebQSP (1.3M entities).
# Two parts so both GPUs can run at once:
#   CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_baselines_b1.sh a    # MetaQA 2-hop and 3-hop
#   CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash scripts/run_baselines_b1.sh b    # PQ-3H and WebQSP
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PART=${1:-a}
V=${V:-v2}          # v2: entity embeddings fine-tuned during QA training (EmbedKGQA --freeze 0)
LOG=results/logs/baselines_$V; mkdir -p "$LOG"
OUT=results/baselines_$V
METHODS=${METHODS:-embedkgqa fede fedr}

run () {  # $1 name, $2 data dir, then extra args
  local N=$1 D=$2; shift 2
  for M in $METHODS; do
    python experiments/b_kgqa.py --data-dir "$D" --method "$M" --out "$OUT/$N" "$@" \
        2>&1 | tee "$LOG/${N}_${M}.log"
  done
}

if [ "$PART" = check ]; then      # quick check: EmbedKGQA on MetaQA 2-hop only
  METHODS=embedkgqa run metaqa2 data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \
      --kge-epochs 100 --qa-rounds 20 --qa-local-epochs 1
elif [ "$PART" = a ]; then
  run metaqa2 data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \
      --kge-epochs 100 --kge-rounds 20 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 1
  run metaqa3 data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr \
      --kge-epochs 100 --kge-rounds 20 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 1
else
  run PQ-3H data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta \
      --kge-epochs 200 --kge-rounds 40 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 5
  run webqsp data/processed/webqsp/2hop/C3_rho0.5_balanced_s0 \
      --kge-epochs 60 --kge-rounds 20 --kge-local-epochs 3 --kge-neg 4096 --qa-rounds 50 --qa-local-epochs 3
fi
echo "Done ($PART). Share: tar czf baselines_b1_${V}_$PART.tar.gz $LOG $OUT/*/*/metrics.json"
