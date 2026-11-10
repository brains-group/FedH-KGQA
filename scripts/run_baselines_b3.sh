#!/usr/bin/env bash
# Baseline batch 3: PFedEG, PFLU (federated KGE + the same EmbedKGQA-style federated QA head as FedE/FedR)
# and FedNGDB (federated neural graph database answering the relation path that FedH-KGQA's reasoner
# predicts). Same partitions and test sets, seed 0. WebQSP uses the entity-embedding setting that dev
# selected for all batch-1 methods (fine-tuning, lr 1e-4).
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_baselines_b3.sh a    # MetaQA 2-hop and 3-hop
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_baselines_b3.sh b    # PQ-3H and WebQSP
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PART=${1:-a}
LOG=results/logs/baselines_v2; mkdir -p "$LOG"
OUT=results/baselines_v2

kg () {  # $1 name, $2 method, $3 data dir, then extra args
  python experiments/b_kgqa.py --data-dir "$3" --method "$2" --out "$OUT/$1" "${@:4}" 2>&1 | tee "$LOG/${1}_$2.log"
}
ng () {  # $1 name, $2 data dir, $3 probs, then extra args
  python experiments/b_fedngdb.py --data-dir "$2" --probs "$3" --out "$OUT/$1" "${@:4}" 2>&1 | tee "$LOG/${1}_fedngdb.log"
}

M2=data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
M3=data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr
PQ=data/processed/pathquestion/PQ-3H/3hop/C3_rho0.5_balanced_s0_ta
WQ=data/processed/webqsp/2hop/C3_rho0.5_balanced_s0
P2=results/p3v2/metaqa2/2hop_C3_rho0.5_balanced_s0_federated_local+disc5_relay_roberta-base_s0/test_probs.npz
P3=results/p3v2/metaqa3/3hop_C3_rho0.5_balanced_s0_nr_federated_local+disc5_relay_roberta-base_s0/test_probs.npz
PP=results/p3v2/PQ-3H/3hop_C3_rho0.5_balanced_s0_ta_federated_local+disc5_relay_roberta-base_s0/test_probs.npz
PW=results/p3_webqsp_v3/fed_id_disc5/test_probs.npz
KM="--kge-epochs 100 --kge-rounds 20 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 1"
KP="--kge-epochs 200 --kge-rounds 40 --kge-local-epochs 5 --qa-rounds 20 --qa-local-epochs 5"
KW="--kge-epochs 60 --kge-rounds 20 --kge-local-epochs 3 --kge-neg 4096 --qa-rounds 50 --qa-local-epochs 3 --ent-lr=1e-4"

for f in $P2 $P3 $PP $PW; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
if [ "$PART" = a ]; then
  for M in pfedeg pflu; do kg metaqa2 $M $M2 $KM; kg metaqa3 $M $M3 $KM; done
  ng metaqa2 $M2 $P2 --rounds 20 --local-epochs 5
  ng metaqa3 $M3 $P3 --rounds 20 --local-epochs 5
else
  for M in pfedeg pflu; do kg PQ-3H $M $PQ $KP; kg webqsp $M $WQ $KW; done
  ng PQ-3H $PQ $PP --rounds 40 --local-epochs 5
  ng webqsp $WQ $PW --rounds 20 --local-epochs 3
fi
echo "Done ($PART)."
