#!/usr/bin/env bash
# End-to-end check on a toy MetaQA-format dataset. Takes a few minutes; needs no download.
# The P1 reasoner part runs only if torch, transformers and peft are installed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/smoke
mkdir -p "$OUT"
python tests/make_toy_metaqa.py --out data/raw/ToyMetaQA
python experiments/p1_prepare_data.py --metaqa-dir data/raw/ToyMetaQA --out data/processed/toy \
    --hops 2 --clients 3 --rho 0.5 --calib-size 800 --workers 2
python experiments/p0_relay_sim.py synthetic --out "$OUT/p0_synthetic" \
    --B 65536 --d 1,3 --L 64 --mass 0.25,1.0 --sigma 0,0.2 --trials 3
python experiments/p0_relay_sim.py metaqa --data-dir data/processed/toy/2hop/C3_rho0.5_balanced_s0 \
    --out "$OUT/p0_metaqa" --B 65536 --d 3 --L 64 --mass 0.25 --sigma 0 --n-questions 100
if python -c "import torch, transformers, peft" 2>/dev/null; then
  D=data/processed/toy/2hop/C3_rho0.5_balanced_s0
  python tests/make_tiny_encoder.py --data-dir "$D" --out data/processed/tiny_encoder
  for S in "centralized --labels gold" "federated --labels relay" "federated --labels local" "isolated"; do
    python experiments/p1_reasoner.py --data-dir "$D" --setting $S --encoder bow --rounds 8 --lr 3e-3 \
        --workers 2 --out "$OUT/p1" | grep -E "^\[(labels|test)\]"
  done
  python experiments/p2_sketch_relay.py --data-dir "$D" \
      --probs "$OUT/p1/2hop_C3_rho0.5_balanced_s0_federated_relay_relay_bow_s0/test_probs.npz" \
      --out "$OUT/p2" --B 65536 --L 64 --mass 0.25 --calib-questions 50 --traffic-questions 50 --workers 2 \
      | grep -E "^\[(exact|push|pull)"
  python experiments/p1_reasoner.py --data-dir "$D" --setting federated --encoder hf \
      --model-name data/processed/tiny_encoder --rounds 2 --lr 3e-3 --workers 2 --out "$OUT/p1" | grep -E "^\[test\]"
else
  echo "torch/transformers/peft not installed: skipping the P1 reasoner check"
fi
echo "Smoke test passed. Outputs in $OUT and data/processed/toy"
