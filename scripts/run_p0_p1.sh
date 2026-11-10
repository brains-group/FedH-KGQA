#!/usr/bin/env bash
# P0 (synthetic), P1 data for MetaQA 2-hop (C=3 at three rho values, plus the pooled C=1
# reference), then P0 on real MetaQA frontiers. Logs go to results/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs
mkdir -p "$LOG"
WORKERS=${WORKERS:-8}
META=${META:-data/raw/MetaQA}
export PYTHONUNBUFFERED=1

python experiments/p0_relay_sim.py synthetic --out results/p0/synthetic 2>&1 | tee "$LOG/p0_synthetic.log"

for RHO in 0.25 0.5 0.75; do
  python experiments/p1_prepare_data.py --metaqa-dir "$META" --hops 2 --clients 3 --rho "$RHO" \
      --seed 0 --workers "$WORKERS" 2>&1 | tee "$LOG/p1_data_2hop_C3_rho${RHO}.log"
done
python experiments/p1_prepare_data.py --metaqa-dir "$META" --hops 2 --clients 1 --rho 0 --seed 0 \
    --workers "$WORKERS" 2>&1 | tee "$LOG/p1_data_2hop_C1.log"

python experiments/p0_relay_sim.py metaqa --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \
    --out results/p0/metaqa_2hop 2>&1 | tee "$LOG/p0_metaqa_2hop.log"
echo "Done. Share results/logs/*.log and results/p0/*/summary.json"
