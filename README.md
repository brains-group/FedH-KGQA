# FedH-KGQA

Code for **FedH-KGQA: Privacy-Preserving Multi-Hop Question Answering over Horizontally Partitioned Knowledge Graphs** (submitted to the *Journal of Intelligent Information Systems*).

FedH-KGQA answers multi-hop questions over a knowledge graph whose entities are split across organizations that share one relation vocabulary.
A federated relation-space reasoner (RoBERTa + LoRA) predicts the relation at each hop and the hop count from the question with its topic entity masked.
An identifier-hiding relay then executes the predicted path across clients.
Clients pass hop scores in fixed-size count-min sketches keyed by OPRF pseudonyms, the server sees only their secure sum, and all messages have a fixed shape.

## Repository layout

```
fedh/
  data/        KB loading with inverse relations, masking, relation-path search, partitions
  model/       relation-space reasoner (encoder, relation and hop-count heads, path-level loss)
  fed/         federated training, exact and sketch-based relay execution, DP-SGD
  relay/       pseudonyms, count-min sketches, padding, secure-sum simulation
  baselines/   federated KG embeddings (FedE, FedR, PFedEG, PFLU) and helpers
experiments/   entry points for data preparation, training, relay, baselines, attacks and metrics
scripts/       one shell script per experiment in the paper (see the table below)
partitions/    entity-to-client assignments used in the paper (seed 0)
figures/       draw.io sources of Figs. 1 and 2, and the script for Fig. 3
tests/         toy data and a tiny encoder for the offline smoke test
```

## Setup

```bash
git clone https://github.com/brains-group/FedH-KGQA.git
cd FedH-KGQA
bash scripts/setup_env.sh        # creates the conda environment FedH-KGQA
conda activate FedH-KGQA
bash scripts/smoke_test.sh       # toy data end to end, a few minutes, no download needed
```

Install PyTorch with the CUDA build that matches your driver before `requirements-train.txt`.
Experiments in the paper ran on one NVIDIA H100.

## Data

The datasets are not redistributed. Download them into `data/raw/`:

| Dataset | Source | Preparation |
| --- | --- | --- |
| MetaQA (2-hop, 3-hop) | https://github.com/yuyuz/MetaQA | `bash scripts/download_metaqa.sh` |
| PathQuestion (PQ-3H) | https://github.com/zmtkeke/IRN | `python experiments/convert_pathquestion.py` |
| WebQSP (RoG subgraphs) | https://huggingface.co/datasets/rmanluo/RoG-webqsp | `python experiments/convert_webqsp.py` |

`experiments/p1_prepare_data.py` builds the federated splits: inverse triples, entity ownership, masked questions and relation-path labels.
The ownership used in the paper is in `partitions/` (`entities.tsv.gz`: entity id, name, owner), and seed 0 regenerates it.

## Reproducing the paper

Each script writes logs to `results/logs/` and outputs to `results/`.

| Paper result | Script |
| --- | --- |
| Table 2 (datasets and partitions) | `stats.json` written by the data step of the scripts below |
| Table 3, MetaQA 2-hop | `scripts/run_metaqa2_final.sh` |
| Table 3, MetaQA 3-hop and PQ-3H | `scripts/run_p3v2.sh` |
| Table 3, WebQSP | `scripts/run_p3_webqsp_v3.sh`, then `scripts/run_webqsp_masked.sh` |
| Table 3, baselines | `scripts/run_baselines_b1.sh`, `run_baselines_b2.sh`, `run_baselines_b3.sh` |
| Table 3, seeds 1 and 2 | `scripts/run_seeds.sh` |
| Significance tests | `scripts/run_significance.sh` |
| Table 4 (answer-set F1, All found) | `scripts/run_answer_f1.sh` |
| Fig. 3 (cross-client ratio, clients, skew) | `scripts/run_ablation_rho.sh`, `run_ablation_clients.sh`, `run_ablation_budget_skew.sh` |
| Table 5 (label-discovery budget) | `scripts/run_ablation_budget_skew.sh` |
| Table 6 (sketch relay and cost) | `scripts/run_p4_attacks.sh` |
| Table 7 (differential privacy) | `scripts/run_ablation_dp.sh` |
| Table 8 (privacy attacks) | `scripts/run_p4_attacks.sh` |

The simulator runs all clients in one process.
It replaces the OPRF with a keyed 64-bit hash and secure aggregation with plain sums, and it counts the communication of both protocols.

## Citation

```bibtex
@article{bappy2026fedhkgqa,
  title   = {{FedH-KGQA}: Privacy-Preserving Multi-Hop Question Answering over Horizontally Partitioned Knowledge Graphs},
  author  = {Islam Khan Bappy, Md Saikat and Seneviratne, Oshani},
  journal = {Journal of Intelligent Information Systems},
  note    = {Under review},
  year    = {2026}
}
```

