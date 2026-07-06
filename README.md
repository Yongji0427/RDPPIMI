# RDPPIMI

RDPPIMI combines ProteinShake-derived residue priors, reverse-distillation weighted protein-pair features, and a PPIMI classifier for compound-protein-pair interaction prediction.

## Best Released Result

| Setting | Protein Features | ROC-AUC | AUPR | F1 | MCC |
|---|---|---:|---:|---:|---:|
| S4 cold pair, 5-fold mean | 3B weighted-pair RD features | 0.802388 | 0.763953 | 0.795794 | 0.562862 |

Weights and reproducibility assets are in GitHub Release `v1.0-rdppimi-s4-3b`.

## Repository Layout

```text
src/rdppimi/ppimi/          PPIMI model, BAN layer, compound GNN, datasets, embedding specs
src/rdppimi/rd/             RD loaders and recursive pair-feature construction
src/rdppimi/proteinshake/   ProteinShake prior and residue-score merge utilities
scripts/                    Training, asset-building, validation, and analysis entry points
scripts/support/            Helper modules used by RD scaler scripts
docs/                       Reproduction and result notes
```

## Core Entry Points

```text
scripts/train_ppimi.py                         Train/evaluate the PPIMI downstream model
scripts/build_rd_assets.py                     Build recursive-chain RD pair features
scripts/train_rd_scalers.py                    Train fold-specific RD scalers
scripts/train_rd_scalers_pcr_incremental.py    Train PCR/incremental RD scaler variant
scripts/export_softmax_pooling_weights.py      Export residue-pooling weights from model outputs
scripts/merge_proteinshake_residue_scores.py   Merge ProteinShake residue scores into residue tables
scripts/validate_weighted_pooling.py           Validate weighted PPI feature pooling inputs
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric and RDKit installation can depend on the CUDA, Python, and OS versions. Install the builds that match your environment.

For release reproduction, see `docs/REPRODUCE.md`. For the final metric table, see `docs/RESULTS.md`.
