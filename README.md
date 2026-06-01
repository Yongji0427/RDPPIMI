# RDPPIMI

RDPPIMI is a research codebase for compound-protein-pair interaction modeling with three main components:

- PPIMI downstream model: graph-based compound encoder, protein-pair feature encoder, and bilinear attention fusion.
- RD feature construction: residue-weighted pooling, reverse-distillation scaler utilities, and recursive pair-feature builders.
- ProteinShake integration: batch prior extraction helpers and residue-score merge utilities for interface-aware analysis.

## Repository Layout

```text
src/rdppimi/ppimi/          PPIMI model, BAN layer, compound GNN, datasets, embedding specs
src/rdppimi/rd/             RD loaders and recursive pair-feature construction
src/rdppimi/proteinshake/   ProteinShake prior and residue-score merge utilities
scripts/                    Training, asset-building, validation, and analysis entry points
scripts/support/            Helper modules used by RD scaler scripts
docs/                       Notes for code organization and expected input schemas
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

## Expected Inputs

The scripts are path-configurable. By default, they expect project-local folders such as:

```text
data/features/
data/features_multiscale/
data/esm2_multiscale_embeddings/
data/esm2_residue_embeddings/
outputs/
external/
```

Use command-line arguments to point each entry point at the desired feature tables, fold definitions, prior indexes, scaler roots, and output directories.

## Minimal Smoke Checks

```bash
python -m py_compile $(find src scripts -name '*.py')
python scripts/train_ppimi.py --help
python scripts/build_rd_assets.py --help
python scripts/merge_proteinshake_residue_scores.py --help
```
