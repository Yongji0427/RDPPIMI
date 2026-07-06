# RDPPIMI

RDPPIMI is a research codebase for compound-protein-pair interaction modeling with three main components:

- ProteinShake PPI prior extraction: residue-level PPI proxy scores and normalized residue weights.
- RD feature construction: ProteinShake-conditioned residue pooling, reverse-distillation scaler utilities, and recursive pair-feature builders.
- PPIMI downstream model: graph-based compound encoder, protein-pair feature encoder, and bilinear attention fusion.

## Best Released Result

The released RDPPIMI S4 3B weighted-pair model uses ProteinShake-derived PPI priors, recursive RD weighted-pair assets, and a no-projector PPIMI classifier.

| Setting | Protein Features | ROC-AUC | AUPR | F1 | MCC |
|---|---|---:|---:|---:|---:|
| S4 cold pair, 5-fold mean | 3B weighted-pair RD features | 0.802388 | 0.763953 | 0.795794 | 0.562862 |

Weights and reproducibility assets are distributed through the GitHub Release `v1.0-rdppimi-s4-3b`, not in git history.

## Repository Layout

```text
src/rdppimi/ppimi/          PPIMI model, BAN layer, compound GNN, datasets, embedding specs
src/rdppimi/rd/             RD loaders and recursive pair-feature construction
src/rdppimi/proteinshake/   ProteinShake prior and residue-score merge utilities
scripts/                    Training, asset-building, validation, and analysis entry points
scripts/support/            Helper modules used by RD scaler scripts
docs/                       Notes for code organization and expected input schemas
```

## Release Assets

The public release uses descriptive asset names:

```text
rdppimi-s4-3b-weighted-pair-fold1-best.pt
rdppimi-s4-3b-weighted-pair-fold2-best.pt
rdppimi-s4-3b-weighted-pair-fold3-best.pt
rdppimi-s4-3b-weighted-pair-fold4-best.pt
rdppimi-s4-3b-weighted-pair-fold5-best.pt
proteinshake-ppi-esm2-35m-final.pt
graphmvp-molecular-encoder-init.pt
rdppimi-s4-3b-weighted-pair-results.md
rdppimi-s4-3b-weighted-pair-results.json
rdppimi-s4-3b-weighted-pair-grid-registry.csv
rdppimi-s4-3b-weighted-pair-run-configs.tar.gz
rdppimi-s4-3b-weighted-pair-minimal-assets.tar.gz
SHA256SUMS.txt
```

See `docs/WEIGHTS.md`, `docs/RESULTS.md`, `docs/REPRODUCE.md`, and `docs/PROTEINSHAKE_PRIOR.md` for details.

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

## CD-HIT Pair Selection

RD scaler fitting uses side-specific CD-HIT clustering by default. Protein-A and protein-B sequences are clustered independently, and the fitting set retains the intersection of pairs whose A and B endpoints are representatives in their respective clusters:

```bash
python scripts/train_rd_scalers.py \
  --cdhit_pair_mode sidewise_intersection \
  --cdhit_threshold 0.5
```

The previous pooled-protein behavior remains available with `--cdhit_pair_mode pooled`.

## Minimal Smoke Checks

```bash
python -m py_compile $(find src scripts -name '*.py')
python scripts/train_ppimi.py --help
python scripts/build_rd_assets.py --help
python scripts/merge_proteinshake_residue_scores.py --help
```
