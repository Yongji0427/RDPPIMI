# Reproduce The Release

This guide covers the released downstream model and precomputed weighted-pair
features. RD scaler fitting and recursive feature-generation code are currently
excluded from the public snapshot. The released checkpoints and supporting
asset files remain available in `v1.0-rdppimi-s4-3b`.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Install PyTorch, PyTorch Geometric, RDKit, and CUDA builds matching your machine.

## 2. Download Release Assets

Download all files from release `v1.0-rdppimi-s4-3b`.

Required files:

```text
rdppimi-s4-3b-weighted-pair-fold1-best.pt
rdppimi-s4-3b-weighted-pair-fold2-best.pt
rdppimi-s4-3b-weighted-pair-fold3-best.pt
rdppimi-s4-3b-weighted-pair-fold4-best.pt
rdppimi-s4-3b-weighted-pair-fold5-best.pt
proteinshake-ppi-esm2-35m-final.pt
graphmvp-molecular-encoder-init.pt
rdppimi-s4-3b-weighted-pair-run-configs.tar.gz
rdppimi-s4-3b-weighted-pair-minimal-assets.tar.gz
SHA256SUMS.txt
```

Verify checksums:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Unpack the supporting assets:

```bash
mkdir -p release_assets
tar -xzf rdppimi-s4-3b-weighted-pair-run-configs.tar.gz -C release_assets
tar -xzf rdppimi-s4-3b-weighted-pair-minimal-assets.tar.gz -C release_assets
```

The run-config archive contains one shared `training_config.json`; `fold_overrides.csv` only records fold-specific split ids, weighted-asset paths, selected checkpoint files, best epochs, and fold metrics.

## 3. Model Files

- `proteinshake-ppi-esm2-35m-final.pt`: ProteinShake ESM2-35M PPI prior model.
- `graphmvp-molecular-encoder-init.pt`: molecular encoder initialization.
- `rdppimi-s4-3b-weighted-pair-fold*-best.pt`: one RDPPIMI checkpoint per cross-validation fold.

ProteinShake uses `facebook/esm2_t12_35M_UR50D`; download it from Hugging Face or provide a local model cache.

## 4. RDPPIMI Fold Loading

Use the shared configuration:

```text
protein_feature_source = weighted_pair_embedding
pooling_mode = mean
protein_embedding_model = 3B
protein_projector_mode = none
ppi_hidden_dim = 5158
```

Then load a fold checkpoint:

```python
import torch

state_dict = torch.load("rdppimi-s4-3b-weighted-pair-fold1-best.pt", map_location="cpu")
model.load_state_dict(state_dict)
```

Full metric reproduction requires the released fold-specific weighted-pair assets and MultiPPIMI fold definitions.
