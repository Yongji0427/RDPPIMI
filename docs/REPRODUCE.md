# Reproduce The Release

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install compatible PyTorch, PyTorch Geometric, RDKit, and CUDA builds for your machine.

## 2. Download Release Assets

Download all files from release `v1.0-rdppimi-s4-3b`, then verify:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Unpack the supporting assets:

```bash
mkdir -p release_assets
tar -xzf rdppimi-s4-3b-weighted-pair-run-configs.tar.gz -C release_assets
tar -xzf rdppimi-s4-3b-weighted-pair-minimal-assets.tar.gz -C release_assets
```

## 3. Verify Weight Files

```bash
python scripts/verify_release_assets.py --asset-dir /path/to/downloaded/release/files
```

This checks the expected filenames, verifies `SHA256SUMS.txt`, and confirms that PyTorch can load each `.pt` file.

## 4. ProteinShake Prior Smoke Test

ProteinShake uses `facebook/esm2_t12_35M_UR50D` as its ESM2 backbone. The released `proteinshake-ppi-esm2-35m-final.pt` contains the fine-tuned sequence encoder and pair head, while the Hugging Face backbone files should be obtained through the standard model cache or a local mirror.

Expected ProteinShake outputs for a protein pair are:

```text
pair_logits.npy
pair_probs.npy
residue_scores_A.npy
residue_scores_B.npy
residue_weights_A.npy
residue_weights_B.npy
metadata.json
```

Residue weights are normalized to sum to 1 for each protein chain.

## 5. RDPPIMI Fold Loading

Instantiate the PPIMI model with:

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

Full metric reproduction requires the released fold-specific weighted-pair assets and the original MultiPPIMI fold definitions.
