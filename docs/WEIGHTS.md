# Release Weights

This repository keeps large model files out of git history. Download the weights from the GitHub Release `v1.0-rdppimi-s4-3b`.

## Files

| File | Purpose | Approx. Size | Notes |
|---|---|---:|---|
| `rdppimi-s4-3b-weighted-pair-fold1-best.pt` | RDPPIMI fold 1 classifier | 44 MB | Raw PyTorch `state_dict`. |
| `rdppimi-s4-3b-weighted-pair-fold2-best.pt` | RDPPIMI fold 2 classifier | 44 MB | Raw PyTorch `state_dict`. |
| `rdppimi-s4-3b-weighted-pair-fold3-best.pt` | RDPPIMI fold 3 classifier | 44 MB | Raw PyTorch `state_dict`. |
| `rdppimi-s4-3b-weighted-pair-fold4-best.pt` | RDPPIMI fold 4 classifier | 44 MB | Raw PyTorch `state_dict`. |
| `rdppimi-s4-3b-weighted-pair-fold5-best.pt` | RDPPIMI fold 5 classifier | 44 MB | Raw PyTorch `state_dict`. |
| `proteinshake-ppi-esm2-35m-final.pt` | ProteinShake PPI residue-prior model | 138 MB | Contains the fine-tuned ESM2-35M sequence encoder and pair head. |
| `graphmvp-molecular-encoder-init.pt` | Molecular graph encoder initialization | 7.6 MB | Used by the PPIMI compound encoder. |

## External Backbone

Do not download a copied ESM2 backbone from this release. ProteinShake uses `facebook/esm2_t12_35M_UR50D`; obtain it from Hugging Face or provide a local cache path in the inference config.

## Checksum Verification

After downloading release files:

```bash
sha256sum -c SHA256SUMS.txt
```

On macOS:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Loading RDPPIMI Weights

The RDPPIMI fold weights are raw model `state_dict` files. Instantiate the matching PPIMI architecture with:

```text
protein_feature_source = weighted_pair_embedding
pooling_mode = mean
protein_embedding_model = 3B
protein_projector_mode = none
ppi_hidden_dim = 5158
```

Then load the selected fold weight with `torch.load(..., map_location="cpu")` and `model.load_state_dict(...)`.

## Loading ProteinShake Weights

The ProteinShake release weight is a dictionary with:

```text
cfg
state_dict
```

Use `cfg` to reconstruct the ESM2-35M sequence encoder and `concat_mlp2` pair scorer, then load `state_dict`.
