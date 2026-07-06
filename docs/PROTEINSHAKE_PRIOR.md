# ProteinShake Prior

RDPPIMI uses ProteinShake to create an external residue-level PPI prior. This prior is not a ground-truth interface label; it is a model-derived proxy used to guide residue pooling.

## Inference Model

```text
Backbone: facebook/esm2_t12_35M_UR50D
Representation: sequence
Max length: 512
Fine-tuning: freeze ESM2 except the last 2 layers
Pair scorer: concat_mlp2
Score activation: sigmoid
Residue weight mode: row_col_max
```

The release file `proteinshake-ppi-esm2-35m-final.pt` stores the fine-tuned ProteinShake model as:

```text
cfg
state_dict
```

## Data Flow

For a pair of protein sequences A and B:

1. ProteinShake predicts a residue-pair logit matrix `S` with shape `[L_A, L_B]`.
2. Probabilities are computed as `P = sigmoid(S)`.
3. Chain-level residue scores are derived as:
   - `rA_i = max_j P_ij`
   - `rB_j = max_i P_ij`
4. Residue weights are normalized:
   - `wA_i = (rA_i + eps) / sum(rA + eps)`
   - `wB_j = (rB_j + eps) / sum(rB + eps)`
5. RDPPIMI uses these weights to construct ProteinShake-conditioned pooled protein-pair features.

## Released Supporting Assets

The minimal release asset archive includes:

```text
multippimi_ppi_prior_index.csv
ppi_prior_cache/
fixed_softmax_score_pooled_assets/t0p7/
fold-specific RD weighted-pair assets
```

These assets let users inspect the prior-derived residue weights and rebuild the weighted-pair feature inputs used by the released S4 3B classifier.
