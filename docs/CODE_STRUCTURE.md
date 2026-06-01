# Code Structure

## PPIMI

The PPIMI module contains the downstream neural architecture:

- `MultiPPIMI.py`: compound-protein-pair interaction model and optional protein feature projection blocks.
- `ban.py`: bilinear attention network layer.
- `compound_gnn_model.py`: graph neural network compound encoder.
- `datasets/`: molecule graph conversion, fold loading, weighted-pair feature loading, and metric helpers.
- `protein_embedding_specs.py`: named protein embedding scale definitions and dimensional metadata.

## RD

The RD module contains the feature construction path:

- `fixed_weighted_residue_rd_loader.py`: residue-weighted transition matrix loading.
- `fixed_softmax_side_instance_loader.py`: fixed softmax pooled side-instance loading.
- `fixed_weighted_rd_pair_feature_builder.py`: adjacent full-step pair-feature construction.
- `recursive_chain_rd_pair_feature_builder.py`: recursive-chain pair-feature construction.

## ProteinShake

The ProteinShake module contains prior extraction and alignment utilities:

- `proteinshake_prior_batch.py`: batch wrapper around a ProteinShake-style pair inference script.
- `attention_overlap.py`: residue-score standardization, proxy merge, and overlap-table utilities.
