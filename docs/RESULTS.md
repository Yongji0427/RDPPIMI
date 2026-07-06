# RDPPIMI Results

Best released setting: S4 cold pair, 3B weighted-pair RD features, 5-fold mean.

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.802388 | 0.081969 | 0.694609 | 0.908522 |
| AUPR | 0.763953 | 0.118697 | 0.588912 | 0.889411 |
| Accuracy | 0.766306 | 0.067862 | 0.702760 | 0.854244 |
| Precision | 0.711685 | 0.098897 | 0.610778 | 0.853448 |
| Recall | 0.914566 | 0.045886 | 0.840735 | 0.953271 |
| Specificity | 0.619294 | 0.153867 | 0.474843 | 0.802235 |
| F1 | 0.795794 | 0.053904 | 0.744526 | 0.882615 |
| MCC | 0.562862 | 0.096053 | 0.482378 | 0.693805 |
| Best epoch | 220.400000 | 51.340043 | 174 | 299 |

## Fold Metrics

| Fold | Best Epoch | ROC-AUC | AUPR | Accuracy | Precision | Recall | Specificity | F1 | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 243 | 0.694609 | 0.588912 | 0.705691 | 0.628889 | 0.952862 | 0.474843 | 0.757697 | 0.482378 |
| 2 | 299 | 0.847678 | 0.831987 | 0.854244 | 0.853448 | 0.913846 | 0.764977 | 0.882615 | 0.693805 |
| 3 | 174 | 0.758300 | 0.703843 | 0.702760 | 0.610778 | 0.953271 | 0.494163 | 0.744526 | 0.490534 |
| 4 | 183 | 0.908522 | 0.889411 | 0.818475 | 0.756198 | 0.840735 | 0.802235 | 0.796229 | 0.636289 |
| 5 | 203 | 0.802834 | 0.805611 | 0.750360 | 0.709110 | 0.912117 | 0.560250 | 0.797903 | 0.511304 |

## Training Configuration

```text
eval_setting = S4
protein_embedding_model = 3B
protein_feature_source = weighted_pair_embedding
pooling_mode = mean
protein_projector_mode = none
ppi_hidden_dim = 5158
checkpoint_selection_metric = roc_auc
epochs = 350
batch_size = 32
learning_rate = 2e-05
lr_min = 2e-06
weight_decay = 0.001
dropout_ratio = 0.35
grad_clip_norm = 0.5
seed = 42
runseed = 123
```
