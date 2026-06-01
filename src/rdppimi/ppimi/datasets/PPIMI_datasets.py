import copy
import hashlib
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, \
        precision_score, recall_score, \
        f1_score, confusion_matrix, accuracy_score, matthews_corrcoef
from rdppimi.ppimi.protein_embedding_specs import PROTEIN_PHY_CSV_PATH, get_embedding_spec, load_canonical_protein_ids
from rdppimi.ppimi.datasets.molecule_datasets import mol_to_graph_data_obj_simple


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def get_best_threshold(output, labels):
    preds = output[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    best_threshold = thresholds[f1_scores.argmax()]
    return best_threshold


def performance_evaluation(output, labels):
    output = torch.softmax(torch.from_numpy(output), dim=1)
    pred_scores = output[:, 1]
    roc_auc = roc_auc_score(labels, pred_scores)
    prec, reca, _ = precision_recall_curve(labels, pred_scores)
    aupr = auc(reca, prec)

    best_threshold = get_best_threshold(output, labels)
    pred_labels = output[:, 1] > best_threshold
    precision = precision_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    (tn, fp, fn, tp) = confusion_matrix(labels, pred_labels).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, pred_labels)

    return roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _validate_unique_ids(frame, id_column, source_name):
    id_values = frame[id_column].astype(str).tolist()
    if len(id_values) != len(set(id_values)):
        duplicate_ids = pd.Series(id_values)[pd.Series(id_values).duplicated()].unique().tolist()
        raise ValueError(f"{source_name} contains duplicate uniprot_id values: {duplicate_ids[:10]}")
    return id_values


def _validate_numeric_frame(frame, source_name):
    numeric_frame = frame.apply(pd.to_numeric, errors='raise')
    if numeric_frame.isna().any().any():
        raise ValueError(f"{source_name} contains missing or non-numeric values")
    return numeric_frame


def _ordered_value_checksum(values):
    digest = hashlib.md5()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _load_original_pooled_esm_embeddings(spec):
    esm_csv_path = spec.csv_path
    if not os.path.exists(esm_csv_path):
        raise FileNotFoundError(f"ESM CSV not found: {esm_csv_path}")

    esm_df = pd.read_csv(esm_csv_path, header=None)
    if esm_df.shape[1] < 2:
        raise ValueError(
            f"ESM CSV must contain an id column plus {spec.esm_dim} embedding columns, got {esm_df.shape[1]}"
        )
    esm_ids = esm_df.iloc[:, 0].astype(str).tolist()
    if len(esm_ids) != len(set(esm_ids)):
        duplicate_ids = pd.Series(esm_ids)[pd.Series(esm_ids).duplicated()].unique().tolist()
        raise ValueError(f"ESM CSV contains duplicate uniprot_id values: {duplicate_ids[:10]}")

    esm_embedding = esm_df.iloc[:, 1:]
    if esm_embedding.shape[1] != spec.esm_dim:
        raise ValueError(
            f"ESM CSV expected {spec.esm_dim} embedding columns, got {esm_embedding.shape[1]}"
        )
    esm_embedding = _validate_numeric_frame(esm_embedding, "ESM CSV")
    esm_embedding.index = esm_ids
    esm_embedding.index.name = 'uniprot_id'
    return esm_embedding, esm_ids


def _select_residue_index_rows(spec, index_csv_path):
    index_csv_path = Path(index_csv_path)
    if not index_csv_path.exists():
        raise FileNotFoundError(f"Residue embedding index not found: {index_csv_path}")

    idx_df = pd.read_csv(index_csv_path)
    required_cols = {"uniprot_id", "status", "embedding_path"}
    missing_cols = required_cols - set(idx_df.columns)
    if missing_cols:
        raise ValueError(
            f"Residue embedding index missing required columns {sorted(missing_cols)}: {index_csv_path}"
        )

    status_mask = idx_df["status"].astype(str).isin(["success", "cached"])
    model_mask = pd.Series([True] * len(idx_df), index=idx_df.index)
    expected_model_name = spec.raw_embedding_dir.name
    if "model_key" in idx_df.columns:
        model_mask &= idx_df["model_key"].astype(str) == spec.key
    elif "model_name" in idx_df.columns:
        model_mask &= idx_df["model_name"].astype(str) == expected_model_name

    selected = idx_df[status_mask & model_mask].copy()
    if selected.empty:
        raise ValueError(
            f"No usable residue embeddings found for model={spec.key} in index: {index_csv_path}"
        )

    selected["uniprot_id"] = selected["uniprot_id"].map(_normalize_id)
    selected = selected[selected["uniprot_id"] != ""]
    selected = selected.drop_duplicates(subset=["uniprot_id"], keep="first")
    return selected


def _load_residue_level_embedding_paths(spec, index_csv_path):
    selected = _select_residue_index_rows(spec, index_csv_path)
    path_map = {}
    for row in selected.itertuples(index=False):
        uniprot_id = _normalize_id(row.uniprot_id)
        emb_path = str(row.embedding_path).strip()
        if not emb_path or emb_path.lower() == "nan":
            continue
        path_map[uniprot_id] = emb_path
    if not path_map:
        raise ValueError(
            f"No usable embedding_path entries found for model={spec.key} in index: {index_csv_path}"
        )
    return path_map


def _load_residue_level_mean_esm_embeddings(spec, index_csv_path, canonical_ids):
    residue_path_map = _load_residue_level_embedding_paths(spec, index_csv_path)

    residue_mean_map = {}
    for uniprot_id, emb_path in residue_path_map.items():
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Residue embedding file not found for {uniprot_id}: {emb_path}")
        residue_embedding = np.load(emb_path)
        if residue_embedding.ndim != 2:
            raise ValueError(f"Residue embedding for {uniprot_id} must be 2D, got ndim={residue_embedding.ndim}")
        if residue_embedding.shape[1] != spec.esm_dim:
            raise ValueError(
                f"Residue embedding for {uniprot_id} expected dim {spec.esm_dim}, got {residue_embedding.shape[1]}"
            )
        mean_embedding = residue_embedding.mean(axis=0)
        if not np.isfinite(mean_embedding).all():
            raise ValueError(f"Residue embedding mean contains non-finite values for {uniprot_id}")
        residue_mean_map[uniprot_id] = mean_embedding.astype(np.float32)

    rows = []
    missing_ids = []
    for uniprot_id in canonical_ids:
        if uniprot_id in residue_mean_map:
            rows.append(residue_mean_map[uniprot_id])
        else:
            rows.append(np.zeros(spec.esm_dim, dtype=np.float32))
            missing_ids.append(uniprot_id)

    if missing_ids:
        print(
            "[residue_level_esm2] warning: missing embeddings for "
            f"{len(missing_ids)} canonical ids; using zero vectors (first 10: {missing_ids[:10]})"
        )

    arr = np.stack(rows, axis=0)
    esm_embedding = pd.DataFrame(arr, index=canonical_ids)
    esm_embedding.index.name = "uniprot_id"
    return esm_embedding


def _load_pair_manifest_lookup(manifest_path, require_runnable=False):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    required_cols = {"pair_id", "proteinA_id", "proteinB_id"}
    missing_cols = required_cols - set(manifest_df.columns)
    if missing_cols:
        raise ValueError(
            f"Pair manifest missing required columns {sorted(missing_cols)}: {manifest_path}"
        )

    if require_runnable:
        if "is_runnable" not in manifest_df.columns:
            raise ValueError(
                f"Pair manifest requires is_runnable column when require_runnable=True: {manifest_path}"
            )
        manifest_df = manifest_df[manifest_df["is_runnable"].map(_to_bool)].copy()

    lookup = {}
    for row in manifest_df.itertuples(index=False):
        pair_id = _normalize_id(row.pair_id)
        protein_a = _normalize_id(row.proteinA_id)
        protein_b = _normalize_id(row.proteinB_id)
        if not pair_id or not protein_a or not protein_b:
            continue

        direct_key = (protein_a, protein_b)
        reverse_key = (protein_b, protein_a)

        if direct_key in lookup and lookup[direct_key][0] != pair_id:
            raise ValueError(f"Manifest key collision for pair {direct_key}: {lookup[direct_key][0]} vs {pair_id}")
        lookup[direct_key] = (pair_id, False)

        reverse_swapped = reverse_key != direct_key
        if reverse_key in lookup and lookup[reverse_key][0] != pair_id:
            raise ValueError(f"Manifest key collision for pair {reverse_key}: {lookup[reverse_key][0]} vs {pair_id}")
        lookup[reverse_key] = (pair_id, reverse_swapped)

    return lookup


def _load_prior_weight_map(prior_index_path, pooling_mode):
    prior_index_path = Path(prior_index_path)
    if not prior_index_path.exists():
        raise FileNotFoundError(f"PPI prior index not found: {prior_index_path}")

    prior_df = pd.read_csv(prior_index_path)
    required_cols = {"pair_id", "status"}
    if pooling_mode == "ppi_softmax_sum":
        required_cols.update({"residue_scores_A_path", "residue_scores_B_path"})
    elif pooling_mode == "ppi_weighted_sum":
        required_cols.update({"residue_weights_A_path", "residue_weights_B_path"})
    else:
        raise ValueError(f"Unsupported pooling_mode for prior path loading: {pooling_mode}")

    missing_cols = required_cols - set(prior_df.columns)
    if missing_cols:
        raise ValueError(
            f"PPI prior index missing required columns for pooling_mode={pooling_mode}: "
            f"{sorted(missing_cols)}: {prior_index_path}"
        )

    selected = prior_df[prior_df["status"].astype(str).isin(["success", "cached"])].copy()
    if selected.empty:
        raise ValueError(f"No successful/cached prior entries found in: {prior_index_path}")

    selected["pair_id"] = selected["pair_id"].map(_normalize_id)
    selected = selected[selected["pair_id"] != ""]
    selected = selected.drop_duplicates(subset=["pair_id"], keep="first")

    prior_map = {}
    for row in selected.itertuples(index=False):
        pair_id = _normalize_id(row.pair_id)
        prior_map[pair_id] = {
            "status": str(row.status),
            "residue_weights_A_path": str(getattr(row, "residue_weights_A_path", "")).strip(),
            "residue_weights_B_path": str(getattr(row, "residue_weights_B_path", "")).strip(),
            "residue_scores_A_path": str(getattr(row, "residue_scores_A_path", "")).strip(),
            "residue_scores_B_path": str(getattr(row, "residue_scores_B_path", "")).strip(),
        }
    return prior_map


def _load_weighted_pair_embedding_map(spec, weighted_index_path):
    weighted_index_path = Path(weighted_index_path)
    if not weighted_index_path.exists():
        raise FileNotFoundError(f"Weighted embedding index not found: {weighted_index_path}")

    index_df = pd.read_csv(weighted_index_path)
    required_cols = {
        "pair_id",
        "status",
        "weighted_embedding_A_path",
        "weighted_embedding_B_path",
        "proteinA_id",
        "proteinB_id",
    }
    missing_cols = required_cols - set(index_df.columns)
    if missing_cols:
        raise ValueError(
            f"Weighted embedding index missing required columns {sorted(missing_cols)}: {weighted_index_path}"
        )

    selected = index_df[index_df["status"].astype(str).isin(["success", "cached"])].copy()
    if "model_key" in selected.columns:
        selected = selected[selected["model_key"].astype(str) == spec.key].copy()
    if selected.empty:
        raise ValueError(
            f"No successful/cached weighted embedding rows found for model={spec.key}: {weighted_index_path}"
        )

    selected["pair_id"] = selected["pair_id"].map(_normalize_id)
    selected = selected[selected["pair_id"] != ""]
    selected = selected.drop_duplicates(subset=["pair_id"], keep="first")

    weighted_map = {}
    for row in selected.itertuples(index=False):
        pair_id = _normalize_id(row.pair_id)
        weighted_a_path = str(row.weighted_embedding_A_path).strip()
        weighted_b_path = str(row.weighted_embedding_B_path).strip()
        protein_a = _normalize_id(row.proteinA_id)
        protein_b = _normalize_id(row.proteinB_id)
        if (
            not weighted_a_path
            or not weighted_b_path
            or weighted_a_path.lower() == "nan"
            or weighted_b_path.lower() == "nan"
        ):
            continue
        weighted_map[pair_id] = {
            "status": str(row.status),
            "proteinA_id": protein_a,
            "proteinB_id": protein_b,
            "weighted_embedding_A_path": weighted_a_path,
            "weighted_embedding_B_path": weighted_b_path,
        }

    if not weighted_map:
        raise ValueError(
            f"No usable weighted_embedding_A/B_path rows found for model={spec.key}: {weighted_index_path}"
        )
    return weighted_map


class ModulatorPPIDataset(InMemoryDataset):
    def __init__(
        self,
        mode,
        setting,
        fold,
        protein_embedding_model='150M',
        protein_feature_source='original_pooled_esm2',
        residue_embedding_index_path='./multippimi_residue_embedding_index.csv',
        pooling_mode='mean',
        pair_manifest_path='./multippimi_pair_manifest.csv',
        ppi_prior_index_path='./multippimi_ppi_prior_index.csv',
        weighted_embedding_index_path='./multippimi_weighted_embedding_index.csv',
        train_pair_allowlist_csv=None,
        pooling_softmax_temperature=1.0,
        enable_softmax_pooling_export=False,
        missing_sequence_pair_fallback_ids=None,
    ):
        super(InMemoryDataset, self).__init__()
        self.mode = str(mode)
        self.setting = str(setting)
        self.fold = str(fold)
        self.protein_embedding_model = protein_embedding_model
        self.protein_feature_source = protein_feature_source
        self.residue_embedding_index_path = residue_embedding_index_path
        self.pooling_mode = pooling_mode
        self.pair_manifest_path = pair_manifest_path
        self.ppi_prior_index_path = ppi_prior_index_path
        self.weighted_embedding_index_path = weighted_embedding_index_path
        self.train_pair_allowlist_csv = str(train_pair_allowlist_csv or '').strip()
        self.pooling_softmax_temperature = float(pooling_softmax_temperature)
        self.enable_softmax_pooling_export = bool(enable_softmax_pooling_export)
        self.missing_sequence_pair_fallback_ids = {
            _normalize_id(value)
            for value in (missing_sequence_pair_fallback_ids or [])
            if _normalize_id(value)
        }
        self.missing_sequence_fallback_hits = []

        softmax_residue_sources = {"residue_level_esm2", "rd_residue_level_esm2"}
        if self.protein_feature_source not in {"original_pooled_esm2", "residue_level_esm2", "rd_residue_level_esm2", "weighted_pair_embedding"}:
            raise ValueError(
                f"Unsupported protein_feature_source={self.protein_feature_source}, "
                "expected one of: original_pooled_esm2, residue_level_esm2, rd_residue_level_esm2, weighted_pair_embedding"
            )
        if self.pooling_mode not in {"mean", "ppi_weighted_sum", "ppi_softmax_sum"}:
            raise ValueError(
                f"Unsupported pooling_mode={self.pooling_mode}, expected one of: mean, ppi_weighted_sum, ppi_softmax_sum"
            )
        if self.pooling_mode in {"ppi_weighted_sum", "ppi_softmax_sum"} and self.protein_feature_source not in softmax_residue_sources:
            raise ValueError(
                "pooling_mode in {ppi_weighted_sum, ppi_softmax_sum} requires "
                "protein_feature_source in {residue_level_esm2, rd_residue_level_esm2}"
            )
        if self.pooling_mode == "ppi_softmax_sum" and self.pooling_softmax_temperature <= 0:
            raise ValueError("pooling_softmax_temperature must be > 0 when pooling_mode=ppi_softmax_sum")
        if self.enable_softmax_pooling_export and self.pooling_mode != "ppi_softmax_sum":
            raise ValueError("enable_softmax_pooling_export requires pooling_mode=ppi_softmax_sum")
        if self.protein_feature_source == "rd_residue_level_esm2":
            if self.protein_embedding_model not in {"35M", "150M", "650M", "3B"}:
                raise ValueError(
                    "protein_feature_source=rd_residue_level_esm2 currently supports only "
                    "protein_embedding_model in {35M, 150M, 650M, 3B}"
                )
            if self.pooling_mode != "ppi_softmax_sum":
                raise ValueError(
                    "protein_feature_source=rd_residue_level_esm2 currently supports only pooling_mode=ppi_softmax_sum"
                )
        if self.protein_feature_source == "weighted_pair_embedding" and self.pooling_mode != "mean":
            raise ValueError(
                "protein_feature_source=weighted_pair_embedding requires pooling_mode=mean "
                "(compatibility placeholder only; no additional mean pooling is applied because the "
                "pair-level weighted/softmax pooling is already precomputed)."
            )

        datapath = f"./data/folds/{setting}/{mode}_fold{fold}.csv"
        print('datapath\t', datapath)

        self.raw_df = pd.read_csv(datapath).reset_index().rename(columns={'index': '_source_row_order'})
        self.raw_df = self.raw_df.sort_values('_source_row_order', kind='stable').reset_index(drop=True)
        self.active_df = self.raw_df.copy()

        self.use_pairwise_protein_features = False
        self.sample_pair_feature_matrix = None
        self.sample_pair_ids = []
        self.softmax_pooling_exports = []
        self.filtering_summary = {}
        self.dataset_debug_summary = {}
        self.train_pair_allowlist_pair_ids = None
        self.train_pair_allowlist_summary = self._build_default_train_pair_allowlist_summary()
        self._initialize_train_pair_allowlist_context()

        self.process_molecule()
        self.process_protein()

        df = self.active_df.sort_values('_source_row_order', kind='stable')
        df = df.drop(columns=['_source_row_order'], errors='ignore').reset_index(drop=True)
        self.active_df = df

        self.molecule_index_list = df['SMILES'].tolist()
        self.protein_index1_list = df['uniprot_id1'].tolist()
        self.protein_index2_list = df['uniprot_id2'].tolist()
        self.label_list = torch.LongTensor(df['label'].tolist())
        self._refresh_dataset_debug_summary()

        return

    @staticmethod
    def _label_summary(label_tensor):
        counts = torch.bincount(label_tensor.cpu(), minlength=2).tolist()
        negative = int(counts[0]) if len(counts) > 0 else 0
        positive = int(counts[1]) if len(counts) > 1 else 0
        total = negative + positive
        return {
            'negative': negative,
            'positive': positive,
            'total': total,
            'positive_fraction': float(positive / total) if total else 0.0,
        }

    def _build_default_train_pair_allowlist_summary(self):
        if self.mode != 'train':
            return {
                'original_train_rows': None,
                'train_rows_after_allowlist': None,
                'original_unique_pairs': None,
                'allowed_unique_pairs': None,
                'dropped_unique_pairs': None,
                'allowlist_csv_path': '',
            }
        return {
            'original_train_rows': int(len(self.raw_df)),
            'train_rows_after_allowlist': int(len(self.raw_df)),
            'original_unique_pairs': 0,
            'allowed_unique_pairs': 0,
            'dropped_unique_pairs': 0,
            'allowlist_csv_path': str(self.train_pair_allowlist_csv),
        }

    @staticmethod
    def _load_train_pair_allowlist_pair_ids(allowlist_csv_path):
        allowlist_csv_path = Path(allowlist_csv_path)
        if not allowlist_csv_path.exists():
            raise FileNotFoundError(f"train_pair_allowlist_csv not found: {allowlist_csv_path}")
        allowlist_df = pd.read_csv(allowlist_csv_path)
        if 'pair_id' not in allowlist_df.columns:
            raise ValueError(f"train_pair_allowlist_csv missing pair_id column: {allowlist_csv_path}")
        pair_ids = [_normalize_id(value) for value in allowlist_df['pair_id'].tolist()]
        return {pair_id for pair_id in pair_ids if pair_id}

    @staticmethod
    def _resolve_row_pair_id(row, pair_lookup):
        protein1_id = _normalize_id(row.get('uniprot_id1'))
        protein2_id = _normalize_id(row.get('uniprot_id2'))
        pair_meta = pair_lookup.get((protein1_id, protein2_id))
        if pair_meta is None:
            return ''
        return _normalize_id(pair_meta[0])

    def _initialize_train_pair_allowlist_context(self):
        self.train_pair_allowlist_summary = self._build_default_train_pair_allowlist_summary()
        self.train_pair_allowlist_pair_ids = None
        if self.mode != 'train':
            return

        pair_lookup = _load_pair_manifest_lookup(self.pair_manifest_path, require_runnable=True)
        mapped_pair_ids = [
            self._resolve_row_pair_id(row, pair_lookup)
            for _, row in self.raw_df.iterrows()
        ]
        original_unique_pairs = sorted({pair_id for pair_id in mapped_pair_ids if pair_id})
        allowed_pair_ids = set(original_unique_pairs)
        train_rows_after_allowlist = int(len(self.raw_df))

        if self.train_pair_allowlist_csv:
            allowed_pair_ids = self._load_train_pair_allowlist_pair_ids(self.train_pair_allowlist_csv)
            train_rows_after_allowlist = int(sum(1 for pair_id in mapped_pair_ids if pair_id in allowed_pair_ids))
            self.train_pair_allowlist_pair_ids = allowed_pair_ids

        allowed_unique_pairs = sorted({pair_id for pair_id in mapped_pair_ids if pair_id in allowed_pair_ids})
        self.train_pair_allowlist_summary = {
            'original_train_rows': int(len(self.raw_df)),
            'train_rows_after_allowlist': int(train_rows_after_allowlist),
            'original_unique_pairs': int(len(original_unique_pairs)),
            'allowed_unique_pairs': int(len(allowed_unique_pairs)),
            'dropped_unique_pairs': int(len(original_unique_pairs) - len(allowed_unique_pairs)),
            'allowlist_csv_path': str(self.train_pair_allowlist_csv),
        }

    def _is_train_pair_allowed(self, pair_id):
        if self.mode != 'train' or not self.train_pair_allowlist_csv:
            return True
        return _normalize_id(pair_id) in self.train_pair_allowlist_pair_ids

    def _record_filtering_summary(self, source_name, kept_indices, kept_pair_ids, drop_reasons):
        normalized_pair_ids = [str(_normalize_id(pid)) for pid in kept_pair_ids]
        self.filtering_summary = {
            'mode': self.mode,
            'setting': self.setting,
            'fold': self.fold,
            'source_name': str(source_name),
            'protein_feature_source': self.protein_feature_source,
            'pooling_mode': self.pooling_mode,
            'pooling_softmax_temperature': float(self.pooling_softmax_temperature),
            'residue_embedding_index_path': str(self.residue_embedding_index_path),
            'pair_manifest_path': str(self.pair_manifest_path),
            'ppi_prior_index_path': str(self.ppi_prior_index_path),
            'weighted_embedding_index_path': str(self.weighted_embedding_index_path),
            'raw_sample_count': int(len(self.raw_df)),
            'kept_sample_count': int(len(kept_indices)),
            'dropped_sample_count': int(len(self.raw_df) - len(kept_indices)),
            'kept_row_indices': [int(v) for v in kept_indices],
            'sample_pair_ids_order': normalized_pair_ids,
            'sample_pair_ids_order_checksum': _ordered_value_checksum(normalized_pair_ids),
            'drop_reasons': {str(k): int(v) for k, v in sorted(dict(drop_reasons).items())},
            'original_train_rows': self.train_pair_allowlist_summary.get('original_train_rows'),
            'train_rows_after_allowlist': self.train_pair_allowlist_summary.get('train_rows_after_allowlist'),
            'original_unique_pairs': self.train_pair_allowlist_summary.get('original_unique_pairs'),
            'allowed_unique_pairs': self.train_pair_allowlist_summary.get('allowed_unique_pairs'),
            'dropped_unique_pairs': self.train_pair_allowlist_summary.get('dropped_unique_pairs'),
            'allowlist_csv_path': self.train_pair_allowlist_summary.get('allowlist_csv_path'),
        }

    def _refresh_dataset_debug_summary(self):
        sample_pair_ids_order = [str(_normalize_id(pid)) for pid in self.sample_pair_ids]
        self.dataset_debug_summary = {
            'mode': self.mode,
            'setting': self.setting,
            'fold': self.fold,
            'protein_embedding_model': self.protein_embedding_model,
            'protein_feature_source': self.protein_feature_source,
            'pooling_mode': self.pooling_mode,
            'pooling_softmax_temperature': float(self.pooling_softmax_temperature),
            'active_sample_count': int(len(self.active_df)),
            'class_summary': self._label_summary(self.label_list),
            'sample_pair_ids_order': sample_pair_ids_order,
            'sample_pair_ids_order_checksum': _ordered_value_checksum(sample_pair_ids_order),
            'order_policy': 'active_df_file_order_after_filtering',
            'filtering_summary': copy.deepcopy(self.filtering_summary),
            'missing_sequence_pair_fallback_ids': sorted(self.missing_sequence_pair_fallback_ids),
            'missing_sequence_fallback_hit_count': int(len(self.missing_sequence_fallback_hits)),
            'missing_sequence_fallback_pairs': sorted({str(hit.get('pair_id')) for hit in self.missing_sequence_fallback_hits}),
        }

    def get_debug_summary(self):
        return copy.deepcopy(self.dataset_debug_summary)

    def process_molecule(self):
        rdkit_descriptors_path = './data/features/compound_phy.tsv'
        rdkit_descriptors = pd.read_csv(rdkit_descriptors_path, sep=' ')
        rdkit_descriptors = rdkit_descriptors.rename(columns={"smiles":"SMILES"})
        self.rdkit_descriptors = torch.FloatTensor(rdkit_descriptors.drop('SMILES', axis=1).to_numpy())

        smiles_list = rdkit_descriptors['SMILES']
        self.smiles_list = smiles_list.tolist()

        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
        preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
        assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
        assert len(smiles_list) == len(preprocessed_smiles_list)

        smiles_list, rdkit_mol_objs = preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list

        data_list = []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol != None:
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data_list.append(data)

        self.molecule_graph_list = data_list

        return

    @staticmethod
    def _load_cached_residue_embedding(uniprot_id, emb_path, spec, emb_cache, emb_path_cache):
        if uniprot_id in emb_cache:
            cached_path = emb_path_cache.get(uniprot_id)
            if cached_path is not None and cached_path != emb_path:
                raise ValueError(
                    f"Inconsistent embedding path for {uniprot_id}: {cached_path} vs {emb_path}"
                )
            return emb_cache[uniprot_id]

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Residue embedding file not found for {uniprot_id}: {emb_path}")

        emb = np.load(emb_path)
        if emb.ndim != 2:
            raise ValueError(f"Residue embedding for {uniprot_id} must be 2D, got ndim={emb.ndim}")
        if emb.shape[1] != spec.esm_dim:
            raise ValueError(
                f"Residue embedding for {uniprot_id} expected dim {spec.esm_dim}, got {emb.shape[1]}"
            )
        if not np.isfinite(emb).all():
            raise ValueError(f"Residue embedding contains non-finite values for {uniprot_id}")

        emb = emb.astype(np.float32, copy=False)
        emb_cache[uniprot_id] = emb
        emb_path_cache[uniprot_id] = emb_path
        return emb

    @staticmethod
    def _load_cached_weight(weight_path, weight_cache):
        if weight_path in weight_cache:
            return weight_cache[weight_path]

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Residue weight file not found: {weight_path}")

        weight = np.load(weight_path)
        if weight.ndim != 1:
            raise ValueError(f"Residue weight must be 1D, got ndim={weight.ndim}: {weight_path}")
        if not np.isfinite(weight).all():
            raise ValueError(f"Residue weight contains non-finite values: {weight_path}")
        if (weight < 0).any():
            raise ValueError(f"Residue weight contains negative values: {weight_path}")

        weight = weight.astype(np.float32, copy=False)
        weight_sum = float(weight.sum(dtype=np.float64))
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise ValueError(f"Residue weight sum must be 1, got {weight_sum:.8f}: {weight_path}")

        weight_cache[weight_path] = weight
        return weight

    @staticmethod
    def _load_cached_softmax_scores(score_path, score_cache):
        if score_path in score_cache:
            return score_cache[score_path]

        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Residue score file not found: {score_path}")

        scores = np.load(score_path)
        if scores.ndim != 1:
            raise ValueError(f"Residue score must be 1D, got ndim={scores.ndim}: {score_path}")
        if not np.isfinite(scores).all():
            raise ValueError(f"Residue score contains non-finite values: {score_path}")

        scores = scores.astype(np.float32, copy=False)
        score_cache[score_path] = scores
        return scores

    @staticmethod
    def _softmax_normalize_scores(scores, temperature):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 for softmax pooling, got {temperature}")
        logits = scores.astype(np.float64) / float(temperature)
        max_logit = float(np.max(logits))
        shifted = logits - max_logit
        exp_shifted = np.exp(shifted)
        denom = float(exp_shifted.sum(dtype=np.float64))
        if denom <= 0.0 or not np.isfinite(denom):
            raise RuntimeError(
                f"Invalid softmax denominator: denom={denom}, temperature={temperature}"
            )
        weights = exp_shifted / denom
        if not np.isfinite(weights).all():
            raise RuntimeError("Softmax produced non-finite residue weights")
        if (weights < -1e-7).any():
            raise RuntimeError("Softmax produced negative residue weights")
        weights = np.clip(weights, 0.0, None)
        clipped_sum = float(weights.sum(dtype=np.float64))
        if clipped_sum <= 0.0 or not np.isfinite(clipped_sum):
            raise RuntimeError(f"Invalid clipped softmax sum: {clipped_sum}")
        weights = weights / clipped_sum
        correction = 1.0 - float(weights.sum(dtype=np.float64))
        weights[-1] += correction
        weights = weights.astype(np.float32, copy=False)
        weight_sum = float(weights.sum(dtype=np.float64))
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise RuntimeError(
                f"Softmax weights sum must be 1 after normalization, got {weight_sum:.8f}"
            )
        return weights

    @staticmethod
    def _metadata_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _shape_list(array):
        return [int(v) for v in np.asarray(array).shape]

    @staticmethod
    def _ensure_finite_array(name, array, pair_id, protein_id=None):
        arr = np.asarray(array)
        if not np.isfinite(arr).all():
            scope = f"{name} for pair_id={pair_id}"
            if protein_id:
                scope += f", protein_id={protein_id}"
            raise RuntimeError(f"Non-finite values detected in {scope}")

    @staticmethod
    def _copy_valid_softmax_weights(weights, pair_id, role):
        weights = np.asarray(weights, dtype=np.float32).copy()
        if weights.ndim != 1:
            raise ValueError(f"Softmax export weights for {pair_id} {role} must be 1D, got {weights.ndim}D")
        if not np.isfinite(weights).all():
            raise ValueError(f"Softmax export weights for {pair_id} {role} contain non-finite values")
        if (weights < -1e-7).any():
            raise ValueError(f"Softmax export weights for {pair_id} {role} contain negative values")
        weight_sum = float(weights.sum(dtype=np.float64))
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise ValueError(f"Softmax export weights for {pair_id} {role} sum to {weight_sum:.8f}")
        return weights

    @staticmethod
    def _is_missing_sequence_sentinel_id(uniprot_id):
        return _normalize_id(uniprot_id).lower() == 'na'

    @staticmethod
    def _sentinel_zero_residue_embedding(spec):
        return np.zeros((1, spec.esm_dim), dtype=np.float32)

    @staticmethod
    def _uniform_scores(length):
        length = int(length)
        if length <= 0:
            raise ValueError(f'Uniform score length must be > 0, got {length}')
        return np.zeros((length,), dtype=np.float32)

    def _build_missing_sequence_pair_fallback(
        self,
        pair_id,
        protein1_id,
        protein2_id,
        spec,
        residue_path_map,
        emb_cache,
        emb_path_cache,
    ):
        pair_id = _normalize_id(pair_id)
        if pair_id not in self.missing_sequence_pair_fallback_ids:
            return None
        if not (
            self._is_missing_sequence_sentinel_id(protein1_id)
            or self._is_missing_sequence_sentinel_id(protein2_id)
        ):
            return None

        residue_embs = []
        residue_paths = []
        fallback_roles = []
        for protein_id in (protein1_id, protein2_id):
            if protein_id in residue_path_map:
                residue_path = residue_path_map[protein_id]
                residue_emb = self._load_cached_residue_embedding(
                    protein_id,
                    residue_path,
                    spec,
                    emb_cache,
                    emb_path_cache,
                )
                fallback_roles.append({'protein_id': protein_id, 'mode': 'existing_residue_embedding'})
            elif self._is_missing_sequence_sentinel_id(protein_id):
                residue_path = '__sentinel_zero_residue__'
                residue_emb = self._sentinel_zero_residue_embedding(spec)
                fallback_roles.append({'protein_id': protein_id, 'mode': 'sentinel_zero_residue_embedding'})
            else:
                return None
            residue_embs.append(residue_emb)
            residue_paths.append(residue_path)

        return {
            'pair_id': pair_id,
            'protein1_residue_path': residue_paths[0],
            'protein2_residue_path': residue_paths[1],
            'protein1_score_path': '__fallback_uniform_score__',
            'protein2_score_path': '__fallback_uniform_score__',
            'protein1_residue_embedding': residue_embs[0],
            'protein2_residue_embedding': residue_embs[1],
            'protein1_scores': self._uniform_scores(residue_embs[0].shape[0]),
            'protein2_scores': self._uniform_scores(residue_embs[1].shape[0]),
            'roles': fallback_roles,
        }

    def _build_softmax_pooling_export_record(
        self,
        sample_index,
        row_idx,
        row,
        pair_id,
        swapped,
        protein1_id,
        protein2_id,
        pooling_weight1,
        pooling_weight2,
        score_path1,
        score_path2,
        residue_path1,
        residue_path2,
        residue_emb1,
        residue_emb2,
        score1,
        score2,
        pooled1,
        pooled2,
        pair_feature,
    ):
        weights1 = self._copy_valid_softmax_weights(pooling_weight1, pair_id, "protein1")
        weights2 = self._copy_valid_softmax_weights(pooling_weight2, pair_id, "protein2")
        sample_metadata = {str(key): self._metadata_value(value) for key, value in row.items()}

        if swapped:
            protein_a_id = protein2_id
            protein_b_id = protein1_id
            weights_a = weights2
            weights_b = weights1
            score_path_a = score_path2
            score_path_b = score_path1
            residue_path_a = residue_path2
            residue_path_b = residue_path1
            residue_shape_a = self._shape_list(residue_emb2)
            residue_shape_b = self._shape_list(residue_emb1)
            score_shape_a = self._shape_list(score2)
            score_shape_b = self._shape_list(score1)
            pooled_shape_a = self._shape_list(pooled2)
            pooled_shape_b = self._shape_list(pooled1)
        else:
            protein_a_id = protein1_id
            protein_b_id = protein2_id
            weights_a = weights1
            weights_b = weights2
            score_path_a = score_path1
            score_path_b = score_path2
            residue_path_a = residue_path1
            residue_path_b = residue_path2
            residue_shape_a = self._shape_list(residue_emb1)
            residue_shape_b = self._shape_list(residue_emb2)
            score_shape_a = self._shape_list(score1)
            score_shape_b = self._shape_list(score2)
            pooled_shape_a = self._shape_list(pooled1)
            pooled_shape_b = self._shape_list(pooled2)

        return {
            "sample_index": int(sample_index),
            "raw_row_index": int(row_idx),
            "pair_id": pair_id,
            "swapped": bool(swapped),
            "sample_protein1_id": protein1_id,
            "sample_protein2_id": protein2_id,
            "proteinA_id": protein_a_id,
            "proteinB_id": protein_b_id,
            "proteinA_residue_count": int(weights_a.shape[0]),
            "proteinB_residue_count": int(weights_b.shape[0]),
            "pooling_temperature": float(self.pooling_softmax_temperature),
            "pooling_mode": self.pooling_mode,
            "proteinA_softmax_weights": weights_a,
            "proteinB_softmax_weights": weights_b,
            "proteinA_source_pooling_input_path": score_path_a,
            "proteinB_source_pooling_input_path": score_path_b,
            "proteinA_residue_embedding_path": residue_path_a,
            "proteinB_residue_embedding_path": residue_path_b,
            "proteinA_score_path": score_path_a,
            "proteinB_score_path": score_path_b,
            "proteinA_residue_embedding_shape": residue_shape_a,
            "proteinB_residue_embedding_shape": residue_shape_b,
            "proteinA_score_shape": score_shape_a,
            "proteinB_score_shape": score_shape_b,
            "proteinA_softmax_weight_shape": self._shape_list(weights_a),
            "proteinB_softmax_weight_shape": self._shape_list(weights_b),
            "proteinA_pooled_feature_shape": pooled_shape_a,
            "proteinB_pooled_feature_shape": pooled_shape_b,
            "pair_feature_shape": self._shape_list(pair_feature),
            "export_residue_index_base": 0,
            "exported_padding_count": 0,
            "sample_metadata": sample_metadata,
        }

    def get_softmax_pooling_export(self, idx, prediction_logits=None):
        if not self.enable_softmax_pooling_export:
            raise RuntimeError("Softmax pooling export is disabled for this dataset")
        if idx < 0 or idx >= len(self.softmax_pooling_exports):
            raise IndexError(f"softmax pooling export index out of range: {idx}")

        record = dict(self.softmax_pooling_exports[idx])
        record["proteinA_softmax_weights"] = record["proteinA_softmax_weights"].copy()
        record["proteinB_softmax_weights"] = record["proteinB_softmax_weights"].copy()
        if prediction_logits is not None:
            logits = np.asarray(prediction_logits, dtype=np.float64).reshape(-1)
            if not np.isfinite(logits).all():
                raise RuntimeError(
                    f"Prediction logits contain non-finite values for pair_id={record.get('pair_id')}"
                )
            record["prediction_logits"] = logits.astype(float).tolist()
            if logits.size == 1:
                record["prediction_score"] = float(logits[0])
            elif logits.size >= 2:
                shifted = logits - float(np.max(logits))
                exp_shifted = np.exp(shifted)
                probs = exp_shifted / float(exp_shifted.sum(dtype=np.float64))
                if not np.isfinite(probs).all():
                    raise RuntimeError(
                        f"Prediction probabilities contain non-finite values for pair_id={record.get('pair_id')}"
                    )
                record["prediction_logit_positive"] = float(logits[1])
                record["prediction_score"] = float(probs[1])
        return record

    def _build_weighted_pair_features(self, spec, phy_feature):
        residue_path_map = _load_residue_level_embedding_paths(spec, self.residue_embedding_index_path)
        pair_lookup = _load_pair_manifest_lookup(self.pair_manifest_path)
        prior_weight_map = _load_prior_weight_map(self.ppi_prior_index_path, self.pooling_mode)

        phy_map = {
            _normalize_id(uniprot_id): phy_feature.loc[uniprot_id].to_numpy(dtype=np.float32)
            for uniprot_id in phy_feature.index
        }

        emb_cache = {}
        emb_path_cache = {}
        weight_cache = {}

        paired_features = []
        kept_indices = []
        kept_pair_ids = []
        softmax_pooling_exports = []
        drop_reasons = Counter()

        for row_idx, row in self.raw_df.iterrows():
            protein1_id = _normalize_id(row.get('uniprot_id1'))
            protein2_id = _normalize_id(row.get('uniprot_id2'))

            pair_meta = pair_lookup.get((protein1_id, protein2_id))
            if pair_meta is None:
                drop_reasons['missing_manifest_pair'] += 1
                continue

            pair_id, swapped = pair_meta
            if not self._is_train_pair_allowed(pair_id):
                drop_reasons['train_pair_allowlist'] += 1
                continue
            prior_meta = prior_weight_map.get(pair_id)
            fallback_payload = None
            if self.pooling_mode == 'ppi_softmax_sum' and (
                prior_meta is None
                or protein1_id not in residue_path_map
                or protein2_id not in residue_path_map
            ):
                fallback_payload = self._build_missing_sequence_pair_fallback(
                    pair_id=pair_id,
                    protein1_id=protein1_id,
                    protein2_id=protein2_id,
                    spec=spec,
                    residue_path_map=residue_path_map,
                    emb_cache=emb_cache,
                    emb_path_cache=emb_path_cache,
                )

            if prior_meta is None and fallback_payload is None:
                drop_reasons['missing_or_failed_pair_prior'] += 1
                continue

            if protein1_id not in phy_map or protein2_id not in phy_map:
                drop_reasons['missing_phy_feature'] += 1
                continue

            score1 = None
            score2 = None
            if fallback_payload is None:
                if protein1_id not in residue_path_map or protein2_id not in residue_path_map:
                    drop_reasons['missing_residue_embedding'] += 1
                    continue

                if self.pooling_mode == "ppi_softmax_sum":
                    path_key_a = 'residue_scores_A_path'
                    path_key_b = 'residue_scores_B_path'
                    missing_path_reason = 'missing_score_path'
                else:
                    path_key_a = 'residue_weights_A_path'
                    path_key_b = 'residue_weights_B_path'
                    missing_path_reason = 'missing_weight_path'

                w1_path = prior_meta[path_key_b] if swapped else prior_meta[path_key_a]
                w2_path = prior_meta[path_key_a] if swapped else prior_meta[path_key_b]
                if not w1_path or not w2_path or w1_path.lower() == 'nan' or w2_path.lower() == 'nan':
                    drop_reasons[missing_path_reason] += 1
                    continue

                residue_path1 = residue_path_map[protein1_id]
                residue_path2 = residue_path_map[protein2_id]
                residue_emb1 = self._load_cached_residue_embedding(
                    protein1_id,
                    residue_path1,
                    spec,
                    emb_cache,
                    emb_path_cache,
                )
                residue_emb2 = self._load_cached_residue_embedding(
                    protein2_id,
                    residue_path2,
                    spec,
                    emb_cache,
                    emb_path_cache,
                )
                if self.pooling_mode == "ppi_softmax_sum":
                    score1 = self._load_cached_softmax_scores(w1_path, weight_cache)
                    score2 = self._load_cached_softmax_scores(w2_path, weight_cache)
                    if residue_emb1.shape[0] != len(score1):
                        raise ValueError(
                            f"Length mismatch for {protein1_id} in pair_id={pair_id}: "
                            f"embedding_len={residue_emb1.shape[0]} vs score_len={len(score1)}"
                        )
                    if residue_emb2.shape[0] != len(score2):
                        raise ValueError(
                            f"Length mismatch for {protein2_id} in pair_id={pair_id}: "
                            f"embedding_len={residue_emb2.shape[0]} vs score_len={len(score2)}"
                        )
                    pooling_weight1 = self._softmax_normalize_scores(score1, self.pooling_softmax_temperature)
                    pooling_weight2 = self._softmax_normalize_scores(score2, self.pooling_softmax_temperature)
                else:
                    pooling_weight1 = self._load_cached_weight(w1_path, weight_cache)
                    pooling_weight2 = self._load_cached_weight(w2_path, weight_cache)
                    if residue_emb1.shape[0] != len(pooling_weight1):
                        raise ValueError(
                            f"Length mismatch for {protein1_id} in pair_id={pair_id}: "
                            f"embedding_len={residue_emb1.shape[0]} vs weight_len={len(pooling_weight1)}"
                        )
                    if residue_emb2.shape[0] != len(pooling_weight2):
                        raise ValueError(
                            f"Length mismatch for {protein2_id} in pair_id={pair_id}: "
                            f"embedding_len={residue_emb2.shape[0]} vs weight_len={len(pooling_weight2)}"
                        )
            else:
                residue_path1 = fallback_payload['protein1_residue_path']
                residue_path2 = fallback_payload['protein2_residue_path']
                residue_emb1 = fallback_payload['protein1_residue_embedding']
                residue_emb2 = fallback_payload['protein2_residue_embedding']
                score1 = fallback_payload['protein1_scores']
                score2 = fallback_payload['protein2_scores']
                w1_path = fallback_payload['protein1_score_path']
                w2_path = fallback_payload['protein2_score_path']
                pooling_weight1 = self._softmax_normalize_scores(score1, self.pooling_softmax_temperature)
                pooling_weight2 = self._softmax_normalize_scores(score2, self.pooling_softmax_temperature)
                self.missing_sequence_fallback_hits.append({
                    'pair_id': pair_id,
                    'sample_row_index': int(row_idx),
                    'protein1_id': protein1_id,
                    'protein2_id': protein2_id,
                    'swapped': bool(swapped),
                    'roles': list(fallback_payload['roles']),
                })

            pooled1 = np.matmul(pooling_weight1, residue_emb1)
            pooled2 = np.matmul(pooling_weight2, residue_emb2)
            self._ensure_finite_array("pooled_feature", pooled1, pair_id, protein1_id)
            self._ensure_finite_array("pooled_feature", pooled2, pair_id, protein2_id)

            protein_feat1 = np.concatenate((pooled1, phy_map[protein1_id]), axis=0)
            protein_feat2 = np.concatenate((pooled2, phy_map[protein2_id]), axis=0)
            self._ensure_finite_array("protein_feature", protein_feat1, pair_id, protein1_id)
            self._ensure_finite_array("protein_feature", protein_feat2, pair_id, protein2_id)

            if protein_feat1.shape[0] != spec.per_protein_dim or protein_feat2.shape[0] != spec.per_protein_dim:
                raise ValueError(
                    f"Per-protein feature width mismatch for pair_id={pair_id}: "
                    f"expected {spec.per_protein_dim}, got {protein_feat1.shape[0]} and {protein_feat2.shape[0]}"
                )

            pair_feature = np.concatenate((protein_feat1, protein_feat2), axis=0).astype(np.float32, copy=False)
            if pair_feature.shape[0] != spec.paired_dim:
                raise ValueError(
                    f"Paired protein feature width mismatch for pair_id={pair_id}: "
                    f"expected {spec.paired_dim}, got {pair_feature.shape[0]}"
                )
            self._ensure_finite_array("pair_feature", pair_feature, pair_id)

            if self.pooling_mode == "ppi_softmax_sum" and self.enable_softmax_pooling_export:
                softmax_pooling_exports.append(
                    self._build_softmax_pooling_export_record(
                        sample_index=len(paired_features),
                        row_idx=row_idx,
                        row=row,
                        pair_id=pair_id,
                        swapped=swapped,
                        protein1_id=protein1_id,
                        protein2_id=protein2_id,
                        pooling_weight1=pooling_weight1,
                        pooling_weight2=pooling_weight2,
                        score_path1=w1_path,
                        score_path2=w2_path,
                        residue_path1=residue_path1,
                        residue_path2=residue_path2,
                        residue_emb1=residue_emb1,
                        residue_emb2=residue_emb2,
                        score1=score1,
                        score2=score2,
                        pooled1=pooled1,
                        pooled2=pooled2,
                        pair_feature=pair_feature,
                    )
                )

            paired_features.append(pair_feature)
            kept_indices.append(row_idx)
            kept_pair_ids.append(pair_id)

        if not paired_features:
            raise RuntimeError(
                f"No samples left after applying pooling_mode={self.pooling_mode} filtering "
                "(manifest + successful prior + residue embeddings)."
            )

        dropped_total = len(self.raw_df) - len(kept_indices)
        if dropped_total > 0:
            print(
                f"[{self.pooling_mode}] filtered samples without usable pair prior/residue embedding: "
                f"dropped={dropped_total}, kept={len(kept_indices)}, reasons={dict(drop_reasons)}"
            )

        if self.enable_softmax_pooling_export and len(softmax_pooling_exports) != len(paired_features):
            raise RuntimeError(
                f"Softmax pooling export count mismatch: exports={len(softmax_pooling_exports)} features={len(paired_features)}"
            )

        self.active_df = self.raw_df.iloc[kept_indices].reset_index(drop=True)
        stacked = np.stack(paired_features, axis=0)
        self.sample_pair_feature_matrix = torch.FloatTensor(stacked)
        self.sample_pair_ids = kept_pair_ids
        self.softmax_pooling_exports = softmax_pooling_exports
        self.use_pairwise_protein_features = True
        self._record_filtering_summary(
            source_name=f'{self.protein_feature_source}:{self.pooling_mode}',
            kept_indices=kept_indices,
            kept_pair_ids=kept_pair_ids,
            drop_reasons=drop_reasons,
        )

    @staticmethod
    def _load_cached_weighted_embedding(path, spec, emb_cache):
        if path in emb_cache:
            return emb_cache[path]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Weighted embedding file not found: {path}")

        emb = np.load(path)
        if emb.ndim != 1:
            raise ValueError(f"Weighted embedding must be 1D, got ndim={emb.ndim}: {path}")
        if emb.shape[0] != spec.esm_dim:
            raise ValueError(
                f"Weighted embedding dim mismatch: expected {spec.esm_dim}, got {emb.shape[0]}: {path}"
            )
        if not np.isfinite(emb).all():
            raise ValueError(f"Weighted embedding contains non-finite values: {path}")

        emb = emb.astype(np.float32, copy=False)
        emb_cache[path] = emb
        return emb

    def _build_cached_weighted_pair_features(self, spec, phy_feature):
        pair_lookup = _load_pair_manifest_lookup(self.pair_manifest_path, require_runnable=True)
        weighted_pair_map = _load_weighted_pair_embedding_map(spec, self.weighted_embedding_index_path)
        phy_map = {
            _normalize_id(uniprot_id): phy_feature.loc[uniprot_id].to_numpy(dtype=np.float32)
            for uniprot_id in phy_feature.index
        }

        emb_cache = {}
        paired_features = []
        kept_indices = []
        kept_pair_ids = []
        drop_reasons = Counter()

        for row_idx, row in self.raw_df.iterrows():
            protein1_id = _normalize_id(row.get('uniprot_id1'))
            protein2_id = _normalize_id(row.get('uniprot_id2'))

            pair_meta = pair_lookup.get((protein1_id, protein2_id))
            if pair_meta is None:
                drop_reasons['missing_or_non_runnable_pair'] += 1
                continue

            pair_id, swapped = pair_meta
            if not self._is_train_pair_allowed(pair_id):
                drop_reasons['train_pair_allowlist'] += 1
                continue
            cached_meta = weighted_pair_map.get(pair_id)
            if cached_meta is None:
                drop_reasons['missing_weighted_embedding_cache'] += 1
                continue

            if protein1_id not in phy_map or protein2_id not in phy_map:
                drop_reasons['missing_phy_feature'] += 1
                continue

            emb_a = self._load_cached_weighted_embedding(
                cached_meta["weighted_embedding_A_path"],
                spec,
                emb_cache,
            )
            emb_b = self._load_cached_weighted_embedding(
                cached_meta["weighted_embedding_B_path"],
                spec,
                emb_cache,
            )
            pooled1 = emb_b if swapped else emb_a
            pooled2 = emb_a if swapped else emb_b

            protein_feat1 = np.concatenate((pooled1, phy_map[protein1_id]), axis=0)
            protein_feat2 = np.concatenate((pooled2, phy_map[protein2_id]), axis=0)

            if protein_feat1.shape[0] != spec.per_protein_dim or protein_feat2.shape[0] != spec.per_protein_dim:
                raise ValueError(
                    f"Per-protein feature width mismatch for pair_id={pair_id}: "
                    f"expected {spec.per_protein_dim}, got {protein_feat1.shape[0]} and {protein_feat2.shape[0]}"
                )

            pair_feature = np.concatenate((protein_feat1, protein_feat2), axis=0).astype(np.float32, copy=False)
            if pair_feature.shape[0] != spec.paired_dim:
                raise ValueError(
                    f"Paired protein feature width mismatch for pair_id={pair_id}: "
                    f"expected {spec.paired_dim}, got {pair_feature.shape[0]}"
                )

            paired_features.append(pair_feature)
            kept_indices.append(row_idx)
            kept_pair_ids.append(pair_id)

        if not paired_features:
            raise RuntimeError(
                "No samples left after applying protein_feature_source=weighted_pair_embedding filtering "
                "(runnable pair + cached weighted pair embedding)."
            )

        dropped_total = len(self.raw_df) - len(kept_indices)
        if dropped_total > 0:
            print(
                "[weighted_pair_embedding] filtered samples without usable cached pair embedding: "
                f"dropped={dropped_total}, kept={len(kept_indices)}, reasons={dict(drop_reasons)}"
            )

        self.active_df = self.raw_df.iloc[kept_indices].reset_index(drop=True)
        stacked = np.stack(paired_features, axis=0)
        self.sample_pair_feature_matrix = torch.FloatTensor(stacked)
        self.sample_pair_ids = kept_pair_ids
        self.use_pairwise_protein_features = True
        self._record_filtering_summary(
            source_name=self.protein_feature_source,
            kept_indices=kept_indices,
            kept_pair_ids=kept_pair_ids,
            drop_reasons=drop_reasons,
        )

    def process_protein(self):
        spec = get_embedding_spec(self.protein_embedding_model)
        phy_csv_path = PROTEIN_PHY_CSV_PATH

        if not os.path.exists(phy_csv_path):
            raise FileNotFoundError(f"Protein physicochemical CSV not found: {phy_csv_path}")

        pfeature = pd.read_csv(phy_csv_path)
        if 'uniprot_id' not in pfeature.columns:
            raise ValueError("protein_phy.csv missing uniprot_id column")
        phy_ids = _validate_unique_ids(pfeature, 'uniprot_id', "protein_phy.csv")
        phy_feature = pfeature.drop(columns=['uniprot_id'])
        if phy_feature.shape[1] != spec.phy_dim:
            raise ValueError(
                f"protein_phy.csv expected {spec.phy_dim} feature columns, got {phy_feature.shape[1]}"
            )
        phy_feature = _validate_numeric_frame(phy_feature, "protein_phy.csv")
        phy_feature.index = phy_ids
        phy_feature.index.name = 'uniprot_id'

        canonical_ids = load_canonical_protein_ids()
        if canonical_ids != phy_ids:
            raise ValueError("protein_phy.csv canonical uniprot_id order is inconsistent")

        if self.protein_feature_source == "weighted_pair_embedding":
            self._build_cached_weighted_pair_features(spec, phy_feature)
            return

        if self.pooling_mode in {"ppi_weighted_sum", "ppi_softmax_sum"}:
            self._build_weighted_pair_features(spec, phy_feature)
            return

        if self.protein_feature_source == "original_pooled_esm2":
            esm_embedding, esm_ids = _load_original_pooled_esm_embeddings(spec)
            if set(esm_ids) != set(phy_ids):
                missing_in_esm = sorted(set(phy_ids) - set(esm_ids))
                missing_in_phy = sorted(set(esm_ids) - set(phy_ids))
                raise ValueError(
                    "ESM/phy uniprot_id sets do not match: "
                    f"missing_in_esm={missing_in_esm[:10]}, missing_in_phy={missing_in_phy[:10]}"
                )
            esm_embedding = esm_embedding.reindex(canonical_ids)
            if esm_embedding.index.tolist() != canonical_ids:
                raise ValueError("ESM CSV could not be reindexed to the canonical protein_phy order")
        elif self.protein_feature_source == "residue_level_esm2":
            esm_embedding = _load_residue_level_mean_esm_embeddings(
                spec=spec,
                index_csv_path=self.residue_embedding_index_path,
                canonical_ids=canonical_ids,
            )
        else:
            raise ValueError(f"Unsupported protein_feature_source={self.protein_feature_source}")

        combined_features = np.concatenate((esm_embedding.to_numpy(), phy_feature.to_numpy()), axis=1)
        if combined_features.shape[1] != spec.per_protein_dim:
            raise ValueError(
                f"Per-protein feature width expected {spec.per_protein_dim}, got {combined_features.shape[1]}"
            )
        if 2 * combined_features.shape[1] != spec.paired_dim:
            raise ValueError(
                f"Paired protein feature width expected {spec.paired_dim}, got {2 * combined_features.shape[1]}"
            )

        self.uniprot_id_list = canonical_ids
        self.protein_feats_list = torch.FloatTensor(combined_features)
        self.sample_pair_feature_matrix = None
        self.sample_pair_ids = []
        self.use_pairwise_protein_features = False
        self.active_df = self.raw_df.copy()
        self._record_filtering_summary(
            source_name=self.protein_feature_source,
            kept_indices=list(range(len(self.raw_df))),
            kept_pair_ids=[],
            drop_reasons=Counter(),
        )

    def __getitem__(self, idx):
        molecule_idx = self.smiles_list.index(self.molecule_index_list[idx])
        molecule_graph = self.molecule_graph_list[molecule_idx]
        rdkit_descriptors = self.rdkit_descriptors[molecule_idx]

        if self.use_pairwise_protein_features:
            protein_feats = self.sample_pair_feature_matrix[idx]
        else:
            protein_idx1 = self.uniprot_id_list.index(self.protein_index1_list[idx])
            protein_feats1 = self.protein_feats_list[protein_idx1]    # for target 1
            protein_idx2 = self.uniprot_id_list.index(self.protein_index2_list[idx])
            protein_feats2 = self.protein_feats_list[protein_idx2]    # for target 2
            protein_feats = torch.cat((protein_feats1, protein_feats2), 0)

        label = self.label_list[idx]
        return molecule_graph, rdkit_descriptors, protein_feats, label

    def __len__(self):
        return len(self.label_list)
