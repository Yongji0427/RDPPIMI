from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_WEIGHT_ROOT = PROJECT_ROOT / 'fixed_residue_softmax_weights' / 't0p7'
CANONICAL_WEIGHT_INDEX_CSV = CANONICAL_WEIGHT_ROOT / 'fixed_residue_softmax_weight_index.csv'
PAIR_MANIFEST_CSV = PROJECT_ROOT / 'multippimi_pair_manifest.csv'
PPI_PRIOR_INDEX_CSV = PROJECT_ROOT / 'multippimi_ppi_prior_index.csv'
ALL_SCALES: Tuple[str, ...] = ('8M', '35M', '150M', '650M', '3B')
VALID_STATUSES = {'success', 'cached'}


@dataclass(frozen=True)
class ResidueRowKey:
    pair_index: int
    pair_id: str
    side: str
    protein_id: str
    residue_index: int

    def as_tuple(self) -> Tuple[int, str, str, str, int]:
        return (
            self.pair_index,
            self.pair_id,
            self.side,
            self.protein_id,
            self.residue_index,
        )

    def as_dict(self, row_index: int) -> Dict[str, object]:
        return {
            'row_index': row_index,
            'pair_index': self.pair_index,
            'pair_id': self.pair_id,
            'side': self.side,
            'protein_id': self.protein_id,
            'residue_index': self.residue_index,
        }


@dataclass(frozen=True)
class FixedWeightedResidueTransitionBundle:
    scale_in: str
    scale_out: str
    weight_root: Path
    weight_index_csv: Path
    row_keys: List[ResidueRowKey]
    xin: np.ndarray
    xout: np.ndarray
    summary: Dict[str, object]

    def row_keys_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([row.as_dict(idx) for idx, row in enumerate(self.row_keys)])

    def row_key_format(self) -> str:
        return 'row_index,pair_index,pair_id,side,protein_id,residue_index'


@dataclass(frozen=True)
class ProteinEmbeddingRecord:
    protein_id: str
    seq_len: int
    embedding_dim: int
    embedding_path: Path


REQUIRED_WEIGHT_COLUMNS = {
    'pair_id',
    'protein_id_A',
    'protein_id_B',
    'path_weights_A',
    'path_weights_B',
    'length_A',
    'length_B',
    'status',
}
REQUIRED_MANIFEST_COLUMNS = {'pair_id', 'proteinA_id', 'proteinB_id', 'is_runnable'}
REQUIRED_PRIOR_COLUMNS = {'pair_id', 'proteinA_id', 'proteinB_id', 'lenA', 'lenB', 'status'}
REQUIRED_EMBEDDING_COLUMNS = {'protein_id', 'seq_len', 'embedding_dim', 'embedding_path', 'status'}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip()


def _is_valid_status(value: object) -> bool:
    return _normalize_text(value) in VALID_STATUSES


def _is_runnable(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalize_text(value).lower()
    return text in {'1', 'true', 't', 'yes', 'y'}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing required CSV: {path}')
    return pd.read_csv(path)


def _read_weight_index(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    missing = REQUIRED_WEIGHT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Weight index missing required columns {sorted(missing)}: {path}')
    df = df.copy()
    df['pair_id'] = df['pair_id'].map(_normalize_text)
    df['protein_id_A'] = df['protein_id_A'].map(_normalize_text)
    df['protein_id_B'] = df['protein_id_B'].map(_normalize_text)
    df = df[df['status'].map(_is_valid_status)].reset_index(drop=True)
    if df['pair_id'].duplicated().any():
        dup_pairs = sorted(df.loc[df['pair_id'].duplicated(), 'pair_id'].unique().tolist())
        raise ValueError(f'Duplicate pair_id rows in weight index: {dup_pairs[:10]}')
    return df


def _read_manifest(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Pair manifest missing required columns {sorted(missing)}: {path}')
    df = df.copy()
    df['pair_id'] = df['pair_id'].map(_normalize_text)
    df['proteinA_id'] = df['proteinA_id'].map(_normalize_text)
    df['proteinB_id'] = df['proteinB_id'].map(_normalize_text)
    df = df[df['is_runnable'].map(_is_runnable)].drop_duplicates(subset=['pair_id'], keep='first').reset_index(drop=True)
    return df


def _read_prior_index(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    missing = REQUIRED_PRIOR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Prior index missing required columns {sorted(missing)}: {path}')
    df = df.copy()
    df['pair_id'] = df['pair_id'].map(_normalize_text)
    df['proteinA_id'] = df['proteinA_id'].map(_normalize_text)
    df['proteinB_id'] = df['proteinB_id'].map(_normalize_text)
    df = df[df['status'].map(_is_valid_status)].drop_duplicates(subset=['pair_id'], keep='first').reset_index(drop=True)
    return df


def _embedding_index_path(scale: str) -> Path:
    if scale not in ALL_SCALES:
        raise ValueError(f'Unsupported scale {scale!r}; expected one of {ALL_SCALES}')
    return PROJECT_ROOT / f'multippimi_residue_embedding_index_{scale}.csv'


def _read_embedding_lookup(scale: str) -> Dict[str, ProteinEmbeddingRecord]:
    path = _embedding_index_path(scale)
    df = _read_csv(path)
    missing = REQUIRED_EMBEDDING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Embedding index missing required columns {sorted(missing)}: {path}')
    df = df.copy()
    df['protein_id'] = df['protein_id'].map(_normalize_text)
    df = df[df['status'].map(_is_valid_status)].reset_index(drop=True)
    if df['protein_id'].duplicated().any():
        dup = sorted(df.loc[df['protein_id'].duplicated(), 'protein_id'].unique().tolist())
        raise ValueError(f'Duplicate protein_id rows in embedding index {path}: {dup[:10]}')
    expected_dim = get_embedding_spec(scale).esm_dim
    lookup: Dict[str, ProteinEmbeddingRecord] = {}
    for row in df.itertuples(index=False):
        embedding_dim = int(row.embedding_dim)
        if embedding_dim != expected_dim:
            raise ValueError(
                f'Unexpected embedding_dim for scale={scale}, protein={row.protein_id}: '
                f'expected {expected_dim}, got {embedding_dim}'
            )
        lookup[row.protein_id] = ProteinEmbeddingRecord(
            protein_id=row.protein_id,
            seq_len=int(row.seq_len),
            embedding_dim=embedding_dim,
            embedding_path=Path(_normalize_text(row.embedding_path)),
        )
    return lookup


def _load_weight_array(path: Path, expected_length: int, side_label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Missing fixed residue weight file for {side_label}: {path}')
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f'Weight array must be 1D for {side_label}, got shape={arr.shape}: {path}')
    if arr.shape[0] != expected_length:
        raise ValueError(
            f'Weight length mismatch for {side_label}: expected {expected_length}, got {arr.shape[0]}: {path}'
        )
    if not np.isfinite(arr).all():
        raise ValueError(f'Weight array contains NaN/inf for {side_label}: {path}')
    if (arr < -1e-7).any():
        raise ValueError(f'Weight array contains negative values for {side_label}: {path}')
    weight_sum = float(arr.sum(dtype=np.float64))
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(f'Weight sum must be ~1 for {side_label}, got {weight_sum}: {path}')
    return arr.astype(np.float32, copy=False)


def _load_embedding_matrix(path: Path, expected_length: int, expected_dim: int, scale: str, side_label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Missing residue embedding for {scale} {side_label}: {path}')
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f'Embedding must be 2D for {scale} {side_label}, got shape={arr.shape}: {path}')
    if arr.shape != (expected_length, expected_dim):
        raise ValueError(
            f'Embedding shape mismatch for {scale} {side_label}: '
            f'expected {(expected_length, expected_dim)}, got {arr.shape}: {path}'
        )
    if not np.isfinite(arr).all():
        raise ValueError(f'Embedding contains NaN/inf for {scale} {side_label}: {path}')
    return arr.astype(np.float32, copy=False)


def _read_pair_allowlist(path: Path) -> List[str]:
    df = _read_csv(path)
    if 'pair_id' not in df.columns:
        raise ValueError(f'Pair allowlist missing required pair_id column: {path}')
    pair_ids = [_normalize_text(value) for value in df['pair_id'].tolist()]
    pair_ids = [pair_id for pair_id in pair_ids if pair_id]
    if not pair_ids:
        raise ValueError(f'Pair allowlist has no usable pair_id rows: {path}')
    if len(set(pair_ids)) != len(pair_ids):
        duplicated = sorted({pair_id for pair_id in pair_ids if pair_ids.count(pair_id) > 1})
        raise ValueError(f'Pair allowlist contains duplicate pair_id values: {duplicated[:10]}')
    return pair_ids


def _align_selected_pairs(
    weight_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    limit_pairs: int | None,
    pair_allowlist_csv: Path | None = None,
) -> pd.DataFrame:
    manifest_map = manifest_df.set_index('pair_id')[['proteinA_id', 'proteinB_id']]
    prior_map = prior_df.set_index('pair_id')[['proteinA_id', 'proteinB_id', 'lenA', 'lenB']]

    manifest_pairs = set(manifest_map.index.tolist())
    prior_pairs = set(prior_map.index.tolist())
    weight_pairs = set(weight_df['pair_id'].tolist())

    if manifest_pairs != weight_pairs:
        missing_in_weights = sorted(manifest_pairs - weight_pairs)
        missing_in_manifest = sorted(weight_pairs - manifest_pairs)
        raise ValueError(
            'Weight index pair set mismatch against runnable manifest: '
            f'missing_in_weights={missing_in_weights[:10]}, missing_in_manifest={missing_in_manifest[:10]}'
        )
    if prior_pairs != weight_pairs:
        missing_in_weights = sorted(prior_pairs - weight_pairs)
        missing_in_prior = sorted(weight_pairs - prior_pairs)
        raise ValueError(
            'Weight index pair set mismatch against prior index: '
            f'missing_in_weights={missing_in_weights[:10]}, missing_in_prior={missing_in_prior[:10]}'
        )

    selected = weight_df.copy()
    if pair_allowlist_csv is not None:
        allowlist_path = Path(pair_allowlist_csv)
        allowed_pair_ids = _read_pair_allowlist(allowlist_path)
        allowed_set = set(allowed_pair_ids)
        missing_allowed = sorted(allowed_set - weight_pairs)
        if missing_allowed:
            raise ValueError(
                f'Pair allowlist contains pair_id values absent from runnable fixed-weight assets: '
                f'{missing_allowed[:10]} path={allowlist_path}'
            )
        selected = selected[selected['pair_id'].isin(allowed_set)].copy()
        selected['_allowlist_order'] = selected['pair_id'].map({pair_id: idx for idx, pair_id in enumerate(allowed_pair_ids)})
        selected = selected.sort_values('_allowlist_order', kind='stable').drop(columns=['_allowlist_order']).reset_index(drop=True)
        if selected.empty:
            raise ValueError(f'Pair allowlist selected zero runnable pairs: {allowlist_path}')

    if limit_pairs is not None:
        if limit_pairs < 1:
            raise ValueError(f'limit_pairs must be >= 1, got {limit_pairs}')
        selected = selected.iloc[:limit_pairs].copy()

    for row in selected.itertuples(index=False):
        manifest_row = manifest_map.loc[row.pair_id]
        prior_row = prior_map.loc[row.pair_id]
        if row.protein_id_A != manifest_row.proteinA_id or row.protein_id_B != manifest_row.proteinB_id:
            raise ValueError(
                f'Pair/protein mismatch against manifest for {row.pair_id}: '
                f'weights=({row.protein_id_A},{row.protein_id_B}) '
                f'manifest=({manifest_row.proteinA_id},{manifest_row.proteinB_id})'
            )
        if row.protein_id_A != prior_row.proteinA_id or row.protein_id_B != prior_row.proteinB_id:
            raise ValueError(
                f'Pair/protein mismatch against prior index for {row.pair_id}: '
                f'weights=({row.protein_id_A},{row.protein_id_B}) '
                f'prior=({prior_row.proteinA_id},{prior_row.proteinB_id})'
            )
        if int(row.length_A) != int(prior_row.lenA) or int(row.length_B) != int(prior_row.lenB):
            raise ValueError(
                f'Length mismatch against prior index for {row.pair_id}: '
                f'weights=({int(row.length_A)},{int(row.length_B)}) '
                f'prior=({int(prior_row.lenA)},{int(prior_row.lenB)})'
            )

    return selected.reset_index(drop=True)


def _validate_scale_length_consistency(
    selected_df: pd.DataFrame,
    embedding_lookups: Dict[str, Dict[str, ProteinEmbeddingRecord]],
) -> None:
    proteins: List[Tuple[str, int]] = []
    for row in selected_df.itertuples(index=False):
        proteins.append((row.protein_id_A, int(row.length_A)))
        proteins.append((row.protein_id_B, int(row.length_B)))

    seen: set[Tuple[str, int]] = set()
    for protein_id, expected_length in proteins:
        key = (protein_id, expected_length)
        if key in seen:
            continue
        seen.add(key)
        for scale in ALL_SCALES:
            lookup = embedding_lookups[scale]
            if protein_id not in lookup:
                raise KeyError(f'Missing embedding asset for protein {protein_id} in scale {scale}')
            record = lookup[protein_id]
            if record.seq_len != expected_length:
                raise ValueError(
                    f'Seq length mismatch for protein {protein_id} in scale {scale}: '
                    f'expected {expected_length}, got {record.seq_len}'
                )


def _build_weight_summary(selected_df: pd.DataFrame) -> Dict[str, float]:
    mins: List[float] = []
    maxs: List[float] = []
    means: List[float] = []
    sum_a_total = 0.0
    sum_b_total = 0.0
    for row in selected_df.itertuples(index=False):
        weight_a = _load_weight_array(Path(row.path_weights_A), int(row.length_A), f'{row.pair_id}:A')
        weight_b = _load_weight_array(Path(row.path_weights_B), int(row.length_B), f'{row.pair_id}:B')
        mins.extend([float(weight_a.min()), float(weight_b.min())])
        maxs.extend([float(weight_a.max()), float(weight_b.max())])
        means.extend([float(weight_a.mean()), float(weight_b.mean())])
        sum_a_total += float(weight_a.sum(dtype=np.float64))
        sum_b_total += float(weight_b.sum(dtype=np.float64))
    return {
        'weight_min': min(mins),
        'weight_max': max(maxs),
        'weight_mean': float(np.mean(means, dtype=np.float64)),
        'weight_sum_A_total': sum_a_total,
        'weight_sum_B_total': sum_b_total,
    }


def _iter_residue_keys(pair_index: int, pair_id: str, side: str, protein_id: str, length: int) -> Iterable[ResidueRowKey]:
    for residue_index in range(length):
        yield ResidueRowKey(
            pair_index=pair_index,
            pair_id=pair_id,
            side=side,
            protein_id=protein_id,
            residue_index=residue_index,
        )


def load_fixed_weighted_residue_transition(
    scale_in: str,
    scale_out: str,
    weight_index_csv: Path = CANONICAL_WEIGHT_INDEX_CSV,
    limit_pairs: int | None = None,
    verbose: bool = False,
    pair_allowlist_csv: Path | None = None,
) -> FixedWeightedResidueTransitionBundle:
    if scale_in not in ALL_SCALES or scale_out not in ALL_SCALES:
        raise ValueError(f'Unsupported transition {scale_in}->{scale_out}; expected scales from {ALL_SCALES}')
    if scale_in == scale_out:
        raise ValueError(f'scale_in and scale_out must differ, got {scale_in}')

    weight_index_csv = Path(weight_index_csv).resolve()
    weight_root = weight_index_csv.parent

    weight_df = _read_weight_index(weight_index_csv)
    manifest_df = _read_manifest(PAIR_MANIFEST_CSV)
    prior_df = _read_prior_index(PPI_PRIOR_INDEX_CSV)
    selected_df = _align_selected_pairs(
        weight_df=weight_df,
        manifest_df=manifest_df,
        prior_df=prior_df,
        limit_pairs=limit_pairs,
        pair_allowlist_csv=pair_allowlist_csv,
    )

    embedding_lookups = {scale: _read_embedding_lookup(scale) for scale in ALL_SCALES}
    _validate_scale_length_consistency(selected_df=selected_df, embedding_lookups=embedding_lookups)

    dim_in = get_embedding_spec(scale_in).esm_dim
    dim_out = get_embedding_spec(scale_out).esm_dim
    total_residue_count = int((selected_df['length_A'] + selected_df['length_B']).sum())
    pair_count = len(selected_df)
    side_count = pair_count * 2

    xin = np.empty((total_residue_count, dim_in), dtype=np.float32)
    xout = np.empty((total_residue_count, dim_out), dtype=np.float32)
    row_keys: List[ResidueRowKey] = []

    row_offset = 0
    for pair_index, row in enumerate(selected_df.itertuples(index=False)):
        pair_id = row.pair_id
        len_a = int(row.length_A)
        len_b = int(row.length_B)
        protein_a = row.protein_id_A
        protein_b = row.protein_id_B

        weight_a = _load_weight_array(Path(row.path_weights_A), len_a, f'{pair_id}:A')
        weight_b = _load_weight_array(Path(row.path_weights_B), len_b, f'{pair_id}:B')

        rec_in_a = embedding_lookups[scale_in][protein_a]
        rec_out_a = embedding_lookups[scale_out][protein_a]
        rec_in_b = embedding_lookups[scale_in][protein_b]
        rec_out_b = embedding_lookups[scale_out][protein_b]

        emb_in_a = _load_embedding_matrix(rec_in_a.embedding_path, len_a, dim_in, scale_in, f'{pair_id}:A')
        emb_out_a = _load_embedding_matrix(rec_out_a.embedding_path, len_a, dim_out, scale_out, f'{pair_id}:A')
        emb_in_b = _load_embedding_matrix(rec_in_b.embedding_path, len_b, dim_in, scale_in, f'{pair_id}:B')
        emb_out_b = _load_embedding_matrix(rec_out_b.embedding_path, len_b, dim_out, scale_out, f'{pair_id}:B')

        xin_a = emb_in_a * weight_a[:, None]
        xout_a = emb_out_a * weight_a[:, None]
        xin_b = emb_in_b * weight_b[:, None]
        xout_b = emb_out_b * weight_b[:, None]

        xin[row_offset:row_offset + len_a] = xin_a
        xout[row_offset:row_offset + len_a] = xout_a
        row_keys.extend(_iter_residue_keys(pair_index, pair_id, 'A', protein_a, len_a))
        row_offset += len_a

        xin[row_offset:row_offset + len_b] = xin_b
        xout[row_offset:row_offset + len_b] = xout_b
        row_keys.extend(_iter_residue_keys(pair_index, pair_id, 'B', protein_b, len_b))
        row_offset += len_b

        if verbose and ((pair_index + 1) % 10 == 0 or pair_index + 1 == pair_count):
            print(
                f'[loader] built {pair_index + 1}/{pair_count} pairs '
                f'for {scale_in}->{scale_out}; residue_rows={row_offset}'
            )

    if row_offset != total_residue_count:
        raise ValueError(f'Row assembly mismatch: expected {total_residue_count}, wrote {row_offset}')
    if len(row_keys) != total_residue_count:
        raise ValueError(f'Row key count mismatch: expected {total_residue_count}, got {len(row_keys)}')
    if len({(key.pair_id, key.side, key.residue_index) for key in row_keys}) != len(row_keys):
        raise ValueError('Row keys are not unique by (pair_id, side, residue_index)')
    if xin.ndim != 2 or xout.ndim != 2:
        raise ValueError(f'xin/xout must be 2D, got {xin.ndim} and {xout.ndim}')
    if xin.shape != (total_residue_count, dim_in):
        raise ValueError(f'Unexpected xin shape: expected {(total_residue_count, dim_in)}, got {xin.shape}')
    if xout.shape != (total_residue_count, dim_out):
        raise ValueError(f'Unexpected xout shape: expected {(total_residue_count, dim_out)}, got {xout.shape}')
    if not np.isfinite(xin).all():
        raise ValueError('xin contains NaN/inf')
    if not np.isfinite(xout).all():
        raise ValueError('xout contains NaN/inf')

    weight_summary = _build_weight_summary(selected_df)
    summary: Dict[str, object] = {
        'semantic_mode': 'fixed_weighted_residue_level_rd',
        'weight_root': str(weight_root),
        'weight_index_csv': str(weight_index_csv),
        'pair_allowlist_csv': str(pair_allowlist_csv or ''),
        'pair_count': pair_count,
        'side_count': side_count,
        'total_residue_count': total_residue_count,
        'scale_in': scale_in,
        'scale_out': scale_out,
        'input_dim': dim_in,
        'output_dim': dim_out,
        **weight_summary,
    }

    return FixedWeightedResidueTransitionBundle(
        scale_in=scale_in,
        scale_out=scale_out,
        weight_root=weight_root,
        weight_index_csv=weight_index_csv,
        row_keys=row_keys,
        xin=xin,
        xout=xout,
        summary=summary,
    )
