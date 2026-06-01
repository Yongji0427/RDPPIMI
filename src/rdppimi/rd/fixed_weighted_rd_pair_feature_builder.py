from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
import json
import time

import numpy as np
import pandas as pd

from rdppimi.rd.fixed_weighted_residue_rd_loader import (
    ALL_SCALES,
    CANONICAL_WEIGHT_INDEX_CSV,
    PAIR_MANIFEST_CSV,
    PPI_PRIOR_INDEX_CSV,
    _align_selected_pairs,
    _load_embedding_matrix,
    _load_weight_array,
    _read_embedding_lookup,
    _read_manifest,
    _read_prior_index,
    _read_weight_index,
    _validate_scale_length_consistency,
)
from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


CANONICAL_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_RD_CHAIN_ROOT = (
    CANONICAL_PROJECT_ROOT / 'outputs' / 'rd_scalers' / 'full_chain'
)
FEATURE_TYPE = 'training_consistent_adjacent_true_scale_full_step_rd_pair_embedding'
WEIGHT_SEMANTICS = 'fixed_residue_softmax_weights_t0p7'
EIGHT_M_SEMANTICS = 'weighted_residue_sum_pair_embedding'
RD_TARGET_SEMANTICS = 'rd_encoded_full_step_pair_embedding'
TRANSITION_BY_TARGET = {
    '35M': ('8M', '35M'),
    '150M': ('35M', '150M'),
    '650M': ('150M', '650M'),
    '3B': ('650M', '3B'),
}


def _canonical_and_resolved(path: Path | str) -> Tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    canonical = candidate.absolute()
    resolved = canonical.resolve()
    try:
        rel = resolved.relative_to(CANONICAL_PROJECT_ROOT.resolve())
        canonical = CANONICAL_PROJECT_ROOT / rel
    except ValueError:
        pass
    return canonical, resolved


def infer_rd_scaler_types(rd_chain_root_resolved: Path) -> Tuple[str, str]:
    metadata_path = rd_chain_root_resolved / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        regressor_type = str(metadata.get('regressor_type') or 'linear')
        pca_type = str(metadata.get('pca_type') or 'incremental')
        return regressor_type, pca_type
    return 'linear', 'incremental'


def find_rd_scaler_file(
    rd_chain_root: Path,
    rd_chain_root_resolved: Path,
    scale_in: str,
    scale_out: str,
) -> Tuple[Path, Path, str, str]:
    regressor_type, pca_type = infer_rd_scaler_types(rd_chain_root_resolved)
    pattern = f'rd-scaler-{scale_in}-{scale_out}-{regressor_type}-{pca_type}-*.npz'
    matches = sorted(rd_chain_root_resolved.glob(pattern))
    if len(matches) != 1:
        fallback_pattern = f'rd-scaler-{scale_in}-{scale_out}-*-*.npz'
        fallback_matches = sorted(rd_chain_root_resolved.glob(fallback_pattern))
        if len(fallback_matches) == 1:
            matches = fallback_matches
            name = matches[0].name
            prefix = f'rd-scaler-{scale_in}-{scale_out}-'
            suffix = name[len(prefix):].split('-fixedresidue', 1)[0]
            parts = suffix.split('-')
            if len(parts) >= 2:
                regressor_type, pca_type = parts[0], parts[1]
        else:
            raise FileNotFoundError(
                f'Expected exactly one RD scaler for {scale_in}->{scale_out} under {rd_chain_root_resolved}, '
                f'got {len(matches)} matches for pattern {pattern} and {len(fallback_matches)} matches '
                f'for fallback pattern {fallback_pattern}'
            )
    display_path = rd_chain_root / matches[0].name
    return matches[0], display_path, regressor_type, pca_type


@dataclass(frozen=True)
class LocalFullStepScaler:
    scale_in: str
    scale_out: str
    path: Path
    path_resolved: Path
    coef: np.ndarray
    intercept: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    regressor_type: str = 'linear'
    pca_type: str = 'incremental'
    input_mean: np.ndarray | None = None
    output_mean: np.ndarray | None = None
    input_components: np.ndarray | None = None
    n_significant_pcs: int | None = None
    johnstone_threshold: float | None = None
    sigma_sq: float | None = None

    @classmethod
    def from_npz(
        cls,
        path: Path,
        scale_in: str,
        scale_out: str,
        display_path: Path | None = None,
        regressor_type: str | None = None,
        pca_type: str | None = None,
    ) -> 'LocalFullStepScaler':
        canonical_path, resolved_path = _canonical_and_resolved(path)
        if display_path is not None:
            canonical_path = Path(display_path).absolute()
        if not resolved_path.exists():
            raise FileNotFoundError(f'Missing RD scaler file: {resolved_path}')
        with np.load(resolved_path) as state:
            required = {
                'regressor__coef_',
                'regressor__intercept_',
                'pca__mean_',
                'pca__components_',
            }
            missing = required - set(state.files)
            if missing:
                raise ValueError(f'RD scaler missing required keys {sorted(missing)}: {resolved_path}')
            coef = np.asarray(state['regressor__coef_'], dtype=np.float32)
            intercept = np.asarray(state['regressor__intercept_'], dtype=np.float32)
            pca_mean = np.asarray(state['pca__mean_'], dtype=np.float32)
            pca_components = np.asarray(state['pca__components_'], dtype=np.float32)
            inferred_regressor_type = 'pcr' if 'regressor__input_components_' in state.files else 'linear'
            effective_regressor_type = str(regressor_type or inferred_regressor_type)
            effective_pca_type = str(pca_type or 'incremental')
            input_mean = None
            output_mean = None
            input_components = None
            n_significant_pcs = None
            johnstone_threshold = None
            sigma_sq = None
            if effective_regressor_type == 'pcr':
                pcr_required = {
                    'regressor__input_mean_',
                    'regressor__output_mean_',
                    'regressor__input_components_',
                    'regressor__n_significant_pcs_',
                }
                pcr_missing = pcr_required - set(state.files)
                if pcr_missing:
                    raise ValueError(f'PCR RD scaler missing required keys {sorted(pcr_missing)}: {resolved_path}')
                input_mean = np.asarray(state['regressor__input_mean_'], dtype=np.float32)
                output_mean = np.asarray(state['regressor__output_mean_'], dtype=np.float32)
                input_components = np.asarray(state['regressor__input_components_'], dtype=np.float32)
                n_significant_pcs = int(np.asarray(state['regressor__n_significant_pcs_']).reshape(-1)[0])
                if 'regressor__johnstone_threshold_' in state.files:
                    johnstone_threshold = float(np.asarray(state['regressor__johnstone_threshold_']).reshape(-1)[0])
                if 'regressor__sigma_sq_' in state.files:
                    sigma_sq = float(np.asarray(state['regressor__sigma_sq_']).reshape(-1)[0])

        input_dim = get_embedding_spec(scale_in).esm_dim
        output_dim = get_embedding_spec(scale_out).esm_dim
        expected_residual_width = output_dim - input_dim
        if effective_regressor_type == 'pcr':
            if input_mean is None or output_mean is None or input_components is None or n_significant_pcs is None:
                raise ValueError(f'PCR RD scaler state was not fully loaded: {resolved_path}')
            if input_mean.shape != (input_dim,):
                raise ValueError(
                    f'Unexpected PCR input mean shape for {scale_in}->{scale_out}: '
                    f'expected {(input_dim,)}, got {input_mean.shape}: {resolved_path}'
                )
            if output_mean.shape != (output_dim,):
                raise ValueError(
                    f'Unexpected PCR output mean shape for {scale_in}->{scale_out}: '
                    f'expected {(output_dim,)}, got {output_mean.shape}: {resolved_path}'
                )
            if input_components.ndim != 2 or input_components.shape[0] != input_dim:
                raise ValueError(
                    f'Unexpected PCR input components shape for {scale_in}->{scale_out}: '
                    f'expected first dim {input_dim}, got {input_components.shape}: {resolved_path}'
                )
            if n_significant_pcs < 1 or n_significant_pcs > input_components.shape[1]:
                raise ValueError(
                    f'Invalid PCR selected PC count for {scale_in}->{scale_out}: '
                    f'{n_significant_pcs} with components shape {input_components.shape}: {resolved_path}'
                )
            if coef.shape != (output_dim, n_significant_pcs):
                raise ValueError(
                    f'Unexpected PCR regressor coef shape for {scale_in}->{scale_out}: '
                    f'expected {(output_dim, n_significant_pcs)}, got {coef.shape}: {resolved_path}'
                )
        elif coef.shape != (output_dim, input_dim):
            raise ValueError(
                f'Unexpected regressor coef shape for {scale_in}->{scale_out}: '
                f'expected {(output_dim, input_dim)}, got {coef.shape}: {resolved_path}'
            )
        if intercept.shape != (output_dim,):
            raise ValueError(
                f'Unexpected regressor intercept shape for {scale_in}->{scale_out}: '
                f'expected {(output_dim,)}, got {intercept.shape}: {resolved_path}'
            )
        if pca_mean.shape != (output_dim,):
            raise ValueError(
                f'Unexpected pca mean shape for {scale_in}->{scale_out}: '
                f'expected {(output_dim,)}, got {pca_mean.shape}: {resolved_path}'
            )
        if pca_components.shape != (expected_residual_width, output_dim):
            raise ValueError(
                f'Unexpected pca components shape for {scale_in}->{scale_out}: '
                f'expected {(expected_residual_width, output_dim)}, got {pca_components.shape}: {resolved_path}'
            )
        return cls(
            scale_in=scale_in,
            scale_out=scale_out,
            path=canonical_path,
            path_resolved=resolved_path,
            coef=coef,
            intercept=intercept,
            pca_mean=pca_mean,
            pca_components=pca_components,
            regressor_type=effective_regressor_type,
            pca_type=effective_pca_type,
            input_mean=input_mean,
            output_mean=output_mean,
            input_components=input_components,
            n_significant_pcs=n_significant_pcs,
            johnstone_threshold=johnstone_threshold,
            sigma_sq=sigma_sq,
        )

    @property
    def input_dim(self) -> int:
        return int(get_embedding_spec(self.scale_in).esm_dim)

    @property
    def output_dim(self) -> int:
        return int(get_embedding_spec(self.scale_out).esm_dim)

    @property
    def residual_width(self) -> int:
        return int(self.pca_components.shape[0])

    def step(self, xin: np.ndarray, xout: np.ndarray) -> np.ndarray:
        if xin.ndim != 2 or xout.ndim != 2:
            raise ValueError(
                f'RD full-step expects 2D xin/xout, got shapes {xin.shape} and {xout.shape} '
                f'for {self.scale_in}->{self.scale_out}'
            )
        if xin.shape[0] != xout.shape[0]:
            raise ValueError(
                f'RD full-step row mismatch for {self.scale_in}->{self.scale_out}: '
                f'{xin.shape[0]} vs {xout.shape[0]}'
            )
        if xin.shape[1] != self.input_dim:
            raise ValueError(
                f'RD full-step input dim mismatch for {self.scale_in}->{self.scale_out}: '
                f'expected {self.input_dim}, got {xin.shape[1]}'
            )
        if xout.shape[1] != self.output_dim:
            raise ValueError(
                f'RD full-step output dim mismatch for {self.scale_in}->{self.scale_out}: '
                f'expected {self.output_dim}, got {xout.shape[1]}'
            )

        xin = np.asarray(xin, dtype=np.float32)
        xout = np.asarray(xout, dtype=np.float32)
        if self.regressor_type == 'pcr':
            if self.input_mean is None or self.output_mean is None or self.input_components is None or self.n_significant_pcs is None:
                raise ValueError(f'PCR RD scaler state is incomplete for {self.scale_in}->{self.scale_out}')
            x_centered = xin - self.input_mean
            x_pc = x_centered @ self.input_components[:, : self.n_significant_pcs]
            xin_transformed = x_pc @ self.coef.T + self.intercept + self.output_mean
        else:
            xin_transformed = xin @ self.coef.T + self.intercept
        residuals = xout - xin_transformed
        residuals_pc = (residuals - self.pca_mean) @ self.pca_components.T
        scaled = np.concatenate([xin, residuals_pc], axis=1)
        if scaled.shape != xout.shape:
            raise ValueError(
                f'RD full-step shape mismatch for {self.scale_in}->{self.scale_out}: '
                f'step output {scaled.shape}, expected {xout.shape}'
            )
        if not np.isfinite(scaled).all():
            raise ValueError(f'RD full-step produced NaN/inf for {self.scale_in}->{self.scale_out}')
        return scaled.astype(np.float32, copy=False)


@dataclass(frozen=True)
class PairAssetBuildBundle:
    output_root: Path
    selected_pairs: pd.DataFrame
    per_scale_registry_paths: Dict[str, Path]
    combined_registry_path: Path
    pair_index_path: Path
    build_status_path: Path
    run_config_path: Path
    summary_path: Path
    loader_validation_path: Path


class FixedWeightedRDPairFeatureBuilder:
    def __init__(
        self,
        output_root: Path,
        weight_index_csv: Path = CANONICAL_WEIGHT_INDEX_CSV,
        rd_chain_root: Path = CANONICAL_RD_CHAIN_ROOT,
        limit_pairs: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.output_root, self.output_root_resolved = _canonical_and_resolved(output_root)
        self.weight_index_csv, self.weight_index_csv_resolved = _canonical_and_resolved(weight_index_csv)
        self.rd_chain_root, self.rd_chain_root_resolved = _canonical_and_resolved(rd_chain_root)
        self.weight_root = self.weight_index_csv.parent
        self.weight_root_resolved = self.weight_index_csv_resolved.parent
        self.limit_pairs = limit_pairs
        self.verbose = verbose

        self.weight_df = _read_weight_index(self.weight_index_csv_resolved)
        self.manifest_df = _read_manifest(PAIR_MANIFEST_CSV)
        self.prior_df = _read_prior_index(PPI_PRIOR_INDEX_CSV)
        self.selected_pairs = _align_selected_pairs(
            weight_df=self.weight_df,
            manifest_df=self.manifest_df,
            prior_df=self.prior_df,
            limit_pairs=self.limit_pairs,
        )
        self.embedding_lookups = {scale: _read_embedding_lookup(scale) for scale in ALL_SCALES}
        _validate_scale_length_consistency(self.selected_pairs, self.embedding_lookups)
        self.scalers = self._load_scalers()

    def _load_scalers(self) -> Dict[Tuple[str, str], LocalFullStepScaler]:
        scalers: Dict[Tuple[str, str], LocalFullStepScaler] = {}
        for scale_out, (scale_in, _) in TRANSITION_BY_TARGET.items():
            scaler_path, display_path, regressor_type, pca_type = find_rd_scaler_file(
                self.rd_chain_root,
                self.rd_chain_root_resolved,
                scale_in,
                scale_out,
            )
            scalers[(scale_in, scale_out)] = LocalFullStepScaler.from_npz(
                scaler_path,
                scale_in,
                scale_out,
                display_path=display_path,
                regressor_type=regressor_type,
                pca_type=pca_type,
            )
        return scalers

    def _load_weighted_residue_matrices(
        self,
        protein_id: str,
        pair_id: str,
        side: str,
        length: int,
        weight_path: Path,
    ) -> Dict[str, np.ndarray]:
        weight = _load_weight_array(weight_path, length, f'{pair_id}:{side}')
        weighted: Dict[str, np.ndarray] = {}
        for scale in ALL_SCALES:
            record = self.embedding_lookups[scale][protein_id]
            emb = _load_embedding_matrix(
                record.embedding_path,
                length,
                get_embedding_spec(scale).esm_dim,
                scale,
                f'{pair_id}:{side}',
            )
            weighted[scale] = emb * weight[:, None]
        return weighted

    def _sum_weighted_residues(self, weighted_residues: np.ndarray, scale: str, pair_id: str, side: str) -> np.ndarray:
        expected_dim = get_embedding_spec(scale).esm_dim
        if weighted_residues.ndim != 2 or weighted_residues.shape[1] != expected_dim:
            raise ValueError(
                f'Unexpected weighted residue matrix for {pair_id}:{side} scale={scale}: {weighted_residues.shape}'
            )
        side_embedding = weighted_residues.sum(axis=0, dtype=np.float64).astype(np.float32)
        if side_embedding.ndim != 1 or side_embedding.shape[0] != expected_dim:
            raise ValueError(
                f'Unexpected side embedding shape for {pair_id}:{side} scale={scale}: {side_embedding.shape}'
            )
        if not np.isfinite(side_embedding).all():
            raise ValueError(f'Side embedding contains NaN/inf for {pair_id}:{side} scale={scale}')
        return side_embedding

    def _build_scale_side_embedding(
        self,
        weighted_residues: Mapping[str, np.ndarray],
        scale: str,
        pair_id: str,
        side: str,
    ) -> np.ndarray:
        if scale == '8M':
            return self._sum_weighted_residues(weighted_residues['8M'], scale, pair_id, side)
        scale_in, scale_out = TRANSITION_BY_TARGET[scale]
        scaler = self.scalers[(scale_in, scale_out)]
        scaled_weighted_residues = scaler.step(weighted_residues[scale_in], weighted_residues[scale_out])
        return self._sum_weighted_residues(scaled_weighted_residues, scale, pair_id, side)

    def _cache_dir(self, scale: str, pair_id: str) -> Path:
        return self.output_root / 'cache' / scale / pair_id

    def _metadata_for_scale(
        self,
        row: pd.Series,
        scale: str,
        pair_id: str,
        cache_dir: Path,
    ) -> Dict[str, object]:
        model_key = scale
        len_a = int(row['length_A'])
        len_b = int(row['length_B'])
        protein_a = str(row['protein_id_A'])
        protein_b = str(row['protein_id_B'])
        metadata: Dict[str, object] = {
            'pair_id': pair_id,
            'proteinA_id': protein_a,
            'proteinB_id': protein_b,
            'model_key': model_key,
            'model_scale': model_key,
            'embed_dim': int(get_embedding_spec(scale).esm_dim),
            'lenA': len_a,
            'lenB': len_b,
            'weight_root': str(self.weight_root),
            'weight_root_resolved': str(self.weight_root_resolved),
            'weight_path_A': str(Path(row['path_weights_A']).absolute()),
            'weight_path_B': str(Path(row['path_weights_B']).absolute()),
            'training_consistent': True,
            'recursive_chain': False,
            'weight_application': 'exactly_once',
            'aggregation': 'sum_over_weighted_residues',
            'double_weighting': False,
            'feature_type': FEATURE_TYPE,
            'rd_inference_mode': 'full_step',
            'weight_semantics': WEIGHT_SEMANTICS,
            'target_representation_semantics': EIGHT_M_SEMANTICS if scale == '8M' else RD_TARGET_SEMANTICS,
            'weighted_embedding_A_path': str((cache_dir / 'weighted_embedding_A.npy').absolute()),
            'weighted_embedding_B_path': str((cache_dir / 'weighted_embedding_B.npy').absolute()),
            'metadata_path': str((cache_dir / 'metadata.json').absolute()),
            'cache_dir': str(cache_dir.absolute()),
            'rd_chain_root': str(self.rd_chain_root),
            'rd_chain_root_resolved': str(self.rd_chain_root_resolved),
        }
        if scale == '8M':
            emb_a_8m = self.embedding_lookups['8M'][protein_a].embedding_path.absolute()
            emb_b_8m = self.embedding_lookups['8M'][protein_b].embedding_path.absolute()
            metadata.update(
                {
                    'xin_embedding_path_A': str(emb_a_8m),
                    'xin_embedding_path_B': str(emb_b_8m),
                    'xout_embedding_path_A': str(emb_a_8m),
                    'xout_embedding_path_B': str(emb_b_8m),
                    'rd_scaler_path': None,
                    'rd_scaler_path_resolved': None,
                    'full_step_applied': False,
                    'scale_in_for_step': '8M',
                    'scale_out_for_step': '8M',
                }
            )
        else:
            scale_in, scale_out = TRANSITION_BY_TARGET[scale]
            scaler = self.scalers[(scale_in, scale_out)]
            metadata.update(
                {
                    'xin_embedding_path_A': str(self.embedding_lookups[scale_in][protein_a].embedding_path.absolute()),
                    'xin_embedding_path_B': str(self.embedding_lookups[scale_in][protein_b].embedding_path.absolute()),
                    'xout_embedding_path_A': str(self.embedding_lookups[scale_out][protein_a].embedding_path.absolute()),
                    'xout_embedding_path_B': str(self.embedding_lookups[scale_out][protein_b].embedding_path.absolute()),
                    'rd_scaler_path': str(scaler.path),
                    'rd_scaler_path_resolved': str(scaler.path_resolved),
                    'full_step_applied': True,
                    'scale_in_for_step': scale_in,
                    'scale_out_for_step': scale_out,
                }
            )
        return metadata

    def _registry_row(
        self,
        row: pd.Series,
        scale: str,
        cache_dir: Path,
        metadata_path: Path,
        elapsed_sec: float,
        status: str,
        error: str,
    ) -> Dict[str, object]:
        return {
            'pair_id': str(row['pair_id']),
            'proteinA_id': str(row['protein_id_A']),
            'proteinB_id': str(row['protein_id_B']),
            'model_key': scale,
            'model_scale': scale,
            'weighted_embedding_A_path': str((cache_dir / 'weighted_embedding_A.npy').absolute()) if status in {'success', 'cached'} else '',
            'weighted_embedding_B_path': str((cache_dir / 'weighted_embedding_B.npy').absolute()) if status in {'success', 'cached'} else '',
            'metadata_path': str(metadata_path.absolute()) if status in {'success', 'cached'} else '',
            'cache_dir': str(cache_dir.absolute()),
            'status': status,
            'error': error,
            'elapsed_sec': round(float(elapsed_sec), 6),
            'feature_type': FEATURE_TYPE,
            'rd_inference_mode': 'full_step',
            'weight_root': str(self.weight_root),
            'weight_root_resolved': str(self.weight_root_resolved),
            'weight_semantics': WEIGHT_SEMANTICS,
            'aggregation': 'sum_over_weighted_residues',
            'double_weighting': False,
            'target_representation_semantics': EIGHT_M_SEMANTICS if scale == '8M' else RD_TARGET_SEMANTICS,
            'validation_status': 'pending',
            'validation_error': '',
        }

    def build(self) -> PairAssetBuildBundle:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / 'cache').mkdir(parents=True, exist_ok=True)
        indices_dir = self.output_root / 'indices'
        indices_dir.mkdir(parents=True, exist_ok=True)

        per_scale_rows: Dict[str, List[Dict[str, object]]] = {scale: [] for scale in ALL_SCALES}
        build_rows: List[Dict[str, object]] = []

        for pair_index, row in enumerate(self.selected_pairs.itertuples(index=False), start=1):
            row_dict = row._asdict()
            pair_id = str(row_dict['pair_id'])
            start_pair = time.time()
            try:
                weighted_a = self._load_weighted_residue_matrices(
                    protein_id=str(row_dict['protein_id_A']),
                    pair_id=pair_id,
                    side='A',
                    length=int(row_dict['length_A']),
                    weight_path=Path(row_dict['path_weights_A']),
                )
                weighted_b = self._load_weighted_residue_matrices(
                    protein_id=str(row_dict['protein_id_B']),
                    pair_id=pair_id,
                    side='B',
                    length=int(row_dict['length_B']),
                    weight_path=Path(row_dict['path_weights_B']),
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                elapsed = time.time() - start_pair
                for scale in ALL_SCALES:
                    cache_dir = self._cache_dir(scale, pair_id)
                    metadata_path = cache_dir / 'metadata.json'
                    reg_row = self._registry_row(
                        row=pd.Series(row_dict),
                        scale=scale,
                        cache_dir=cache_dir,
                        metadata_path=metadata_path,
                        elapsed_sec=elapsed,
                        status='failed',
                        error=error,
                    )
                    reg_row['validation_status'] = 'failed'
                    reg_row['validation_error'] = error
                    per_scale_rows[scale].append(reg_row)
                    build_rows.append({'pair_id': pair_id, 'scale': scale, 'status': 'failed', 'error': error})
                continue

            for scale in ALL_SCALES:
                cache_dir = self._cache_dir(scale, pair_id)
                cache_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = cache_dir / 'metadata.json'
                scale_start = time.time()
                try:
                    emb_a = self._build_scale_side_embedding(weighted_a, scale, pair_id, 'A')
                    emb_b = self._build_scale_side_embedding(weighted_b, scale, pair_id, 'B')
                    np.save(cache_dir / 'weighted_embedding_A.npy', emb_a)
                    np.save(cache_dir / 'weighted_embedding_B.npy', emb_b)
                    metadata = self._metadata_for_scale(pd.Series(row_dict), scale, pair_id, cache_dir)
                    with metadata_path.open('w') as f:
                        json.dump(metadata, f, indent=2, sort_keys=True)
                    elapsed = time.time() - scale_start
                    reg_row = self._registry_row(
                        row=pd.Series(row_dict),
                        scale=scale,
                        cache_dir=cache_dir,
                        metadata_path=metadata_path,
                        elapsed_sec=elapsed,
                        status='success',
                        error='',
                    )
                    reg_row['validation_status'] = 'success'
                    per_scale_rows[scale].append(reg_row)
                    build_rows.append({'pair_id': pair_id, 'scale': scale, 'status': 'success', 'error': ''})
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    elapsed = time.time() - scale_start
                    reg_row = self._registry_row(
                        row=pd.Series(row_dict),
                        scale=scale,
                        cache_dir=cache_dir,
                        metadata_path=metadata_path,
                        elapsed_sec=elapsed,
                        status='failed',
                        error=error,
                    )
                    reg_row['validation_status'] = 'failed'
                    reg_row['validation_error'] = error
                    per_scale_rows[scale].append(reg_row)
                    build_rows.append({'pair_id': pair_id, 'scale': scale, 'status': 'failed', 'error': error})
            if self.verbose:
                print(f'[builder] completed {pair_index}/{len(self.selected_pairs)} pairs: {pair_id}')

        per_scale_registry_paths: Dict[str, Path] = {}
        combined_rows: List[pd.DataFrame] = []
        for scale in ALL_SCALES:
            df = pd.DataFrame(per_scale_rows[scale])
            registry_path = indices_dir / f'fixed_weighted_rd_pair_registry_{scale}.csv'
            df.to_csv(registry_path, index=False)
            per_scale_registry_paths[scale] = registry_path
            combined_rows.append(df)

        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_registry_path = indices_dir / 'fixed_weighted_rd_pair_registry_all_models.csv'
        combined_df.to_csv(combined_registry_path, index=False)

        pair_index_df = self.selected_pairs[['pair_id', 'protein_id_A', 'protein_id_B', 'length_A', 'length_B']].copy()
        pair_index_df.insert(0, 'pair_index', range(len(pair_index_df)))
        pair_index_df = pair_index_df.rename(
            columns={
                'protein_id_A': 'proteinA_id',
                'protein_id_B': 'proteinB_id',
                'length_A': 'lenA',
                'length_B': 'lenB',
            }
        )
        pair_index_path = self.output_root / 'pair_index.csv'
        pair_index_df.to_csv(pair_index_path, index=False)

        build_status_df = pd.DataFrame(build_rows)
        build_status_path = self.output_root / 'build_status.csv'
        build_status_df.to_csv(build_status_path, index=False)

        run_config = {
            'output_root': str(self.output_root),
            'output_root_resolved': str(self.output_root_resolved),
            'weight_index_csv': str(self.weight_index_csv),
            'weight_index_csv_resolved': str(self.weight_index_csv_resolved),
            'weight_root': str(self.weight_root),
            'weight_root_resolved': str(self.weight_root_resolved),
            'rd_chain_root': str(self.rd_chain_root),
            'rd_chain_root_resolved': str(self.rd_chain_root_resolved),
            'limit_pairs': self.limit_pairs,
            'pair_count': int(len(self.selected_pairs)),
            'scales': list(ALL_SCALES),
            'feature_type': FEATURE_TYPE,
            'training_consistent': True,
            'recursive_chain': False,
            'weight_application': 'exactly_once',
            'aggregation': 'sum_over_weighted_residues',
            'target_representation_semantics_8M': EIGHT_M_SEMANTICS,
            'target_representation_semantics_35M_plus': RD_TARGET_SEMANTICS,
            'model_key_values': list(ALL_SCALES),
        }
        run_config_path = self.output_root / 'run_config.json'
        with run_config_path.open('w') as f:
            json.dump(run_config, f, indent=2, sort_keys=True)

        summary_path = self.output_root / 'fixed_weighted_rd_pair_build_summary.md'
        summary_lines = [
            '# Fixed Weighted RD Pair Build Summary',
            '',
            f'- output_root: `{self.output_root}`',
            f'- output_root_resolved: `{self.output_root_resolved}`',
            f'- pair_count: `{len(self.selected_pairs)}`',
            f'- weight_index_csv: `{self.weight_index_csv}`',
            f'- weight_index_csv_resolved: `{self.weight_index_csv_resolved}`',
            f'- weight_root: `{self.weight_root}`',
            f'- weight_root_resolved: `{self.weight_root_resolved}`',
            f'- rd_chain_root: `{self.rd_chain_root}`',
            f'- rd_chain_root_resolved: `{self.rd_chain_root_resolved}`',
            f'- feature_type: `{FEATURE_TYPE}`',
            f'- training_consistent: `true`',
            f'- recursive_chain: `false`',
            f'- weight_application: `exactly_once`',
            f'- model_key values: `{", ".join(ALL_SCALES)}`',
            f'- 8M target_representation_semantics: `{EIGHT_M_SEMANTICS}`',
            f'- 35M/150M/650M/3B target_representation_semantics: `{RD_TARGET_SEMANTICS}`',
            '',
            '## Per-scale status',
            '',
        ]
        for scale in ALL_SCALES:
            df = pd.DataFrame(per_scale_rows[scale])
            success_count = int(df['status'].isin(['success', 'cached']).sum()) if not df.empty else 0
            failure_count = int((df['status'] == 'failed').sum()) if not df.empty else 0
            summary_lines.append(
                f'- {scale}: success=`{success_count}` failure=`{failure_count}` dim=`{get_embedding_spec(scale).esm_dim}`'
            )
        summary_path.write_text('\n'.join(summary_lines) + '\n')

        loader_validation_path = self.output_root / 'loader_validation.json'
        return PairAssetBuildBundle(
            output_root=self.output_root,
            selected_pairs=self.selected_pairs,
            per_scale_registry_paths=per_scale_registry_paths,
            combined_registry_path=combined_registry_path,
            pair_index_path=pair_index_path,
            build_status_path=build_status_path,
            run_config_path=run_config_path,
            summary_path=summary_path,
            loader_validation_path=loader_validation_path,
        )
