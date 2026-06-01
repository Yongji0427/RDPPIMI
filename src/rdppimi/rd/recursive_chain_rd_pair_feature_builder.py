from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import json
import time

import numpy as np
import pandas as pd

from rdppimi.rd.fixed_weighted_rd_pair_feature_builder import (
    LocalFullStepScaler,
    _canonical_and_resolved,
    find_rd_scaler_file,
)
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
FEATURE_TYPE = 'training_consistent_recursive_chain_rd_pair_embedding'
WEIGHT_SEMANTICS = 'fixed_residue_softmax_weights_t0p7'
EIGHT_M_SEMANTICS = 'weighted_residue_sum_pair_embedding'
RD_TARGET_SEMANTICS = 'rd_encoded_recursive_chain_pair_embedding'
RD_REFERENCE_IMPLEMENTATION = 'external/reverse_distillation/src/reverse_distillation/scaler/rd.py'
WEIGHT_APPLICATION = 'exactly_once'
AGGREGATION = 'sum_over_recursive_chain_residues'
RD_INFERENCE_MODE = 'recursive_full_step'
CHAIN_DEFINITION = (
    'chain[8M]=weighted_raw_8M; '
    'chain[35M]=step_8M_35M(chain[8M], weighted_raw_35M); '
    'chain[150M]=step_35M_150M(chain[35M], weighted_raw_150M); '
    'chain[650M]=step_150M_650M(chain[150M], weighted_raw_650M); '
    'chain[3B]=step_650M_3B(chain[650M], weighted_raw_3B)'
)
CHAIN_TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ('8M', '35M'),
    ('35M', '150M'),
    ('150M', '650M'),
    ('650M', '3B'),
)
TRANSITION_BY_TARGET = {scale_out: (scale_in, scale_out) for scale_in, scale_out in CHAIN_TRANSITIONS}
SCALES_IN_CHAIN_ORDER: Tuple[str, ...] = ('8M', '35M', '150M', '650M', '3B')
EXPECTED_STEP_SEQUENCE_BY_SCALE: Dict[str, Tuple[str, ...]] = {
    scale: tuple(f'{scale_in}->{scale_out}' for scale_in, scale_out in CHAIN_TRANSITIONS[:index])
    for index, scale in enumerate(SCALES_IN_CHAIN_ORDER)
}


def expected_step_sequence(scale: str) -> Tuple[str, ...]:
    scale = str(scale)
    if scale not in EXPECTED_STEP_SEQUENCE_BY_SCALE:
        raise ValueError(f'Unsupported recursive-chain scale for step_sequence validation: {scale}')
    return EXPECTED_STEP_SEQUENCE_BY_SCALE[scale]


def step_sequence_to_text(scale: str) -> str:
    return ';'.join(expected_step_sequence(scale))


def _is_blank_step_sequence_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == '' or text.lower() in {'nan', 'none', 'null'}


def normalize_step_sequence_value(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if _is_blank_step_sequence_value(value):
        return ()
    text = str(value).strip()
    if text.startswith('[') and text.endswith(']'):
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError(f'step_sequence JSON must decode to a list, got {type(loaded).__name__}')
        return normalize_step_sequence_value(loaded)
    return tuple(part.strip() for part in text.split(';') if part.strip())


def normalize_last_step_value(value: Any) -> str:
    if _is_blank_step_sequence_value(value):
        return ''
    return str(value).strip()


def validate_step_sequence_fields(
    scale: str,
    step_sequence: Any,
    chain_depth: Any,
    last_step: Any = '',
    context: str = '',
) -> Tuple[str, ...]:
    expected = expected_step_sequence(scale)
    observed = normalize_step_sequence_value(step_sequence)
    label = context or f'scale={scale}'
    if observed != expected:
        raise ValueError(
            f'step_sequence mismatch for {label}: got {list(observed)}, expected {list(expected)}'
        )
    try:
        observed_depth = int(chain_depth)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'chain_depth is not an integer for {label}: {chain_depth!r}') from exc
    expected_depth = len(expected)
    if observed_depth != expected_depth:
        raise ValueError(f'chain_depth mismatch for {label}: got {observed_depth}, expected {expected_depth}')
    observed_last_step = normalize_last_step_value(last_step)
    expected_last_step = expected[-1] if expected else ''
    if observed_last_step != expected_last_step:
        raise ValueError(
            f'last_step mismatch for {label}: got {observed_last_step!r}, expected {expected_last_step!r}'
        )
    return observed


def validate_metadata_step_sequence(metadata: Mapping[str, Any], context: str = '') -> Tuple[str, ...]:
    scale = str(metadata.get('model_key') or metadata.get('model_scale') or '')
    return validate_step_sequence_fields(
        scale=scale,
        step_sequence=metadata.get('step_sequence'),
        chain_depth=metadata.get('chain_depth'),
        last_step=metadata.get('last_step'),
        context=context or f'metadata pair={metadata.get("pair_id", "")} scale={scale}',
    )


@dataclass(frozen=True)
class RecursiveChainPairAssetBuildBundle:
    output_root: Path
    selected_pairs: pd.DataFrame
    per_scale_registry_paths: Dict[str, Path]
    combined_registry_path: Path
    pair_index_path: Path
    build_status_path: Path
    run_config_path: Path
    summary_path: Path
    loader_validation_path: Path


class RecursiveChainRDPairFeatureBuilder:
    def __init__(
        self,
        output_root: Path,
        weight_index_csv: Path = CANONICAL_WEIGHT_INDEX_CSV,
        rd_chain_root: Path = CANONICAL_RD_CHAIN_ROOT,
        limit_pairs: int = None,
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
        self.embedding_lookups = {scale: _read_embedding_lookup(scale) for scale in SCALES_IN_CHAIN_ORDER}
        _validate_scale_length_consistency(self.selected_pairs, self.embedding_lookups)
        self.scalers = self._load_scalers()

    def _load_scalers(self) -> Dict[Tuple[str, str], LocalFullStepScaler]:
        scalers: Dict[Tuple[str, str], LocalFullStepScaler] = {}
        for scale_in, scale_out in CHAIN_TRANSITIONS:
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

    def _load_weighted_raw_residue_matrices(
        self,
        protein_id: str,
        pair_id: str,
        side: str,
        length: int,
        weight_path: Path,
    ) -> Dict[str, np.ndarray]:
        weight = _load_weight_array(weight_path, length, f'{pair_id}:{side}')
        weighted_raw: Dict[str, np.ndarray] = {}
        for scale in SCALES_IN_CHAIN_ORDER:
            record = self.embedding_lookups[scale][protein_id]
            emb = _load_embedding_matrix(
                record.embedding_path,
                length,
                get_embedding_spec(scale).esm_dim,
                scale,
                f'{pair_id}:{side}',
            )
            weighted_raw[scale] = emb * weight[:, None]
        return weighted_raw

    def _sum_chain_residues(self, chain_residues: np.ndarray, scale: str, pair_id: str, side: str) -> np.ndarray:
        expected_dim = get_embedding_spec(scale).esm_dim
        if chain_residues.ndim != 2 or chain_residues.shape[1] != expected_dim:
            raise ValueError(
                f'Unexpected recursive chain residue matrix for {pair_id}:{side} scale={scale}: {chain_residues.shape}'
            )
        side_embedding = chain_residues.sum(axis=0, dtype=np.float64).astype(np.float32)
        if side_embedding.ndim != 1 or side_embedding.shape[0] != expected_dim:
            raise ValueError(
                f'Unexpected recursive chain side embedding shape for {pair_id}:{side} scale={scale}: {side_embedding.shape}'
            )
        if not np.isfinite(side_embedding).all():
            raise ValueError(f'Recursive chain side embedding contains NaN/inf for {pair_id}:{side} scale={scale}')
        return side_embedding

    def _build_recursive_chain_side_embeddings(
        self,
        weighted_raw: Mapping[str, np.ndarray],
        pair_id: str,
        side: str,
    ) -> Dict[str, np.ndarray]:
        chain_residues: Dict[str, np.ndarray] = {'8M': weighted_raw['8M']}
        for scale_in, scale_out in CHAIN_TRANSITIONS:
            scaler = self.scalers[(scale_in, scale_out)]
            chain_residues[scale_out] = scaler.step(chain_residues[scale_in], weighted_raw[scale_out])
        return {
            scale: self._sum_chain_residues(chain_residues[scale], scale, pair_id, side)
            for scale in SCALES_IN_CHAIN_ORDER
        }

    def _cache_dir(self, scale: str, pair_id: str) -> Path:
        return self.output_root / 'cache' / scale / pair_id

    @staticmethod
    def _step_sequence_for_scale(scale: str) -> List[str]:
        return list(expected_step_sequence(scale))

    def _raw_embedding_paths_for_protein(self, protein_id: str) -> Dict[str, str]:
        return {
            scale: str(self.embedding_lookups[scale][protein_id].embedding_path.absolute())
            for scale in SCALES_IN_CHAIN_ORDER
        }

    def _metadata_for_scale(
        self,
        row: pd.Series,
        scale: str,
        pair_id: str,
        cache_dir: Path,
    ) -> Dict[str, object]:
        len_a = int(row['length_A'])
        len_b = int(row['length_B'])
        protein_a = str(row['protein_id_A'])
        protein_b = str(row['protein_id_B'])
        step_sequence = self._step_sequence_for_scale(scale)
        chain_depth = len(step_sequence)
        metadata: Dict[str, object] = {
            'pair_id': pair_id,
            'proteinA_id': protein_a,
            'proteinB_id': protein_b,
            'model_key': scale,
            'model_scale': scale,
            'embed_dim': int(get_embedding_spec(scale).esm_dim),
            'lenA': len_a,
            'lenB': len_b,
            'weight_root': str(self.weight_root),
            'weight_root_resolved': str(self.weight_root_resolved),
            'weight_path_A': str(Path(row['path_weights_A']).absolute()),
            'weight_path_B': str(Path(row['path_weights_B']).absolute()),
            'training_consistent': True,
            'recursive_chain': True,
            'chain_depth': int(chain_depth),
            'chain_definition': CHAIN_DEFINITION,
            'step_sequence': step_sequence,
            'weight_application': WEIGHT_APPLICATION,
            'aggregation': AGGREGATION,
            'double_weighting': False,
            'feature_type': FEATURE_TYPE,
            'rd_inference_mode': RD_INFERENCE_MODE,
            'rd_reference_implementation': RD_REFERENCE_IMPLEMENTATION,
            'weight_semantics': WEIGHT_SEMANTICS,
            'target_representation_semantics': EIGHT_M_SEMANTICS if scale == '8M' else RD_TARGET_SEMANTICS,
            'weighted_embedding_A_path': str((cache_dir / 'weighted_embedding_A.npy').absolute()),
            'weighted_embedding_B_path': str((cache_dir / 'weighted_embedding_B.npy').absolute()),
            'metadata_path': str((cache_dir / 'metadata.json').absolute()),
            'cache_dir': str(cache_dir.absolute()),
            'rd_chain_root': str(self.rd_chain_root),
            'rd_chain_root_resolved': str(self.rd_chain_root_resolved),
            'raw_embedding_paths_A_by_scale': self._raw_embedding_paths_for_protein(protein_a),
            'raw_embedding_paths_B_by_scale': self._raw_embedding_paths_for_protein(protein_b),
        }
        if scale == '8M':
            metadata.update(
                {
                    'full_step_applied': False,
                    'last_step': None,
                    'scale_in_for_step': '8M',
                    'scale_out_for_step': '8M',
                    'recursive_xin_semantics': 'weighted_raw_8M',
                    'xout_semantics': 'weighted_raw_8M',
                    'rd_scaler_path': None,
                    'rd_scaler_path_resolved': None,
                    'rd_scaler_chain_paths': {},
                    'rd_scaler_chain_paths_resolved': {},
                }
            )
        else:
            scale_in, scale_out = TRANSITION_BY_TARGET[scale]
            scaler = self.scalers[(scale_in, scale_out)]
            chain_scalers = {}
            chain_scalers_resolved = {}
            for step in step_sequence:
                s_in, s_out = step.split('->')
                step_scaler = self.scalers[(s_in, s_out)]
                chain_scalers[step] = str(step_scaler.path)
                chain_scalers_resolved[step] = str(step_scaler.path_resolved)
            metadata.update(
                {
                    'full_step_applied': True,
                    'last_step': f'{scale_in}->{scale_out}',
                    'scale_in_for_step': scale_in,
                    'scale_out_for_step': scale_out,
                    'recursive_xin_semantics': f'chain[{scale_in}]',
                    'xout_semantics': f'weighted_raw_{scale_out}',
                    'rd_scaler_path': str(scaler.path),
                    'rd_scaler_path_resolved': str(scaler.path_resolved),
                    'rd_scaler_chain_paths': chain_scalers,
                    'rd_scaler_chain_paths_resolved': chain_scalers_resolved,
                }
            )
        validate_metadata_step_sequence(metadata, context=f'metadata pair={pair_id} scale={scale}')
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
        step_sequence = self._step_sequence_for_scale(scale)
        reg_row = {
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
            'rd_inference_mode': RD_INFERENCE_MODE,
            'recursive_chain': True,
            'chain_depth': int(len(step_sequence)),
            'step_sequence': step_sequence_to_text(scale),
            'last_step': step_sequence[-1] if step_sequence else '',
            'rd_reference_implementation': RD_REFERENCE_IMPLEMENTATION,
            'weight_root': str(self.weight_root),
            'weight_root_resolved': str(self.weight_root_resolved),
            'weight_semantics': WEIGHT_SEMANTICS,
            'weight_application': WEIGHT_APPLICATION,
            'aggregation': AGGREGATION,
            'double_weighting': False,
            'target_representation_semantics': EIGHT_M_SEMANTICS if scale == '8M' else RD_TARGET_SEMANTICS,
            'validation_status': 'pending',
            'validation_error': '',
        }
        validate_step_sequence_fields(
            scale=scale,
            step_sequence=reg_row['step_sequence'],
            chain_depth=reg_row['chain_depth'],
            last_step=reg_row['last_step'],
            context=f'registry row pair={row["pair_id"]} scale={scale}',
        )
        return reg_row

    def build(self) -> RecursiveChainPairAssetBuildBundle:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / 'cache').mkdir(parents=True, exist_ok=True)
        indices_dir = self.output_root / 'indices'
        indices_dir.mkdir(parents=True, exist_ok=True)

        per_scale_rows: Dict[str, List[Dict[str, object]]] = {scale: [] for scale in SCALES_IN_CHAIN_ORDER}
        build_rows: List[Dict[str, object]] = []

        for pair_index, row in enumerate(self.selected_pairs.itertuples(index=False), start=1):
            row_dict = row._asdict()
            row_series = pd.Series(row_dict)
            pair_id = str(row_dict['pair_id'])
            start_pair = time.time()
            try:
                weighted_raw_a = self._load_weighted_raw_residue_matrices(
                    protein_id=str(row_dict['protein_id_A']),
                    pair_id=pair_id,
                    side='A',
                    length=int(row_dict['length_A']),
                    weight_path=Path(row_dict['path_weights_A']),
                )
                weighted_raw_b = self._load_weighted_raw_residue_matrices(
                    protein_id=str(row_dict['protein_id_B']),
                    pair_id=pair_id,
                    side='B',
                    length=int(row_dict['length_B']),
                    weight_path=Path(row_dict['path_weights_B']),
                )
                side_embeddings_a = self._build_recursive_chain_side_embeddings(weighted_raw_a, pair_id, 'A')
                side_embeddings_b = self._build_recursive_chain_side_embeddings(weighted_raw_b, pair_id, 'B')
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                elapsed = time.time() - start_pair
                for scale in SCALES_IN_CHAIN_ORDER:
                    cache_dir = self._cache_dir(scale, pair_id)
                    metadata_path = cache_dir / 'metadata.json'
                    reg_row = self._registry_row(
                        row=row_series,
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

            for scale in SCALES_IN_CHAIN_ORDER:
                cache_dir = self._cache_dir(scale, pair_id)
                cache_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = cache_dir / 'metadata.json'
                scale_start = time.time()
                try:
                    emb_a = side_embeddings_a[scale]
                    emb_b = side_embeddings_b[scale]
                    np.save(cache_dir / 'weighted_embedding_A.npy', emb_a)
                    np.save(cache_dir / 'weighted_embedding_B.npy', emb_b)
                    metadata = self._metadata_for_scale(row_series, scale, pair_id, cache_dir)
                    with metadata_path.open('w') as f:
                        json.dump(metadata, f, indent=2, sort_keys=True)
                    elapsed = time.time() - scale_start
                    reg_row = self._registry_row(
                        row=row_series,
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
                        row=row_series,
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
                print(f'[recursive-chain-builder] completed {pair_index}/{len(self.selected_pairs)} pairs: {pair_id}')

        per_scale_registry_paths: Dict[str, Path] = {}
        combined_rows: List[pd.DataFrame] = []
        for scale in SCALES_IN_CHAIN_ORDER:
            df = pd.DataFrame(per_scale_rows[scale])
            registry_path = indices_dir / f'recursive_chain_rd_pair_registry_{scale}.csv'
            df.to_csv(registry_path, index=False)
            per_scale_registry_paths[scale] = registry_path
            combined_rows.append(df)

        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_registry_path = indices_dir / 'recursive_chain_rd_pair_registry_all_models.csv'
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
            'rd_reference_implementation': RD_REFERENCE_IMPLEMENTATION,
            'limit_pairs': self.limit_pairs,
            'pair_count': int(len(self.selected_pairs)),
            'scales': list(SCALES_IN_CHAIN_ORDER),
            'feature_type': FEATURE_TYPE,
            'rd_inference_mode': RD_INFERENCE_MODE,
            'training_consistent': True,
            'recursive_chain': True,
            'chain_definition': CHAIN_DEFINITION,
            'weight_application': WEIGHT_APPLICATION,
            'aggregation': AGGREGATION,
            'target_representation_semantics_8M': EIGHT_M_SEMANTICS,
            'target_representation_semantics_35M_plus': RD_TARGET_SEMANTICS,
            'model_key_values': list(SCALES_IN_CHAIN_ORDER),
        }
        run_config_path = self.output_root / 'run_config.json'
        with run_config_path.open('w') as f:
            json.dump(run_config, f, indent=2, sort_keys=True)

        summary_path = self.output_root / 'recursive_chain_rd_pair_build_summary.md'
        summary_lines = [
            '# Recursive-Chain RD Pair Build Summary',
            '',
            f'- output_root: `{self.output_root}`',
            f'- output_root_resolved: `{self.output_root_resolved}`',
            f'- pair_count: `{len(self.selected_pairs)}`',
            f'- weight_index_csv: `{self.weight_index_csv}`',
            f'- weight_index_csv_resolved: `{self.weight_index_csv_resolved}`',
            f'- rd_chain_root: `{self.rd_chain_root}`',
            f'- rd_chain_root_resolved: `{self.rd_chain_root_resolved}`',
            f'- rd_reference_implementation: `{RD_REFERENCE_IMPLEMENTATION}`',
            f'- feature_type: `{FEATURE_TYPE}`',
            f'- rd_inference_mode: `{RD_INFERENCE_MODE}`',
            f'- recursive_chain: `true`',
            f'- chain_definition: `{CHAIN_DEFINITION}`',
            f'- weight_application: `{WEIGHT_APPLICATION}`',
            f'- aggregation: `{AGGREGATION}`',
            f'- model_key values: `{", ".join(SCALES_IN_CHAIN_ORDER)}`',
            f'- 8M target_representation_semantics: `{EIGHT_M_SEMANTICS}`',
            f'- 35M/150M/650M/3B target_representation_semantics: `{RD_TARGET_SEMANTICS}`',
            '',
            '## Per-scale status',
            '',
        ]
        for scale in SCALES_IN_CHAIN_ORDER:
            df = pd.DataFrame(per_scale_rows[scale])
            success_count = int(df['status'].isin(['success', 'cached']).sum()) if not df.empty else 0
            failure_count = int((df['status'] == 'failed').sum()) if not df.empty else 0
            summary_lines.append(
                f'- {scale}: success=`{success_count}` failure=`{failure_count}` dim=`{get_embedding_spec(scale).esm_dim}` chain_depth=`{len(self._step_sequence_for_scale(scale))}`'
            )
        summary_path.write_text('\n'.join(summary_lines) + '\n')

        loader_validation_path = self.output_root / 'loader_validation.json'
        return RecursiveChainPairAssetBuildBundle(
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
