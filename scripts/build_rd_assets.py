#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
for candidate in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec  # noqa: E402
from rdppimi.rd.recursive_chain_rd_pair_feature_builder import (  # noqa: E402
    CANONICAL_RD_CHAIN_ROOT,
    FEATURE_TYPE,
    RD_INFERENCE_MODE,
    RD_REFERENCE_IMPLEMENTATION,
    RD_TARGET_SEMANTICS,
    SCALES_IN_CHAIN_ORDER,
    WEIGHT_SEMANTICS,
    RecursiveChainRDPairFeatureBuilder,
    step_sequence_to_text,
    validate_step_sequence_fields,
)
from rdppimi.rd.fixed_weighted_residue_rd_loader import CANONICAL_WEIGHT_INDEX_CSV  # noqa: E402
from rdppimi.ppimi.datasets.PPIMI_datasets import _load_weighted_pair_embedding_map  # noqa: E402

DEFAULT_ADJACENT_REGISTRY = (
    PROJECT_ROOT
    / 'outputs'
    / 'rd_pair_assets'
    / 'adjacent_full_step'
    / 'indices'
    / 'fixed_weighted_rd_pair_registry_all_models.csv'
)
NON_ALLCLOSE_SCALES = ('150M', '650M', '3B')
ALLCLOSE_RTOL = 1e-5
ALLCLOSE_ATOL = 1e-8


def timestamp_label() -> str:
    return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def default_output_root() -> Path:
    return PROJECT_ROOT / 'outputs' / 'rd_pair_assets' / f'recursive_chain_{timestamp_label()}'


def validate_registry_semantics(registry_path: Path, expected_pairs: int) -> dict:
    df = pd.read_csv(registry_path)
    required = {
        'pair_id',
        'proteinA_id',
        'proteinB_id',
        'model_key',
        'weighted_embedding_A_path',
        'weighted_embedding_B_path',
        'metadata_path',
        'status',
        'feature_type',
        'rd_inference_mode',
        'recursive_chain',
        'chain_depth',
        'step_sequence',
        'last_step',
        'rd_reference_implementation',
        'weight_semantics',
        'weight_application',
        'aggregation',
        'double_weighting',
        'target_representation_semantics',
        'validation_status',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Registry missing required columns {sorted(missing)}: {registry_path}')

    summary = {}
    for scale in SCALES_IN_CHAIN_ORDER:
        rows = df[(df['model_key'].astype(str) == scale) & (df['status'].astype(str).isin(['success', 'cached']))].copy()
        if len(rows) != expected_pairs:
            raise ValueError(f'Registry row count mismatch for {scale}: expected {expected_pairs}, got {len(rows)}')
        if not (rows['feature_type'].astype(str) == FEATURE_TYPE).all():
            raise ValueError(f'feature_type mismatch for {scale}')
        if not (rows['rd_inference_mode'].astype(str) == RD_INFERENCE_MODE).all():
            raise ValueError(f'rd_inference_mode mismatch for {scale}')
        if not (rows['weight_semantics'].astype(str) == WEIGHT_SEMANTICS).all():
            raise ValueError(f'weight_semantics mismatch for {scale}')
        recursive_values = rows['recursive_chain'].astype(str).str.lower()
        if not recursive_values.isin({'true', '1'}).all():
            raise ValueError(f'recursive_chain must be true for {scale}')
        double_values = rows['double_weighting'].astype(str).str.lower()
        if not double_values.isin({'false', '0'}).all():
            raise ValueError(f'double_weighting must be false for {scale}')
        if not (rows['rd_reference_implementation'].astype(str) == RD_REFERENCE_IMPLEMENTATION).all():
            raise ValueError(f'rd_reference_implementation mismatch for {scale}')
        expected_semantics = 'weighted_residue_sum_pair_embedding' if scale == '8M' else RD_TARGET_SEMANTICS
        semantics = set(rows['target_representation_semantics'].astype(str).tolist())
        if semantics != {expected_semantics}:
            raise ValueError(f'target_representation_semantics mismatch for {scale}: {sorted(semantics)}')
        expected_depth = SCALES_IN_CHAIN_ORDER.index(scale)
        depths = set(rows['chain_depth'].astype(int).tolist())
        if depths != {expected_depth}:
            raise ValueError(f'chain_depth mismatch for {scale}: {sorted(depths)} expected {expected_depth}')
        for row in rows.itertuples(index=False):
            validate_step_sequence_fields(
                scale=scale,
                step_sequence=getattr(row, 'step_sequence'),
                chain_depth=getattr(row, 'chain_depth'),
                last_step=getattr(row, 'last_step'),
                context=f'{registry_path}:pair={getattr(row, "pair_id")} scale={scale}',
            )
        summary[scale] = {
            'rows': int(len(rows)),
            'unique_pairs': int(rows['pair_id'].nunique()),
            'chain_depth': int(expected_depth),
            'step_sequence': step_sequence_to_text(scale),
            'status_counts': rows['status'].astype(str).value_counts().to_dict(),
        }
    return summary


def validate_registry_shapes(registry_path: Path) -> dict:
    df = pd.read_csv(registry_path)
    summary = {}
    for scale in SCALES_IN_CHAIN_ORDER:
        sdf = df[(df['model_key'].astype(str) == scale) & (df['status'].astype(str).isin(['success', 'cached']))].copy()
        expected_dim = get_embedding_spec(scale).esm_dim
        dims_a = []
        dims_b = []
        for row in sdf.itertuples(index=False):
            a = np.load(row.weighted_embedding_A_path)
            b = np.load(row.weighted_embedding_B_path)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'NaN/inf detected in scale={scale} pair={row.pair_id}')
            dims_a.append(int(a.shape[0]))
            dims_b.append(int(b.shape[0]))
        if not dims_a or set(dims_a) != {expected_dim} or set(dims_b) != {expected_dim}:
            raise ValueError(f'Unexpected dims in scale={scale}: A={sorted(set(dims_a))} B={sorted(set(dims_b))}')
        summary[scale] = {
            'pair_count': int(len(sdf)),
            'expected_dim': expected_dim,
            'dims_A': sorted(set(dims_a)),
            'dims_B': sorted(set(dims_b)),
        }
    return summary


def validate_registry_loader_acceptance(registry_path: Path, expected_pairs: int) -> dict:
    validation = {'registry_path': str(registry_path), 'expected_pairs': int(expected_pairs), 'per_scale': {}}
    for scale in SCALES_IN_CHAIN_ORDER:
        spec = get_embedding_spec(scale)
        weighted_map = _load_weighted_pair_embedding_map(spec, registry_path)
        pair_count = len(weighted_map)
        if pair_count != expected_pairs:
            raise ValueError(
                f'Loader acceptance failed for scale={scale}: expected {expected_pairs} pairs, got {pair_count}'
            )
        sample_pair_id = next(iter(weighted_map))
        record = weighted_map[sample_pair_id]
        a = np.load(record['weighted_embedding_A_path'])
        b = np.load(record['weighted_embedding_B_path'])
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(f'Loader sample arrays must be 1D for scale={scale}')
        if a.shape[0] != spec.esm_dim or b.shape[0] != spec.esm_dim:
            raise ValueError(
                f'Loader sample dim mismatch for scale={scale}: expected {spec.esm_dim}, got {a.shape[0]} and {b.shape[0]}'
            )
        validation['per_scale'][scale] = {
            'pair_count': pair_count,
            'sample_pair_id': sample_pair_id,
            'sample_dim_A': int(a.shape[0]),
            'sample_dim_B': int(b.shape[0]),
        }
    return validation


def _successful_registry_rows(registry_path: Path) -> pd.DataFrame:
    if not registry_path.exists():
        raise FileNotFoundError(f'Registry not found: {registry_path}')
    df = pd.read_csv(registry_path)
    required = {'pair_id', 'model_key', 'weighted_embedding_A_path', 'weighted_embedding_B_path', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Registry missing required diff columns {sorted(missing)}: {registry_path}')
    return df[df['status'].astype(str).isin(['success', 'cached'])].copy()


def _array_diff_row(pair_id: str, scale: str, side: str, recursive_path: str, adjacent_path: str) -> dict:
    recursive = np.load(recursive_path)
    adjacent = np.load(adjacent_path)
    if recursive.shape != adjacent.shape:
        raise ValueError(
            f'Recursive/adjacent shape mismatch pair={pair_id} scale={scale} side={side}: '
            f'{recursive.shape} vs {adjacent.shape}'
        )
    diff = recursive.astype(np.float64) - adjacent.astype(np.float64)
    return {
        'pair_id': pair_id,
        'model_scale': scale,
        'side': side,
        'shape': 'x'.join(str(dim) for dim in recursive.shape),
        'recursive_path': recursive_path,
        'adjacent_path': adjacent_path,
        'allclose_rtol': ALLCLOSE_RTOL,
        'allclose_atol': ALLCLOSE_ATOL,
        'allclose': bool(np.allclose(recursive, adjacent, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL)),
        'max_abs_diff': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'mean_abs_diff': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        'l2_diff': float(np.linalg.norm(diff)) if diff.size else 0.0,
    }


def validate_recursive_vs_adjacent_numeric_diff(
    recursive_registry_path: Path,
    adjacent_registry_path: Path,
    output_root: Path,
) -> dict:
    recursive_df = _successful_registry_rows(recursive_registry_path)
    adjacent_df = _successful_registry_rows(adjacent_registry_path)
    adjacent_by_key = {
        (str(row.model_key), str(row.pair_id)): row
        for row in adjacent_df.itertuples(index=False)
    }
    rows = []
    for rec_row in recursive_df.itertuples(index=False):
        scale = str(rec_row.model_key)
        pair_id = str(rec_row.pair_id)
        key = (scale, pair_id)
        if key not in adjacent_by_key:
            raise ValueError(f'Adjacent registry missing pair={pair_id} scale={scale}: {adjacent_registry_path}')
        adj_row = adjacent_by_key[key]
        rows.append(_array_diff_row(pair_id, scale, 'A', rec_row.weighted_embedding_A_path, adj_row.weighted_embedding_A_path))
        rows.append(_array_diff_row(pair_id, scale, 'B', rec_row.weighted_embedding_B_path, adj_row.weighted_embedding_B_path))

    if not rows:
        raise ValueError(f'No recursive rows were available for recursive vs adjacent diff: {recursive_registry_path}')

    diff_df = pd.DataFrame(rows)
    csv_path = output_root / 'recursive_vs_adjacent_numeric_diff.csv'
    json_path = output_root / 'recursive_vs_adjacent_numeric_diff.json'
    md_path = output_root / 'recursive_vs_adjacent_numeric_diff.md'
    diff_df.to_csv(csv_path, index=False)

    summary = {}
    for scale in SCALES_IN_CHAIN_ORDER:
        scale_df = diff_df[diff_df['model_scale'].astype(str) == scale]
        if scale_df.empty:
            continue
        summary[scale] = {
            'comparisons': int(len(scale_df)),
            'pair_count': int(scale_df['pair_id'].nunique()),
            'allclose_count': int(scale_df['allclose'].astype(bool).sum()),
            'max_abs_diff_max': float(scale_df['max_abs_diff'].max()),
            'mean_abs_diff_max': float(scale_df['mean_abs_diff'].max()),
        }

    payload = {
        'recursive_registry_path': str(recursive_registry_path),
        'adjacent_registry_path': str(adjacent_registry_path),
        'csv_path': str(csv_path),
        'json_path': str(json_path),
        'md_path': str(md_path),
        'allclose_rtol': ALLCLOSE_RTOL,
        'allclose_atol': ALLCLOSE_ATOL,
        'non_allclose_required_scales': list(NON_ALLCLOSE_SCALES),
        'summary': summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

    lines = [
        '# Recursive vs Adjacent Numeric Diff',
        '',
        f'- recursive_registry_path: `{recursive_registry_path}`',
        f'- adjacent_registry_path: `{adjacent_registry_path}`',
        f'- csv_path: `{csv_path}`',
        f'- allclose tolerance: `rtol={ALLCLOSE_RTOL}, atol={ALLCLOSE_ATOL}`',
        f'- non_allclose_required_scales: `{", ".join(NON_ALLCLOSE_SCALES)}`',
        '',
        '| Model scale | Comparisons | Pairs | Allclose count | Max abs diff | Max mean abs diff |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for scale in SCALES_IN_CHAIN_ORDER:
        info = summary.get(scale)
        if not info:
            continue
        lines.append(
            f'| {scale} | {info["comparisons"]} | {info["pair_count"]} | {info["allclose_count"]} | '
            f'{info["max_abs_diff_max"]:.6g} | {info["mean_abs_diff_max"]:.6g} |'
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    offenders = diff_df[
        diff_df['model_scale'].astype(str).isin(NON_ALLCLOSE_SCALES)
        & diff_df['allclose'].astype(bool)
    ]
    if not offenders.empty:
        sample = offenders[['pair_id', 'model_scale', 'side', 'max_abs_diff', 'mean_abs_diff']].head(10).to_dict(orient='records')
        raise ValueError(f'Recursive vs adjacent allclose violation for {NON_ALLCLOSE_SCALES}: {sample}')
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build recursive-chain RD pair assets using fixed residue softmax weights.'
    )
    parser.add_argument('--output_root', type=Path, default=None)
    parser.add_argument('--weight_index_csv', type=Path, default=CANONICAL_WEIGHT_INDEX_CSV)
    parser.add_argument('--rd_chain_root', type=Path, default=CANONICAL_RD_CHAIN_ROOT)
    parser.add_argument('--adjacent_registry_path', type=Path, default=DEFAULT_ADJACENT_REGISTRY)
    parser.add_argument('--limit_pairs', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root()
    builder = RecursiveChainRDPairFeatureBuilder(
        output_root=output_root,
        weight_index_csv=args.weight_index_csv,
        rd_chain_root=args.rd_chain_root,
        limit_pairs=args.limit_pairs,
        verbose=args.verbose,
    )
    bundle = builder.build()

    semantics_summary = validate_registry_semantics(bundle.combined_registry_path, len(bundle.selected_pairs))
    shape_summary = validate_registry_shapes(bundle.combined_registry_path)
    loader_validation = validate_registry_loader_acceptance(
        registry_path=bundle.combined_registry_path,
        expected_pairs=len(bundle.selected_pairs),
    )
    diff_validation = validate_recursive_vs_adjacent_numeric_diff(
        recursive_registry_path=bundle.combined_registry_path,
        adjacent_registry_path=args.adjacent_registry_path,
        output_root=bundle.output_root,
    )
    loader_validation['semantics_summary'] = semantics_summary
    loader_validation['shape_summary'] = shape_summary
    loader_validation['recursive_vs_adjacent_numeric_diff'] = diff_validation
    with bundle.loader_validation_path.open('w') as f:
        json.dump(loader_validation, f, indent=2, sort_keys=True)

    print('recursive-chain build complete')
    print(f'output_root: {bundle.output_root}')
    print(f'pair_count: {len(bundle.selected_pairs)}')
    print(f'combined_registry: {bundle.combined_registry_path}')
    print(f'loader_validation: {bundle.loader_validation_path}')
    print(f'recursive_vs_adjacent_numeric_diff: {bundle.output_root / "recursive_vs_adjacent_numeric_diff.csv"}')
    for scale, info in shape_summary.items():
        print(
            f'{scale}: pairs={info["pair_count"]} dim={info["expected_dim"]} '
            f'dims_A={info["dims_A"]} dims_B={info["dims_B"]}'
        )


if __name__ == '__main__':
    main()
