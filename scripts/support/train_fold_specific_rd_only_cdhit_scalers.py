#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
for candidate in (str(PROJECT_ROOT), str(SRC_ROOT), str(PROJECT_ROOT / 'scripts')):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from audit_rd_inputs import (  # noqa: E402
    build_official_split,
    build_protein_audit,
    endpoint_protein_sets,
    induce_sampled_train_pairs,
    induce_sidewise_sampled_train_pairs,
    read_runnable_pair_df,
    run_cdhit,
    write_fasta,
)
from support.run_s4_trainonly_sampling_compare_rd_fixed_weighted_35M import (  # noqa: E402
    plan_pair_df,
    unique_train_proteins,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'rd_scalers' / 'fold_specific_sidewise_intersection'
DEFAULT_PYTHON = Path('python')
DEFAULT_WEIGHT_INDEX = PROJECT_ROOT / 'fixed_residue_softmax_weights' / 't0p7' / 'fixed_residue_softmax_weight_index.csv'
TRANSITIONS = ('8M->35M', '35M->150M', '150M->650M', '650M->3B')


def now_utc() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_folds(values: Sequence[str]) -> List[int]:
    if not values or any(str(v).lower() == 'all' for v in values):
        return [1, 2, 3, 4, 5]
    folds = []
    for value in values:
        fold = int(value)
        if fold < 1:
            raise ValueError(f'fold must be >= 1, got {fold}')
        folds.append(fold)
    return list(dict.fromkeys(folds))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_allowlist_csv(path: Path, pair_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['pair_id', 'proteinA_id', 'proteinB_id', 'length_A', 'length_B', 'residue_rows']
    pair_df[columns].to_csv(path, index=False)


def scaler_paths_complete(scaler_dir: Path) -> bool:
    if not (scaler_dir / 'metadata.json').exists():
        return False
    names = [p.name for p in scaler_dir.glob('rd-scaler-*.npz')]
    return all(any(transition.replace('->', '-') in name for name in names) for transition in TRANSITIONS)


def existing_scaler_pair_mode(scaler_dir: Path) -> str:
    metadata_path = scaler_dir / 'metadata.json'
    if not metadata_path.exists():
        return ''
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return str(metadata.get('cdhit_pair_mode', 'pooled'))


def build_fold_plan(args: argparse.Namespace, fold: int, pair_df: pd.DataFrame, sequence_map: Dict[str, str]) -> Dict[str, object]:
    fold_root = args.output_root / f'fold{fold}'
    fold_root.mkdir(parents=True, exist_ok=True)
    split = build_official_split(args.eval_setting, fold, pair_df)
    train_pair_df = plan_pair_df(pair_df, split.mapped_train_pairs)
    valid_pair_df = plan_pair_df(pair_df, split.mapped_valid_pairs)
    test_pair_df = plan_pair_df(pair_df, split.mapped_test_pairs)
    train_proteins = set(unique_train_proteins(train_pair_df))
    valid_test_proteins = set(unique_train_proteins(valid_pair_df)) | set(unique_train_proteins(test_pair_df))
    rd_candidate_proteins = sorted(train_proteins - valid_test_proteins)
    rd_candidate_pair_df = train_pair_df[
        train_pair_df['proteinA_id'].isin(rd_candidate_proteins)
        & train_pair_df['proteinB_id'].isin(rd_candidate_proteins)
    ].copy()
    if rd_candidate_pair_df.empty:
        raise RuntimeError(f'fold{fold} has zero strict train-only RD candidate pairs after excluding valid/test proteins')

    threshold_label = str(args.cdhit_threshold).replace('.', 'p')
    cluster_csv = fold_root / 'rd_fit_cdhit_clusters.csv'
    if args.cdhit_pair_mode == 'sidewise_intersection':
        candidate_proteins_a, candidate_proteins_b = endpoint_protein_sets(rd_candidate_pair_df)
        fasta_path_a = fold_root / 'rd_fit_train_only_no_valid_test_proteins_A.fasta'
        fasta_path_b = fold_root / 'rd_fit_train_only_no_valid_test_proteins_B.fasta'
        cdhit_prefix_a = fold_root / f'rd_fit_train_only_no_valid_test_A_cdhit_c{threshold_label}'
        cdhit_prefix_b = fold_root / f'rd_fit_train_only_no_valid_test_B_cdhit_c{threshold_label}'
        write_fasta(candidate_proteins_a, sequence_map, fasta_path_a)
        write_fasta(candidate_proteins_b, sequence_map, fasta_path_b)
        representatives_a, cluster_df_a = run_cdhit(fasta_path_a, cdhit_prefix_a, args.cdhit_threshold)
        representatives_b, cluster_df_b = run_cdhit(fasta_path_b, cdhit_prefix_b, args.cdhit_threshold)
        pd.concat(
            [
                cluster_df_a.assign(endpoint='A'),
                cluster_df_b.assign(endpoint='B'),
            ],
            ignore_index=True,
        ).to_csv(cluster_csv, index=False)
        pairs_a, pairs_b, sampled_pairs = induce_sidewise_sampled_train_pairs(
            rd_candidate_pair_df,
            set(representatives_a),
            set(representatives_b),
        )
        representatives = sorted(set(representatives_a) | set(representatives_b))
        selection_label = 'official_train_fold_strict_sidewise_cdhit_pair_intersection'
        cdhit_artifacts = {
            'rd_fit_train_fasta_A': str(fasta_path_a),
            'rd_fit_train_fasta_B': str(fasta_path_b),
            'rd_fit_cdhit_output_A': str(cdhit_prefix_a),
            'rd_fit_cdhit_output_B': str(cdhit_prefix_b),
            'cdhit_representative_proteins_A': sorted(representatives_a),
            'cdhit_representative_proteins_B': sorted(representatives_b),
            'cdhit_representative_protein_count_A': int(len(representatives_a)),
            'cdhit_representative_protein_count_B': int(len(representatives_b)),
            'rd_fit_pair_count_A': int(len(pairs_a)),
            'rd_fit_pair_count_B': int(len(pairs_b)),
        }
    else:
        fasta_path = fold_root / 'rd_fit_train_only_no_valid_test_proteins.fasta'
        cdhit_prefix = fold_root / f'rd_fit_train_only_no_valid_test_cdhit_c{threshold_label}'
        write_fasta(rd_candidate_proteins, sequence_map, fasta_path)
        representatives, cluster_df = run_cdhit(fasta_path, cdhit_prefix, args.cdhit_threshold)
        cluster_df.assign(endpoint='pooled').to_csv(cluster_csv, index=False)
        sampled_pairs = induce_sampled_train_pairs(rd_candidate_pair_df, set(representatives))
        selection_label = 'official_train_fold_strict_pooled_cdhit_representative_pairs'
        cdhit_artifacts = {
            'rd_fit_train_fasta': str(fasta_path),
            'rd_fit_cdhit_output': str(cdhit_prefix),
            'cdhit_representative_proteins': sorted(representatives),
        }

    if not sampled_pairs:
        raise RuntimeError(f'CD-HIT fold{fold} selected zero RD fit pairs')
    valid_overlap = sorted(set(sampled_pairs) & set(split.mapped_valid_pairs))
    test_overlap = sorted(set(sampled_pairs) & set(split.mapped_test_pairs))
    if valid_overlap or test_overlap:
        raise RuntimeError(
            f'RD-only CD-HIT fold{fold} selected validation/test pairs: '
            f'valid={valid_overlap[:10]} test={test_overlap[:10]}'
        )

    sampled_pair_df = plan_pair_df(pair_df, sampled_pairs)
    allowlist_csv = fold_root / 'rd_fit_pair_allowlist.csv'
    write_allowlist_csv(allowlist_csv, sampled_pair_df)

    plan = {
        'created_at_utc': now_utc(),
        'eval_setting': args.eval_setting,
        'fold': int(fold),
        'cdhit_threshold': float(args.cdhit_threshold),
        'cdhit_pair_mode': args.cdhit_pair_mode,
        'cdhit_scope': 'rd_scaler_fit_only',
        'ppimi_train_filtering': False,
        'official_valid_test_used_for_rd_fit': False,
        'rd_fit_pair_selection': selection_label,
        'original_train_pair_count': int(len(split.mapped_train_pairs)),
        'original_valid_pair_count': int(len(split.mapped_valid_pairs)),
        'original_test_pair_count': int(len(split.mapped_test_pairs)),
        'train_protein_count_before_excluding_valid_test': int(len(train_proteins)),
        'valid_test_protein_count_excluded_from_rd_fit': int(len(valid_test_proteins)),
        'rd_candidate_protein_count_before_cdhit': int(len(rd_candidate_proteins)),
        'rd_candidate_pair_count_before_cdhit': int(len(rd_candidate_pair_df)),
        'cdhit_representative_protein_count': int(len(representatives)),
        'rd_fit_pair_count': int(len(sampled_pairs)),
        'rd_fit_residue_row_count': int(sampled_pair_df['residue_rows'].sum()),
        'valid_pair_overlap_count': 0,
        'test_pair_overlap_count': 0,
        'rd_fit_pair_allowlist_csv': str(allowlist_csv),
        'rd_fit_pair_allowlist_sha256': sha256_file(allowlist_csv),
        'strict_valid_test_protein_exclusion': True,
        'rd_candidate_proteins': rd_candidate_proteins,
        'excluded_valid_test_proteins': sorted(valid_test_proteins),
        'rd_fit_cdhit_clusters_csv': str(cluster_csv),
        'sampled_pair_ids': sampled_pairs,
        **cdhit_artifacts,
    }
    write_json(fold_root / 'rd_only_cdhit_plan.json', plan)
    return plan


def train_fold_scalers(args: argparse.Namespace, fold: int, plan: Dict[str, object]) -> Dict[str, object]:
    fold_root = args.output_root / f'fold{fold}'
    scaler_dir = fold_root / 'scalers'
    log_path = fold_root / 'train_rd_scalers.log'
    if args.resume and scaler_paths_complete(scaler_dir) and not args.restart:
        existing_mode = existing_scaler_pair_mode(scaler_dir)
        if existing_mode == plan['cdhit_pair_mode']:
            return {
                'fold': fold,
                'status': 'skipped_completed',
                'scaler_dir': str(scaler_dir),
                'log_path': str(log_path),
                'message': 'existing complete scaler set found',
                'cdhit_pair_mode': plan['cdhit_pair_mode'],
                'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
                **{k: plan[k] for k in ('rd_fit_pair_count', 'rd_fit_residue_row_count', 'rd_fit_pair_allowlist_csv', 'rd_fit_pair_allowlist_sha256')},
            }
        raise RuntimeError(
            f'Existing scaler pair mode is {existing_mode!r}, but the requested mode is '
            f'{plan["cdhit_pair_mode"]!r}; use a new output root or pass --restart'
        )
    if args.restart and scaler_dir.exists():
        import shutil
        shutil.rmtree(scaler_dir)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        '-u',
        'scripts/train_rd_from_fixed_weighted_residue_assets.py',
        '--output_dir', str(scaler_dir),
        '--weight_index_csv', str(args.weight_index_csv),
        '--rd_project_root', str(args.rd_project_root),
        '--regressor_type', args.regressor_type,
        '--pca_type', args.pca_type,
        '--pca_batch_size', str(args.pca_batch_size),
        '--pair_allowlist_csv', str(plan['rd_fit_pair_allowlist_csv']),
        '--semantic_mode', 'fixed_weighted_residue_level_rd_only_cdhit',
        '--cdhit_scope', 'rd_scaler_fit_only',
        '--ppimi_train_filtering', 'false',
    ]
    if args.max_transitions is not None:
        cmd.extend(['--max_transitions', str(args.max_transitions)])
    if args.limit_pairs is not None:
        cmd.extend(['--limit_pairs', str(args.limit_pairs)])
    if args.verbose:
        cmd.append('--verbose')

    if args.plan_only:
        return {
            'fold': fold,
            'status': 'planned',
            'scaler_dir': str(scaler_dir),
            'log_path': str(log_path),
            'command': ' '.join(cmd),
            'cdhit_pair_mode': plan['cdhit_pair_mode'],
            'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
            **{k: plan[k] for k in ('rd_fit_pair_count', 'rd_fit_residue_row_count', 'rd_fit_pair_allowlist_csv', 'rd_fit_pair_allowlist_sha256')},
        }

    start = now_utc()
    with log_path.open('w', encoding='utf-8') as log_file:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    finish = now_utc()
    metadata_path = scaler_dir / 'metadata.json'
    status = 'success' if proc.returncode == 0 and scaler_paths_complete(scaler_dir) else 'failed'
    record = {
        'fold': fold,
        'status': status,
        'returncode': proc.returncode,
        'start_time_utc': start,
        'finish_time_utc': finish,
        'scaler_dir': str(scaler_dir),
        'metadata_path': str(metadata_path),
        'log_path': str(log_path),
        'command': ' '.join(cmd),
        'regressor_type': args.regressor_type,
        'pca_type': args.pca_type,
        'pcr_incremental': bool(args.regressor_type == 'pcr' and args.pca_type == 'incremental'),
        'cdhit_pair_mode': plan['cdhit_pair_mode'],
        'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
        **{k: plan[k] for k in ('rd_fit_pair_count', 'rd_fit_residue_row_count', 'rd_fit_pair_allowlist_csv', 'rd_fit_pair_allowlist_sha256')},
    }
    if status != 'success':
        record['message'] = f'RD scaler training failed; inspect {log_path}'
        return record
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('cdhit_scope') != 'rd_scaler_fit_only' or metadata.get('ppimi_train_filtering') != 'false':
        raise RuntimeError(f'RD scaler metadata guard failed for fold{fold}: {metadata_path}')
    metadata.update({
        'cdhit_pair_mode': plan['cdhit_pair_mode'],
        'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
        'rd_fit_plan_json': str(fold_root / 'rd_only_cdhit_plan.json'),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    record['message'] = 'RD-only CD-HIT fold-specific scalers trained successfully'
    return record


def write_campaign_outputs(output_root: Path, records: List[Dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / 'run_registry.json').open('w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
    fieldnames = sorted({key for row in records for key in row.keys()})
    with (output_root / 'run_registry.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    pair_mode = records[-1].get('cdhit_pair_mode', 'unknown') if records else 'unknown'
    selection_label = records[-1].get('rd_fit_pair_selection', 'unknown') if records else 'unknown'
    lines = [
        '# Fold-Specific RD-only CD-HIT Scaler Campaign',
        '',
        f'- updated_at_utc: `{now_utc()}`',
        f'- output_root: `{output_root}`',
        '- cdhit_scope: `rd_scaler_fit_only`',
        '- ppimi_train_filtering: `false`',
        f'- cdhit_pair_mode: `{pair_mode}`',
        f'- rd_fit_pair_selection: `{selection_label}`',
        '- downstream PPIMI must not use `--train_pair_allowlist_csv`.',
        '',
        '| Fold | Status | RD Fit Pairs | RD Fit Residue Rows | Scaler Dir |',
        '| ---: | --- | ---: | ---: | --- |',
    ]
    for row in records:
        lines.append(
            f"| {row.get('fold')} | {row.get('status')} | {row.get('rd_fit_pair_count', '-')} | "
            f"{row.get('rd_fit_residue_row_count', '-')} | `{row.get('scaler_dir', '')}` |"
        )
    (output_root / 'run_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train fold-specific RD-only CD-HIT scalers without filtering PPIMI downstream data.')
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--eval_setting', type=str, default='S4')
    parser.add_argument('--folds', nargs='*', default=['all'])
    parser.add_argument('--cdhit_threshold', type=float, default=0.5)
    parser.add_argument(
        '--cdhit_pair_mode',
        type=str,
        default='sidewise_intersection',
        choices=['sidewise_intersection', 'pooled'],
    )
    parser.add_argument('--weight_index_csv', type=Path, default=DEFAULT_WEIGHT_INDEX)
    parser.add_argument('--rd_project_root', type=Path, default=Path('external/reverse_distillation'))
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--regressor_type', type=str, default='linear', choices=['linear', 'ridge', 'pcr'])
    parser.add_argument('--pca_type', type=str, default='incremental', choices=['incremental', 'fbpca'])
    parser.add_argument('--pca_batch_size', type=int, default=131072)
    parser.add_argument('--max_transitions', type=int, default=None)
    parser.add_argument('--limit_pairs', type=int, default=None)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = parse_folds(args.folds)
    pair_df = read_runnable_pair_df()
    _, sequence_map = build_protein_audit(pair_df)
    records = []
    for fold in folds:
        plan = build_fold_plan(args, fold, pair_df, sequence_map)
        record = train_fold_scalers(args, fold, plan)
        records.append(record)
        write_campaign_outputs(args.output_root, records)
        if record.get('status') == 'failed':
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
