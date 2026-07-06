#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
SUPPORT_ROOT = SCRIPTS_ROOT / 'support'
for candidate in (str(PROJECT_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(SUPPORT_ROOT)):
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
from support.train_fold_specific_rd_only_cdhit_scalers import (  # noqa: E402
    DEFAULT_PYTHON,
    DEFAULT_WEIGHT_INDEX,
    TRANSITIONS,
    now_utc,
    parse_folds,
    sha256_file,
    train_fold_scalers,
    write_allowlist_csv,
    write_json,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'rd_scalers' / 'trainpair_sidewise_intersection'
DEFAULT_RD_PROJECT_ROOT = Path('external/reverse_distillation')
POOLED_SELECTION_LABEL = 'official_train_fold_pooled_cdhit_representative_pairs_overlap_allowed'
SIDEWISE_SELECTION_LABEL = 'official_train_fold_sidewise_cdhit_pair_intersection_overlap_allowed'


def build_fold_plan(args: argparse.Namespace, fold: int, pair_df: pd.DataFrame, sequence_map: Dict[str, str]) -> Dict[str, object]:
    fold_root = args.output_root / f'fold{fold}'
    fold_root.mkdir(parents=True, exist_ok=True)
    split = build_official_split(args.eval_setting, fold, pair_df)
    train_pair_df = plan_pair_df(pair_df, split.mapped_train_pairs)
    valid_pair_df = plan_pair_df(pair_df, split.mapped_valid_pairs)
    test_pair_df = plan_pair_df(pair_df, split.mapped_test_pairs)

    train_proteins_all = sorted(set(unique_train_proteins(train_pair_df)))
    valid_test_proteins = sorted(set(unique_train_proteins(valid_pair_df)) | set(unique_train_proteins(test_pair_df)))
    valid_test_pair_ids = set(split.mapped_valid_pairs) | set(split.mapped_test_pairs)
    rd_candidate_pair_df = train_pair_df.copy()
    if rd_candidate_pair_df.empty:
        raise RuntimeError(f'fold{fold} has zero RD candidate train pairs')
    train_proteins = sorted(set(unique_train_proteins(rd_candidate_pair_df)))
    if not train_proteins:
        raise RuntimeError(f'fold{fold} has zero RD candidate train proteins')

    threshold_label = str(args.cdhit_threshold).replace('.', 'p')
    cluster_csv = fold_root / 'rd_fit_cdhit_clusters.csv'
    if args.cdhit_pair_mode == 'sidewise_intersection':
        train_proteins_a, train_proteins_b = endpoint_protein_sets(rd_candidate_pair_df)
        fasta_path_a = fold_root / 'rd_fit_official_train_pairlevel_proteins_A.fasta'
        fasta_path_b = fold_root / 'rd_fit_official_train_pairlevel_proteins_B.fasta'
        cdhit_prefix_a = fold_root / f'rd_fit_official_train_pairlevel_A_cdhit_c{threshold_label}'
        cdhit_prefix_b = fold_root / f'rd_fit_official_train_pairlevel_B_cdhit_c{threshold_label}'
        write_fasta(train_proteins_a, sequence_map, fasta_path_a)
        write_fasta(train_proteins_b, sequence_map, fasta_path_b)
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
        selection_label = SIDEWISE_SELECTION_LABEL
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
        fasta_path = fold_root / 'rd_fit_official_train_pairlevel_proteins.fasta'
        cdhit_prefix = fold_root / f'rd_fit_official_train_pairlevel_cdhit_c{threshold_label}'
        write_fasta(train_proteins, sequence_map, fasta_path)
        representatives, cluster_df = run_cdhit(fasta_path, cdhit_prefix, args.cdhit_threshold)
        cluster_df.assign(endpoint='pooled').to_csv(cluster_csv, index=False)
        sampled_pairs = induce_sampled_train_pairs(rd_candidate_pair_df, set(representatives))
        selection_label = POOLED_SELECTION_LABEL
        cdhit_artifacts = {
            'rd_fit_train_fasta': str(fasta_path),
            'rd_fit_cdhit_output': str(cdhit_prefix),
            'cdhit_representative_proteins': sorted(representatives),
        }

    if not sampled_pairs:
        raise RuntimeError(f'CD-HIT fold{fold} selected zero RD fit train pairs')
    valid_overlap = sorted(set(sampled_pairs) & set(split.mapped_valid_pairs))
    test_overlap = sorted(set(sampled_pairs) & set(split.mapped_test_pairs))

    sampled_pair_df = plan_pair_df(pair_df, sampled_pairs)
    allowlist_csv = fold_root / 'rd_fit_pair_allowlist.csv'
    write_allowlist_csv(allowlist_csv, sampled_pair_df)

    shared_proteins = sorted(set(train_proteins) & set(valid_test_proteins))
    plan = {
        'created_at_utc': now_utc(),
        'eval_setting': args.eval_setting,
        'fold': int(fold),
        'cdhit_threshold': float(args.cdhit_threshold),
        'cdhit_pair_mode': args.cdhit_pair_mode,
        'cdhit_scope': 'rd_scaler_fit_only',
        'ppimi_train_filtering': False,
        'official_valid_test_used_for_rd_fit': False,
        'pair_overlap_allowed_for_rd_fit': True,
        'rd_fit_pair_selection': selection_label,
        'original_train_pair_count': int(len(split.mapped_train_pairs)),
        'original_valid_pair_count': int(len(split.mapped_valid_pairs)),
        'original_test_pair_count': int(len(split.mapped_test_pairs)),
        'train_protein_count_before_cdhit': int(len(train_proteins)),
        'train_protein_count_before_pair_overlap_exclusion': int(len(train_proteins_all)),
        'train_valid_test_pair_overlap_count_allowed': int(len(set(train_pair_df['pair_id']) & valid_test_pair_ids)),
        'valid_test_protein_count_not_excluded': int(len(valid_test_proteins)),
        'train_valid_test_shared_protein_count': int(len(shared_proteins)),
        'rd_candidate_protein_count_before_cdhit': int(len(train_proteins)),
        'rd_candidate_pair_count_before_cdhit': int(len(rd_candidate_pair_df)),
        'cdhit_representative_protein_count': int(len(representatives)),
        'rd_fit_pair_count': int(len(sampled_pairs)),
        'rd_fit_residue_row_count': int(sampled_pair_df['residue_rows'].sum()),
        'valid_pair_overlap_count': int(len(valid_overlap)),
        'test_pair_overlap_count': int(len(test_overlap)),
        'rd_fit_pair_allowlist_csv': str(allowlist_csv),
        'rd_fit_pair_allowlist_sha256': sha256_file(allowlist_csv),
        'strict_valid_test_protein_exclusion': False,
        'pair_level_valid_test_overlap_exclusion': False,
        'train_pair_only_cdhit': True,
        'rd_fit_cdhit_clusters_csv': str(cluster_csv),
        'sampled_pair_ids': sampled_pairs,
        'train_proteins': train_proteins,
        'valid_test_proteins_not_excluded': valid_test_proteins,
        'shared_train_valid_test_proteins_allowed_for_rd_fit': shared_proteins,
        **cdhit_artifacts,
    }
    write_json(fold_root / 'rd_only_cdhit_trainpair_plan.json', plan)
    return plan


def annotate_scaler_metadata(output_root: Path, fold: int, plan: Dict[str, object]) -> None:
    metadata_path = output_root / f'fold{fold}' / 'scalers' / 'metadata.json'
    if not metadata_path.exists():
        return
    payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    payload.update({
        'cdhit_pair_mode': plan['cdhit_pair_mode'],
        'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
        'strict_valid_test_protein_exclusion': False,
        'pair_level_valid_test_overlap_exclusion': False,
        'train_pair_only_cdhit': True,
        'official_valid_test_used_for_rd_fit': False,
        'pair_overlap_allowed_for_rd_fit': True,
        'valid_test_pairs_used_for_rd_fit': True,
        'valid_test_shared_proteins_excluded': False,
        'pair_level_valid_test_overlap_exclusion': False,
        'pair_overlap_allowed_for_rd_fit': True,
        'rd_candidate_pair_count_before_cdhit': plan['rd_candidate_pair_count_before_cdhit'],
        'rd_candidate_protein_count_before_cdhit': plan['rd_candidate_protein_count_before_cdhit'],
        'cdhit_representative_protein_count': plan['cdhit_representative_protein_count'],
        'rd_fit_pair_count': plan['rd_fit_pair_count'],
        'rd_fit_residue_row_count': plan['rd_fit_residue_row_count'],
        'valid_pair_overlap_count': plan['valid_pair_overlap_count'],
        'test_pair_overlap_count': plan['test_pair_overlap_count'],
        'train_valid_test_shared_protein_count': plan['train_valid_test_shared_protein_count'],
        'train_valid_test_pair_overlap_count_allowed': plan['train_valid_test_pair_overlap_count_allowed'],
        'pair_level_valid_test_overlap_exclusion': False,
        'pair_overlap_allowed_for_rd_fit': True,
        'rd_fit_plan_json': str(output_root / f'fold{fold}' / 'rd_only_cdhit_trainpair_plan.json'),
    })
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_outputs(output_root: Path, records: List[Dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'trainpair_scaler_registry.json').write_text(json.dumps(records, indent=2, sort_keys=True), encoding='utf-8')
    fieldnames = sorted({key for row in records for key in row.keys()})
    with (output_root / 'trainpair_scaler_registry.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    pair_mode = records[-1].get('cdhit_pair_mode', 'unknown') if records else 'unknown'
    selection_label = records[-1].get('rd_fit_pair_selection', 'unknown') if records else 'unknown'
    lines = [
        '# Train-Pair-Only RD-CDHIT Overlap-Allowed Fold-Specific Scalers',
        '',
        f'- updated_at_utc: `{now_utc()}`',
        f'- output_root: `{output_root}`',
        '- cdhit_scope: `rd_scaler_fit_only`',
        '- ppimi_train_filtering: `false`',
        '- strict_valid_test_protein_exclusion: `false`',
        '- pair_overlap_allowed_for_rd_fit: `true`',
        f'- cdhit_pair_mode: `{pair_mode}`',
        f'- rd_fit_pair_selection: `{selection_label}`',
        '',
        '| Fold | Status | RD fit pairs | RD fit residue rows | Candidate pairs before CD-HIT | CD-HIT reps |',
        '| ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in records:
        lines.append(
            f"| {row.get('fold')} | {row.get('status')} | {row.get('rd_fit_pair_count', '-')} | "
            f"{row.get('rd_fit_residue_row_count', '-')} | {row.get('rd_candidate_pair_count_before_cdhit', '-')} | "
            f"{row.get('cdhit_representative_protein_count', '-')} |"
        )
    (output_root / 'trainpair_scaler_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train fold-specific RD-only CD-HIT scalers from official train-pair CD-HIT representatives.')
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--folds', nargs='*', default=['all'])
    parser.add_argument('--eval_setting', type=str, default='S4')
    parser.add_argument('--cdhit_threshold', type=float, default=0.5)
    parser.add_argument(
        '--cdhit_pair_mode',
        type=str,
        default='sidewise_intersection',
        choices=['sidewise_intersection', 'pooled'],
    )
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--weight_index_csv', type=Path, default=DEFAULT_WEIGHT_INDEX)
    parser.add_argument('--rd_project_root', type=Path, default=DEFAULT_RD_PROJECT_ROOT)
    parser.add_argument('--regressor_type', type=str, default='linear', choices=['linear', 'ridge', 'pcr'])
    parser.add_argument('--pca_type', type=str, default='incremental', choices=['incremental', 'fbpca'])
    parser.add_argument('--pca_batch_size', type=int, default=131072)
    parser.add_argument('--max_transitions', type=int, default=None)
    parser.add_argument('--limit_pairs', type=int, default=None)
    parser.add_argument('--plan_only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_df = read_runnable_pair_df()
    _, sequence_map = build_protein_audit(pair_df)
    records: List[Dict[str, object]] = []
    for fold in parse_folds(args.folds):
        plan = build_fold_plan(args, fold, pair_df, sequence_map)
        if args.plan_only:
            record = {
                'fold': fold,
                'status': 'planned',
                'scaler_dir': str(args.output_root / f'fold{fold}' / 'scalers'),
                **{k: plan[k] for k in (
                    'rd_fit_pair_count', 'rd_fit_residue_row_count', 'rd_fit_pair_allowlist_csv',
                    'rd_fit_pair_allowlist_sha256', 'rd_candidate_pair_count_before_cdhit',
                    'cdhit_representative_protein_count', 'strict_valid_test_protein_exclusion',
                    'rd_fit_pair_selection', 'pair_overlap_allowed_for_rd_fit', 'valid_pair_overlap_count',
                    'test_pair_overlap_count', 'train_valid_test_pair_overlap_count_allowed', 'cdhit_pair_mode',
                )},
            }
        else:
            record = train_fold_scalers(args, fold, plan)
            annotate_scaler_metadata(args.output_root, fold, plan)
            record.update({
                'rd_candidate_pair_count_before_cdhit': plan['rd_candidate_pair_count_before_cdhit'],
                'cdhit_representative_protein_count': plan['cdhit_representative_protein_count'],
                'strict_valid_test_protein_exclusion': False,
                'pair_level_valid_test_overlap_exclusion': False,
                'cdhit_pair_mode': plan['cdhit_pair_mode'],
                'rd_fit_pair_selection': plan['rd_fit_pair_selection'],
                'pair_overlap_allowed_for_rd_fit': True,
                'valid_pair_overlap_count': plan['valid_pair_overlap_count'],
                'test_pair_overlap_count': plan['test_pair_overlap_count'],
                'train_valid_test_pair_overlap_count_allowed': plan['train_valid_test_pair_overlap_count_allowed'],
            })
        records.append(record)
        write_outputs(args.output_root, records)
        if record.get('status') == 'failed':
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
