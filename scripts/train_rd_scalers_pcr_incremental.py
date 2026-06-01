#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
SUPPORT_ROOT = SCRIPTS_ROOT / 'support'
for candidate in (str(PROJECT_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(SUPPORT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import train_rd_scalers as base  # noqa: E402
from audit_rd_inputs import build_protein_audit, read_runnable_pair_df  # noqa: E402

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / 'outputs'
    / 'rd_scalers'
    / 'trainpair_overlap_pcr_incremental'
)


def load_existing_records(output_root: Path) -> List[Dict[str, object]]:
    json_path = output_root / 'trainpair_scaler_registry.json'
    csv_path = output_root / 'trainpair_scaler_registry.csv'
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
    if csv_path.exists():
        with csv_path.open('r', newline='', encoding='utf-8') as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return []


def merge_records_by_fold(records: List[Dict[str, object]], new_record: Dict[str, object]) -> List[Dict[str, object]]:
    fold = int(new_record['fold'])
    by_fold: Dict[int, Dict[str, object]] = {}
    for row in records:
        if 'fold' not in row or row.get('fold') in (None, ''):
            continue
        by_fold[int(row['fold'])] = dict(row)
    old = by_fold.get(fold)
    if old and str(new_record.get('status')) == 'skipped_completed' and str(old.get('status')) == 'success':
        merged = dict(old)
        for key, value in new_record.items():
            if key in {'status', 'returncode', 'message', 'start_time_utc', 'finish_time_utc', 'command'}:
                continue
            merged[key] = value
        by_fold[fold] = merged
    else:
        merged = dict(old or {})
        merged.update(new_record)
        by_fold[fold] = merged
    return [by_fold[k] for k in sorted(by_fold)]


def main() -> int:
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    args = base.parse_args()
    args.regressor_type = 'pcr'
    args.pca_type = 'incremental'
    args.output_root = Path(args.output_root)

    pair_df = read_runnable_pair_df()
    _, sequence_map = build_protein_audit(pair_df)
    records: List[Dict[str, object]] = load_existing_records(args.output_root)
    for fold in base.parse_folds(args.folds):
        plan = base.build_fold_plan(args, fold, pair_df, sequence_map)
        if args.plan_only:
            record = {
                'fold': fold,
                'status': 'planned',
                'scaler_dir': str(args.output_root / f'fold{fold}' / 'scalers'),
                'regressor_type': 'pcr',
                'pca_type': 'incremental',
                'pcr_incremental': True,
                **{k: plan[k] for k in (
                    'rd_fit_pair_count', 'rd_fit_residue_row_count', 'rd_fit_pair_allowlist_csv',
                    'rd_fit_pair_allowlist_sha256', 'rd_candidate_pair_count_before_cdhit',
                    'cdhit_representative_protein_count', 'strict_valid_test_protein_exclusion',
                    'rd_fit_pair_selection', 'pair_overlap_allowed_for_rd_fit', 'valid_pair_overlap_count',
                    'test_pair_overlap_count', 'train_valid_test_pair_overlap_count_allowed',
                )},
            }
        else:
            record = base.train_fold_scalers(args, fold, plan)
            base.annotate_scaler_metadata(args.output_root, fold, plan)
            record.update({
                'rd_candidate_pair_count_before_cdhit': plan['rd_candidate_pair_count_before_cdhit'],
                'cdhit_representative_protein_count': plan['cdhit_representative_protein_count'],
                'strict_valid_test_protein_exclusion': False,
                'pair_level_valid_test_overlap_exclusion': False,
                'pair_overlap_allowed_for_rd_fit': True,
                'valid_pair_overlap_count': plan['valid_pair_overlap_count'],
                'test_pair_overlap_count': plan['test_pair_overlap_count'],
                'rd_fit_pair_selection': base.SELECTION_LABEL,
                'train_valid_test_pair_overlap_count_allowed': plan['train_valid_test_pair_overlap_count_allowed'],
                'regressor_type': 'pcr',
                'pca_type': 'incremental',
                'pcr_incremental': True,
            })
        records = merge_records_by_fold(records, record)
        base.write_outputs(args.output_root, records)
        note = [
            '# PCR-Incremental RD-only CD-HIT Train-Pair Overlap-Allowed Scaler Launcher',
            '',
            f'- output_root: `{args.output_root}`',
            '- regressor_type: `pcr`',
            '- pca_type: `incremental`',
            '- johnstone_threshold_active: `true` when present in scaler metadata',
            '- cdhit_scope: `rd_scaler_fit_only`',
            '- ppimi_train_filtering: `false`',
            '- pair_overlap_allowed_for_rd_fit: `true`',
            '- source_linear_baseline_root: `outputs/rd_scalers/trainpair_overlap`',
        ]
        (args.output_root / 'PCR_INCREMENTAL_LAUNCHER_NOTE.md').write_text('\n'.join(note) + '\n', encoding='utf-8')
        if record.get('status') == 'failed':
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
