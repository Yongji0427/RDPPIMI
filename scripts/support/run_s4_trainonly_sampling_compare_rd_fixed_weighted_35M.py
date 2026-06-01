#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / 'scripts'
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from audit_rd_inputs import (  # noqa: E402
    DEFAULT_NUM_RANDOM,
    DEFAULT_SEED_POOL,
    OfficialSplit,
    SamplingPlan,
    build_official_split,
    build_protein_audit,
    choose_random_plans,
    induce_sampled_train_pairs,
    read_runnable_pair_df,
    run_cdhit,
    write_fasta,
)

DEFAULT_WEIGHTED_REGISTRY = (
    PROJECT_ROOT
    / 'outputs'
    / 'rd_pair_assets'
    / 'adjacent_full_step'
    / 'indices'
    / 'fixed_weighted_rd_pair_registry_all_models.csv'
)
DEFAULT_PAIR_MANIFEST = PROJECT_ROOT / 'multippimi_pair_manifest.csv'
DEFAULT_OUTPUT_PARENT = (
    PROJECT_ROOT / 'outputs' / 'sampling_compare'
)


def utc_now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def timestamp_label() -> str:
    return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def build_default_output_root(eval_setting: str, fold: int, cdhit_threshold: float, protein_embedding_model: str) -> Path:
    threshold_label = str(cdhit_threshold).replace('.', 'p')
    return DEFAULT_OUTPUT_PARENT / f'smoke_{eval_setting}_{protein_embedding_model}_fold{fold}_c{threshold_label}_{timestamp_label()}'


def unique_train_proteins(train_pair_df: pd.DataFrame) -> List[str]:
    proteins = sorted(set(train_pair_df['proteinA_id']).union(set(train_pair_df['proteinB_id'])))
    return proteins


def plan_pair_df(pair_df: pd.DataFrame, pair_ids: List[str]) -> pd.DataFrame:
    pair_ids_set = set(pair_ids)
    return pair_df[pair_df['pair_id'].isin(pair_ids_set)].copy()


def train_residue_row_count(pair_df: pd.DataFrame, pair_ids: List[str]) -> int:
    sampled_df = plan_pair_df(pair_df, pair_ids)
    return int(sampled_df['residue_rows'].sum())


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_plan_csv(path: Path, pair_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(path, index=False)


def build_split_audit(split: OfficialSplit, plan: SamplingPlan, pair_df: pd.DataFrame) -> Dict:
    sampled_pair_df = plan_pair_df(pair_df, plan.sampled_train_pairs)
    return {
        'generated_at': utc_now(),
        'eval_setting': split.eval_setting,
        'fold': split.fold,
        'strategy': plan.strategy,
        'plan_name': plan.name,
        'cdhit_threshold': plan.cdhit_threshold,
        'random_seed': plan.random_seed,
        'original_train_pair_count': int(len(split.mapped_train_pairs)),
        'original_valid_pair_count': int(len(split.mapped_valid_pairs)),
        'original_test_pair_count': int(len(split.mapped_test_pairs)),
        'sampled_train_pair_count': int(len(plan.sampled_train_pairs)),
        'sampled_train_protein_count': int(len(plan.sampled_train_proteins)),
        'sampled_train_residue_row_count': int(sampled_pair_df['residue_rows'].sum()),
        'valid_pair_count': int(len(split.mapped_valid_pairs)),
        'test_pair_count': int(len(split.mapped_test_pairs)),
        'official_valid_test_fixed': True,
        'train_pairs_selected_checksum': ';'.join(plan.sampled_train_pairs),
        'unmapped_train_pair_keys': split.unmapped_train_pair_keys,
        'unmapped_valid_pair_keys': split.unmapped_valid_pair_keys,
        'unmapped_test_pair_keys': split.unmapped_test_pair_keys,
    }


def build_sampling_summary(split: OfficialSplit, plan: SamplingPlan, pair_df: pd.DataFrame) -> Dict:
    sampled_pair_df = plan_pair_df(pair_df, plan.sampled_train_pairs)
    return {
        'generated_at': utc_now(),
        'strategy': plan.strategy,
        'plan_name': plan.name,
        'cdhit_threshold': plan.cdhit_threshold,
        'random_seed': plan.random_seed,
        'original_train_pair_count': int(len(split.mapped_train_pairs)),
        'sampled_train_protein_count': int(len(plan.sampled_train_proteins)),
        'sampled_train_pair_count': int(len(plan.sampled_train_pairs)),
        'sampled_train_residue_row_count': int(sampled_pair_df['residue_rows'].sum()),
        'valid_pair_count': int(len(split.mapped_valid_pairs)),
        'test_pair_count': int(len(split.mapped_test_pairs)),
        'same_sampling_logic_as_scaler_compare': True,
    }


def build_main_command(args, run_dir: Path, allowlist_csv: Path) -> List[str]:
    return [
        str(args.python_executable),
        '-u',
        'main.py',
        '--device',
        str(args.device),
        '--eval_setting',
        args.eval_setting,
        '--fold',
        str(args.fold),
        '--epochs',
        str(args.epochs),
        '--batch_size',
        str(args.batch_size),
        '--learning_rate',
        str(args.learning_rate),
        '--protein_embedding_model',
        args.protein_embedding_model,
        '--protein_feature_source',
        'weighted_pair_embedding',
        '--weighted_embedding_index_path',
        str(args.weighted_embedding_index_path),
        '--pair_manifest_path',
        str(args.pair_manifest_path),
        '--pooling_mode',
        'mean',
        '--train_pair_allowlist_csv',
        str(allowlist_csv),
        '--out_path',
        str(run_dir),
    ]


def summarize_run(plan: SamplingPlan, split: OfficialSplit, run_dir: Path, exit_code: int) -> Dict:
    run_summary_path = run_dir / 'run_summary.json'
    run_summary = {}
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text())

    dataset_debug = run_summary.get('dataset_debug_by_split', {})
    train_debug = dataset_debug.get('train', {})
    valid_debug = dataset_debug.get('valid', {})
    test_debug = dataset_debug.get('test', {})

    train_filter = (train_debug or {}).get('filtering_summary', {})
    valid_filter = (valid_debug or {}).get('filtering_summary', {})
    test_filter = (test_debug or {}).get('filtering_summary', {})

    valid_pairs_actual = len(set((valid_debug or {}).get('sample_pair_ids_order', [])))
    test_pairs_actual = len(set((test_debug or {}).get('sample_pair_ids_order', [])))

    summary = {
        'plan_name': plan.name,
        'strategy': plan.strategy,
        'cdhit_threshold': plan.cdhit_threshold,
        'random_seed': plan.random_seed,
        'exit_code': int(exit_code),
        'status': run_summary.get('status', 'missing_run_summary'),
        'run_dir': str(run_dir),
        'run_summary_path': str(run_summary_path) if run_summary_path.exists() else '',
        'train_pair_allowlist_csv': str(run_dir / 'train_pairs_selected.csv'),
        'train_rows_after_allowlist': train_filter.get('train_rows_after_allowlist'),
        'allowed_unique_pairs': train_filter.get('allowed_unique_pairs'),
        'valid_pair_count_expected': int(len(split.mapped_valid_pairs)),
        'test_pair_count_expected': int(len(split.mapped_test_pairs)),
        'valid_pair_count_actual': int(valid_pairs_actual),
        'test_pair_count_actual': int(test_pairs_actual),
        'valid_test_unchanged': bool(
            valid_pairs_actual == len(split.mapped_valid_pairs)
            and test_pairs_actual == len(split.mapped_test_pairs)
            and valid_filter.get('allowlist_csv_path', '') in ['', None]
            and test_filter.get('allowlist_csv_path', '') in ['', None]
        ),
        'best_metrics': run_summary.get('best_metrics'),
        'best_epoch': run_summary.get('best_epoch'),
        'train_dataset_active_sample_count': (train_debug or {}).get('active_sample_count'),
        'valid_dataset_active_sample_count': (valid_debug or {}).get('active_sample_count'),
        'test_dataset_active_sample_count': (test_debug or {}).get('active_sample_count'),
        'train_log_path': str(run_dir / 'train.log'),
        'model_path_exists': (run_dir / f'setting_{split.eval_setting}_fold{split.fold}.model').exists(),
    }
    return summary


def write_launcher_result_csv(path: Path, summary: Dict) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)


def write_root_run_summary(path: Path, rows: List[Dict]) -> None:
    lines = [
        '# Train-Only Sampling Compare (RD Fixed Weighted)',
        '',
        f'- generated_at: `{utc_now()}`',
        '',
        '| run | strategy | exit_code | allowed_train_pairs | valid_pairs_expected | valid_pairs_actual | test_pairs_expected | test_pairs_actual | valid_test_unchanged | best_roc_auc | best_aupr | run_dir |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |',
    ]
    for row in rows:
        metrics = row.get('best_metrics') or {}
        lines.append(
            f"| {row['plan_name']} | {row['strategy']} | {row['exit_code']} | "
            f"{row.get('allowed_unique_pairs', '')} | {row['valid_pair_count_expected']} | {row['valid_pair_count_actual']} | "
            f"{row['test_pair_count_expected']} | {row['test_pair_count_actual']} | {row['valid_test_unchanged']} | "
            f"{metrics.get('roc_auc', '')} | {metrics.get('aupr', '')} | {row['run_dir']} |"
        )
    path.write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run train-only sampling compare on downstream MultiPPIMI using RD fixed weighted pair assets.')
    parser.add_argument('--eval_setting', type=str, default='S4')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--protein_embedding_model', type=str, default='35M')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--cdhit_threshold', type=float, default=0.5)
    parser.add_argument('--num_random', type=int, default=DEFAULT_NUM_RANDOM)
    parser.add_argument('--seed_pool', type=int, default=DEFAULT_SEED_POOL)
    parser.add_argument('--weighted_embedding_index_path', type=Path, default=DEFAULT_WEIGHTED_REGISTRY)
    parser.add_argument('--pair_manifest_path', type=Path, default=DEFAULT_PAIR_MANIFEST)
    parser.add_argument('--python_executable', type=Path, default=Path('python'))
    parser.add_argument('--output_root', type=Path, default=None)
    parser.add_argument('--disable_full_train', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or build_default_output_root(args.eval_setting, args.fold, args.cdhit_threshold, args.protein_embedding_model)
    output_root.mkdir(parents=True, exist_ok=True)

    pair_df = read_runnable_pair_df()
    protein_df, sequence_map = build_protein_audit(pair_df)
    split = build_official_split(args.eval_setting, args.fold, pair_df)
    train_pair_df = plan_pair_df(pair_df, split.mapped_train_pairs)
    train_proteins = unique_train_proteins(train_pair_df)

    fasta_path = output_root / 'cdhit_train_proteins.fasta'
    cdhit_prefix = output_root / f'cdhit_train_c{str(args.cdhit_threshold).replace(".", "p")}'
    write_fasta(train_proteins, sequence_map, fasta_path)
    cdhit_representatives, cluster_df = run_cdhit(fasta_path, cdhit_prefix, args.cdhit_threshold)
    cluster_df.to_csv(output_root / 'cdhit_clusters.csv', index=False)

    cdhit_pairs = induce_sampled_train_pairs(train_pair_df, set(cdhit_representatives))
    cdhit_plan = SamplingPlan(
        name=f'cdhit_c{str(args.cdhit_threshold).replace(".", "p")}',
        strategy='cdhit_train_only',
        cdhit_threshold=float(args.cdhit_threshold),
        random_seed=None,
        sampled_train_proteins=sorted(cdhit_representatives),
        sampled_train_pairs=cdhit_pairs,
    )
    full_train_plan = SamplingPlan(
        name='full_train',
        strategy='full_train',
        cdhit_threshold=None,
        random_seed=None,
        sampled_train_proteins=train_proteins,
        sampled_train_pairs=split.mapped_train_pairs,
    )
    random_plans = choose_random_plans(
        train_pair_df=train_pair_df,
        train_proteins=train_proteins,
        target_protein_count=len(cdhit_plan.sampled_train_proteins),
        target_pair_count=len(cdhit_plan.sampled_train_pairs),
        target_row_count=train_residue_row_count(pair_df, cdhit_plan.sampled_train_pairs),
        num_random=args.num_random,
        seed_pool=args.seed_pool,
    )
    plans = ([] if args.disable_full_train else [full_train_plan]) + [cdhit_plan] + list(random_plans)

    compare_config = {
        'generated_at': utc_now(),
        'eval_setting': args.eval_setting,
        'fold': args.fold,
        'protein_embedding_model': args.protein_embedding_model,
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'learning_rate': float(args.learning_rate),
        'device': str(args.device),
        'cdhit_threshold': float(args.cdhit_threshold),
        'weighted_embedding_index_path': str(args.weighted_embedding_index_path),
        'pair_manifest_path': str(args.pair_manifest_path),
        'plans': [
            {
                'name': plan.name,
                'strategy': plan.strategy,
                'cdhit_threshold': plan.cdhit_threshold,
                'random_seed': plan.random_seed,
                'sampled_train_protein_count': len(plan.sampled_train_proteins),
                'sampled_train_pair_count': len(plan.sampled_train_pairs),
            }
            for plan in plans
        ],
    }
    write_json(output_root / 'compare_config.json', compare_config)
    protein_df.to_csv(output_root / 'protein_audit.csv', index=False)
    pair_df.to_csv(output_root / 'pair_audit.csv', index=False)

    registry_rows: List[Dict] = []
    for plan in plans:
        run_dir = output_root / plan.name
        run_dir.mkdir(parents=True, exist_ok=True)

        sampled_pair_df = plan_pair_df(pair_df, plan.sampled_train_pairs)
        train_pairs_csv = run_dir / 'train_pairs_selected.csv'
        write_plan_csv(train_pairs_csv, sampled_pair_df)
        write_json(run_dir / 'split_audit.json', build_split_audit(split, plan, pair_df))
        write_json(run_dir / 'sampling_summary.json', build_sampling_summary(split, plan, pair_df))

        command = build_main_command(args, run_dir, train_pairs_csv)
        config_payload = {
            'generated_at': utc_now(),
            'plan_name': plan.name,
            'strategy': plan.strategy,
            'cdhit_threshold': plan.cdhit_threshold,
            'random_seed': plan.random_seed,
            'command': command,
        }
        write_json(run_dir / 'config_full.json', config_payload)

        log_path = run_dir / 'train.log'
        with log_path.open('w') as log_file:
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        launcher_summary = summarize_run(plan, split, run_dir, proc.returncode)
        write_json(run_dir / 'launcher_summary.json', launcher_summary)
        write_launcher_result_csv(run_dir / 'launcher_result.csv', launcher_summary)
        registry_rows.append(launcher_summary)

    registry_df = pd.DataFrame(registry_rows)
    registry_df.to_csv(output_root / 'run_registry.csv', index=False)
    write_root_run_summary(output_root / 'run_summary.md', registry_rows)


if __name__ == '__main__':
    main()
