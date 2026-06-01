#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdppimi.rd.fixed_weighted_residue_rd_loader import load_fixed_weighted_residue_transition

SCALE_IN = '8M'
SCALE_OUT = '35M'
PAIR_MANIFEST_CSV = PROJECT_ROOT / 'multippimi_pair_manifest.csv'
WEIGHT_INDEX_CSV = PROJECT_ROOT / 'fixed_residue_softmax_weights' / 't0p7' / 'fixed_residue_softmax_weight_index.csv'
DEFAULT_RD_PROJECT_ROOT = Path('external/reverse_distillation')
DEFAULT_EVAL_SETTING = 'S4'
DEFAULT_FOLD = 1
DEFAULT_CDHIT_THRESHOLD = 0.7
DEFAULT_NUM_RANDOM = 3
DEFAULT_SEED_POOL = 1000


@dataclass(frozen=True)
class OfficialSplit:
    eval_setting: str
    fold: int
    original_train_pair_key_count: int
    original_valid_pair_key_count: int
    original_test_pair_key_count: int
    mapped_train_pairs: List[str]
    mapped_valid_pairs: List[str]
    mapped_test_pairs: List[str]
    unmapped_train_pair_keys: List[str]
    unmapped_valid_pair_keys: List[str]
    unmapped_test_pair_keys: List[str]


@dataclass(frozen=True)
class SamplingPlan:
    name: str
    strategy: str
    cdhit_threshold: Optional[float]
    random_seed: Optional[int]
    sampled_train_proteins: List[str]
    sampled_train_pairs: List[str]


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    strategy: str
    cdhit_threshold: Optional[float]
    random_seed: Optional[int]
    original_train_pair_count: int
    original_valid_pair_count: int
    original_test_pair_count: int
    sampled_train_protein_count: int
    sampled_train_pair_count: int
    sampled_train_residue_row_count: int
    valid_residue_row_count: int
    test_residue_row_count: int
    train_mse: float
    train_r2: float
    valid_mse: float
    valid_r2: float
    test_mse: float
    test_r2: float
    train_valid_gap: float
    train_test_gap: float
    residual_pca_effective_rank: int
    scaler_path: str


def utc_now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def timestamp_label() -> str:
    return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def build_default_output_root() -> Path:
    return PROJECT_ROOT / 'runs' / 'rd_weighted_residue_subset_compare_8M_35M_t0p7' / timestamp_label()


def build_default_report_path() -> Path:
    return PROJECT_ROOT / 'analysis_outputs' / 'reverse_distillation_audit' / f'random_vs_cdhit_weighted_rd_8M_35M_{timestamp_label()}.md'


def add_rd_project_to_path(rd_project_root: Path) -> None:
    rd_src_root = rd_project_root / 'src'
    if not rd_src_root.exists():
        raise FileNotFoundError(f'RD project src root not found: {rd_src_root}')
    if str(rd_src_root) not in sys.path:
        sys.path.insert(0, str(rd_src_root))


def unordered_pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def cdhit_word_size(threshold: float) -> int:
    if threshold >= 0.7:
        return 5
    if threshold >= 0.6:
        return 4
    if threshold >= 0.5:
        return 3
    raise ValueError(f'Unsupported CD-HIT threshold {threshold}; expected >= 0.5 for this script')


def read_runnable_pair_df() -> pd.DataFrame:
    manifest = pd.read_csv(PAIR_MANIFEST_CSV)
    manifest = manifest[manifest['is_runnable'].astype(str).str.lower().isin(['true', '1'])].copy()
    weights = pd.read_csv(WEIGHT_INDEX_CSV)
    weights = weights[weights['status'].astype(str).isin(['success', 'cached'])].copy()
    pair_df = manifest[['pair_id', 'proteinA_id', 'proteinB_id', 'seqA', 'seqB']].merge(
        weights[['pair_id', 'protein_id_A', 'protein_id_B', 'length_A', 'length_B']],
        on='pair_id',
        how='inner',
    )
    if not ((pair_df['proteinA_id'] == pair_df['protein_id_A']) & (pair_df['proteinB_id'] == pair_df['protein_id_B'])).all():
        raise ValueError('Manifest and weight index disagree on protein IDs')
    pair_df['pair_key_unordered'] = pair_df.apply(lambda r: unordered_pair_key(r['proteinA_id'], r['proteinB_id']), axis=1)
    if pair_df['pair_key_unordered'].duplicated().any():
        raise ValueError('Unordered pair key is not unique in runnable pair assets')
    pair_df['residue_rows'] = pair_df['length_A'].astype(int) + pair_df['length_B'].astype(int)
    return pair_df[['pair_id', 'proteinA_id', 'proteinB_id', 'seqA', 'seqB', 'length_A', 'length_B', 'residue_rows', 'pair_key_unordered']].copy()


def read_embedding_lengths() -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    for scale in ['8M', '35M', '150M', '650M', '3B']:
        df = pd.read_csv(PROJECT_ROOT / f'multippimi_residue_embedding_index_{scale}.csv')
        df = df[df['status'].astype(str).isin(['success', 'cached'])].drop_duplicates('protein_id')
        for row in df.itertuples(index=False):
            protein_id = str(row.protein_id)
            seq_len = int(row.seq_len)
            if protein_id in lengths and lengths[protein_id] != seq_len:
                raise ValueError(f'Cross-scale residue embedding length mismatch for {protein_id}')
            lengths.setdefault(protein_id, seq_len)
    return lengths


def build_protein_audit(pair_df: pd.DataFrame) -> pd.DataFrame:
    embedding_lengths = read_embedding_lengths()
    usage: Dict[str, int] = {}
    sequence_map: Dict[str, str] = {}
    source_map: Dict[str, str] = {}
    for row in pair_df.itertuples(index=False):
        for side, protein_id, seq in [('A', row.proteinA_id, str(row.seqA)), ('B', row.proteinB_id, str(row.seqB))]:
            usage[protein_id] = usage.get(protein_id, 0) + 1
            if protein_id in sequence_map and sequence_map[protein_id] != seq:
                raise ValueError(f'Inconsistent sequence observed for {protein_id}')
            sequence_map.setdefault(protein_id, seq)
            source_map.setdefault(protein_id, f'{PAIR_MANIFEST_CSV}:{row.pair_id}:seq{side}')
    rows = []
    for protein_id in sorted(sequence_map):
        if protein_id not in embedding_lengths:
            raise KeyError(f'Missing residue embedding length for protein {protein_id}')
        rows.append(
            {
                'protein_id': protein_id,
                'sequence_source': source_map[protein_id],
                'sequence_length': len(sequence_map[protein_id]),
                'residue_embedding_length': embedding_lengths[protein_id],
                'pair_usage_count': usage[protein_id],
            }
        )
    return pd.DataFrame(rows), sequence_map


def read_official_split_pair_keys(eval_setting: str, fold: int, split: str) -> List[Tuple[str, str]]:
    path = PROJECT_ROOT / 'data' / 'folds' / eval_setting / f'{split}_fold{fold}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing official split CSV: {path}')
    df = pd.read_csv(path)
    keys = sorted({unordered_pair_key(a, b) for a, b in zip(df['uniprot_id1'].astype(str), df['uniprot_id2'].astype(str))})
    return keys


def build_official_split(eval_setting: str, fold: int, pair_df: pd.DataFrame) -> OfficialSplit:
    pair_map = {tuple(key): pair_id for key, pair_id in zip(pair_df['pair_key_unordered'], pair_df['pair_id'])}

    def map_keys(keys: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        mapped = []
        unmapped = []
        for key in keys:
            pair_id = pair_map.get(tuple(key))
            if pair_id is not None:
                mapped.append(pair_id)
            else:
                unmapped.append('|'.join(key))
        return mapped, unmapped

    train_keys = read_official_split_pair_keys(eval_setting, fold, 'train')
    valid_keys = read_official_split_pair_keys(eval_setting, fold, 'valid')
    test_keys = read_official_split_pair_keys(eval_setting, fold, 'test')
    train_pairs, unmapped_train = map_keys(train_keys)
    valid_pairs, unmapped_valid = map_keys(valid_keys)
    test_pairs, unmapped_test = map_keys(test_keys)
    return OfficialSplit(
        eval_setting=eval_setting,
        fold=fold,
        original_train_pair_key_count=len(train_keys),
        original_valid_pair_key_count=len(valid_keys),
        original_test_pair_key_count=len(test_keys),
        mapped_train_pairs=train_pairs,
        mapped_valid_pairs=valid_pairs,
        mapped_test_pairs=test_pairs,
        unmapped_train_pair_keys=unmapped_train,
        unmapped_valid_pair_keys=unmapped_valid,
        unmapped_test_pair_keys=unmapped_test,
    )


def write_fasta(proteins: Sequence[str], sequence_map: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for protein_id in sorted(proteins):
            f.write(f'>{protein_id}\n{sequence_map[protein_id]}\n')


def run_cdhit(fasta_path: Path, output_prefix: Path, threshold: float) -> Tuple[List[str], pd.DataFrame]:
    word_size = cdhit_word_size(threshold)
    cmd = [
        'cd-hit',
        '-i', str(fasta_path),
        '-o', str(output_prefix),
        '-c', str(threshold),
        '-n', str(word_size),
        '-d', '0',
        '-M', '0',
        '-T', '0',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output_prefix.with_suffix('.log').write_text(proc.stdout + '\n' + proc.stderr)

    representatives: List[str] = []
    with output_prefix.open() as f:
        for line in f:
            if line.startswith('>'):
                representatives.append(line[1:].strip().split()[0])

    clusters: List[Dict[str, object]] = []
    current = None
    for raw in output_prefix.with_suffix('.clstr').read_text().splitlines():
        line = raw.strip()
        if line.startswith('>Cluster'):
            if current is not None:
                clusters.append(current)
            current = {'cluster_id': int(line.split()[1]), 'representative_protein': None, 'members': []}
            continue
        if not line:
            continue
        protein_id = line.split('>')[1].split('...')[0]
        current['members'].append(protein_id)
        if line.endswith('*'):
            current['representative_protein'] = protein_id
    if current is not None:
        clusters.append(current)

    cluster_df = pd.DataFrame(
        {
            'cluster_id': cluster['cluster_id'],
            'representative_protein': cluster['representative_protein'],
            'member_count': len(cluster['members']),
            'members': ';'.join(cluster['members']),
        }
        for cluster in clusters
    ).sort_values('cluster_id').reset_index(drop=True)
    return representatives, cluster_df


def induce_sampled_train_pairs(train_pair_df: pd.DataFrame, retained_protein_set: set[str]) -> List[str]:
    sampled = train_pair_df[
        train_pair_df['proteinA_id'].isin(retained_protein_set) & train_pair_df['proteinB_id'].isin(retained_protein_set)
    ]
    return sampled['pair_id'].tolist()


def choose_random_plans(
    train_pair_df: pd.DataFrame,
    train_proteins: List[str],
    target_protein_count: int,
    target_pair_count: int,
    target_row_count: int,
    num_random: int,
    seed_pool: int,
) -> List[SamplingPlan]:
    candidates = []
    for seed in range(seed_pool):
        rng = random.Random(seed)
        chosen = set(rng.sample(train_proteins, target_protein_count))
        sampled_pairs = induce_sampled_train_pairs(train_pair_df, chosen)
        if not sampled_pairs:
            continue
        sampled_row_count = int(train_pair_df[train_pair_df['pair_id'].isin(sampled_pairs)]['residue_rows'].sum())
        candidates.append((abs(len(sampled_pairs) - target_pair_count), abs(sampled_row_count - target_row_count), seed, chosen, sampled_pairs))
    if len(candidates) < num_random:
        raise RuntimeError(f'Only found {len(candidates)} random candidates, need {num_random}')
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    plans = []
    for _, _, seed, chosen, sampled_pairs in candidates[:num_random]:
        plans.append(
            SamplingPlan(
                name=f'random_seed_{seed}',
                strategy='random_match',
                cdhit_threshold=None,
                random_seed=seed,
                sampled_train_proteins=sorted(chosen),
                sampled_train_pairs=sampled_pairs,
            )
        )
    return plans


def build_pair_mask(bundle, pair_ids: Sequence[str]) -> np.ndarray:
    pair_set = set(pair_ids)
    pair_ids_per_row = np.array([row.pair_id for row in bundle.row_keys], dtype=object)
    return np.isin(pair_ids_per_row, list(pair_set))


def build_rd_scaler():
    from reverse_distillation.scaler.modules import IncrementalPCAWrapper
    from reverse_distillation.scaler.rd import rdScaler

    scaler = rdScaler(plm_size_in=SCALE_IN, plm_size_out=SCALE_OUT, regressor_type='linear', pca_type='incremental')
    n_components = scaler.n_features_out - scaler.n_features_in
    scaler.pca = IncrementalPCAWrapper(n_components=n_components, batch_size=max(131072, n_components))
    return scaler


def metric_pair(rd_scaler, xin: np.ndarray, xout: np.ndarray) -> Tuple[float, float]:
    pred = rd_scaler.predict_regressor(xin)
    return float(mean_squared_error(xout, pred)), float(r2_score(xout, pred))


def save_scaler(output_path: Path, state_dict: Dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **state_dict)


def run_plan(bundle, split: OfficialSplit, plan: SamplingPlan, pair_df: pd.DataFrame, output_root: Path) -> ExperimentResult:
    sampled_train_pair_df = pair_df[pair_df['pair_id'].isin(plan.sampled_train_pairs)].copy()
    valid_pair_df = pair_df[pair_df['pair_id'].isin(split.mapped_valid_pairs)].copy()
    test_pair_df = pair_df[pair_df['pair_id'].isin(split.mapped_test_pairs)].copy()

    train_mask = build_pair_mask(bundle, plan.sampled_train_pairs)
    valid_mask = build_pair_mask(bundle, split.mapped_valid_pairs)
    test_mask = build_pair_mask(bundle, split.mapped_test_pairs)

    train_xin, train_xout = bundle.xin[train_mask], bundle.xout[train_mask]
    valid_xin, valid_xout = bundle.xin[valid_mask], bundle.xout[valid_mask]
    test_xin, test_xout = bundle.xin[test_mask], bundle.xout[test_mask]

    if train_xin.shape[0] == 0:
        raise ValueError(f'{plan.name} has zero sampled train residue rows')
    if valid_xin.shape[0] == 0 or test_xin.shape[0] == 0:
        raise ValueError(f'{plan.name} has empty valid/test residue rows; official benchmark mapping failed')

    scaler = build_rd_scaler()
    scaler.fit(xin=train_xin, xout=train_xout, show_r2=True)
    train_mse, train_r2 = metric_pair(scaler, train_xin, train_xout)
    valid_mse, valid_r2 = metric_pair(scaler, valid_xin, valid_xout)
    test_mse, test_r2 = metric_pair(scaler, test_xin, test_xout)

    exp_dir = output_root / plan.name
    scaler_path = exp_dir / f'rd-scaler-{SCALE_IN}-{SCALE_OUT}-linear-incremental-{plan.name}.npz'
    save_scaler(scaler_path, scaler.get_state_dict())
    sampled_train_pair_df.to_csv(exp_dir / 'sampled_train_pairs.csv', index=False)
    valid_pair_df.to_csv(exp_dir / 'valid_pairs.csv', index=False)
    test_pair_df.to_csv(exp_dir / 'test_pairs.csv', index=False)
    (exp_dir / 'sampled_train_proteins.txt').write_text('\n'.join(plan.sampled_train_proteins) + '\n')

    summary = {
        'name': plan.name,
        'strategy': plan.strategy,
        'cdhit_threshold': plan.cdhit_threshold,
        'random_seed': plan.random_seed,
        'original_train_pair_count': len(split.mapped_train_pairs),
        'original_valid_pair_count': len(split.mapped_valid_pairs),
        'original_test_pair_count': len(split.mapped_test_pairs),
        'sampled_train_protein_count': len(plan.sampled_train_proteins),
        'sampled_train_pair_count': len(plan.sampled_train_pairs),
        'sampled_train_residue_row_count': int(train_xin.shape[0]),
        'valid_residue_row_count': int(valid_xin.shape[0]),
        'test_residue_row_count': int(test_xin.shape[0]),
        'train_mse': train_mse,
        'train_r2': train_r2,
        'valid_mse': valid_mse,
        'valid_r2': valid_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'train_valid_gap': train_r2 - valid_r2,
        'train_test_gap': train_r2 - test_r2,
        'residual_pca_effective_rank': int(scaler.pca.components_.shape[0]),
        'scaler_path': str(scaler_path),
    }
    (exp_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))

    result = ExperimentResult(**summary)
    del scaler, train_xin, train_xout, valid_xin, valid_xout, test_xin, test_xout
    gc.collect()
    return result


def write_report(
    report_path: Path,
    output_root: Path,
    protein_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    split: OfficialSplit,
    cdhit_plan: SamplingPlan,
    random_plans: Sequence[SamplingPlan],
    cluster_df: pd.DataFrame,
    results: Sequence[ExperimentResult],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_map = {res.name: res for res in results}
    train_valid_overlap = len(set(split.mapped_train_pairs) & set(split.mapped_valid_pairs))
    train_test_overlap = len(set(split.mapped_train_pairs) & set(split.mapped_test_pairs))
    valid_test_overlap = len(set(split.mapped_valid_pairs) & set(split.mapped_test_pairs))

    lines = [
        '# Random vs CD-HIT Weighted RD Under Fixed Official Valid/Test',
        '',
        '## Summary',
        '',
        f'- generated_at: `{utc_now()}`',
        f'- output_root: `{output_root}`',
        f'- report_path: `{report_path}`',
        f'- transition: `{SCALE_IN} -> {SCALE_OUT}`',
        f'- eval setting: `{split.eval_setting}`',
        f'- fold: `{split.fold}`',
        f'- fixed weight index: `{WEIGHT_INDEX_CSV}`',
        '',
        '## Official Split Audit',
        '',
        f'- runnable pairs in canonical asset universe: `{len(pair_df)}`',
        f'- unique proteins in canonical asset universe: `{len(protein_df)}`',
        f'- original train unique pair keys in official CSV: `{split.original_train_pair_key_count}`',
        f'- original valid unique pair keys in official CSV: `{split.original_valid_pair_key_count}`',
        f'- original test unique pair keys in official CSV: `{split.original_test_pair_key_count}`',
        f'- mapped train pair count in runnable asset universe: `{len(split.mapped_train_pairs)}`',
        f'- mapped valid pair count in runnable asset universe: `{len(split.mapped_valid_pairs)}`',
        f'- mapped test pair count in runnable asset universe: `{len(split.mapped_test_pairs)}`',
        f'- unmapped train pair keys: `{len(split.unmapped_train_pair_keys)}`',
        f'- unmapped valid pair keys: `{len(split.unmapped_valid_pair_keys)}`',
        f'- unmapped test pair keys: `{len(split.unmapped_test_pair_keys)}`',
        f'- pair overlap train-valid after pair aggregation: `{train_valid_overlap}`',
        f'- pair overlap train-test after pair aggregation: `{train_test_overlap}`',
        f'- pair overlap valid-test after pair aggregation: `{valid_test_overlap}`',
        '',
        'Sample protein audit rows:',
        '',
        '| protein_id | sequence_source | residue_embedding_length | pair_usage_count |',
        '| --- | --- | ---: | ---: |',
    ]
    for row in protein_df.head(5).itertuples(index=False):
        lines.append(f'| {row.protein_id} | {row.sequence_source} | {int(row.residue_embedding_length)} | {int(row.pair_usage_count)} |')

    cdhit_result = result_map[cdhit_plan.name]
    lines.extend([
        '',
        '## Sampling Rules',
        '',
        '- valid/test are fixed from the original official split and are never resampled.',
        '- CD-HIT and random-match operate only inside the original train pair universe.',
        '- Sampling unit is train proteins. A train pair is kept only when both proteins are retained.',
        '',
        '## Subset Statistics',
        '',
        '### CD-HIT',
        '',
        f'- threshold: `{cdhit_plan.cdhit_threshold}`',
        f'- original train pair count: `{len(split.mapped_train_pairs)}`',
        f'- sampled train protein count: `{len(cdhit_plan.sampled_train_proteins)}`',
        f'- sampled train pair count: `{len(cdhit_plan.sampled_train_pairs)}`',
        f'- sampled train residue row count: `{cdhit_result.sampled_train_residue_row_count}`',
        f'- original valid pair count: `{len(split.mapped_valid_pairs)}`',
        f'- original test pair count: `{len(split.mapped_test_pairs)}`',
        '',
        '### Random-match',
        '',
        '| seed | sampled train protein count | sampled train pair count | sampled train residue row count | original valid pair count | original test pair count |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for plan in random_plans:
        res = result_map[plan.name]
        lines.append(
            f'| {plan.random_seed} | {len(plan.sampled_train_proteins)} | {len(plan.sampled_train_pairs)} | '
            f'{res.sampled_train_residue_row_count} | {len(split.mapped_valid_pairs)} | {len(split.mapped_test_pairs)} |'
        )

    lines.extend([
        '',
        '## Scaler Metrics',
        '',
        '| experiment | strategy | seed | train MSE | train R2 | valid MSE | valid R2 | test MSE | test R2 | train-valid gap | train-test gap | sampled train rows | scaler path |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ])
    for res in results:
        lines.append(
            f'| {res.name} | {res.strategy} | {res.random_seed if res.random_seed is not None else "-"} | '
            f'{res.train_mse:.10e} | {res.train_r2:.6f} | {res.valid_mse:.10e} | {res.valid_r2:.6f} | '
            f'{res.test_mse:.10e} | {res.test_r2:.6f} | {res.train_valid_gap:.6f} | {res.train_test_gap:.6f} | '
            f'{res.sampled_train_residue_row_count} | `{res.scaler_path}` |'
        )

    random_results = [result_map[plan.name] for plan in random_plans]
    mean_random_train_r2 = sum(res.train_r2 for res in random_results) / len(random_results)
    mean_random_valid_r2 = sum(res.valid_r2 for res in random_results) / len(random_results)
    mean_random_test_r2 = sum(res.test_r2 for res in random_results) / len(random_results)
    best_random_test = max(random_results, key=lambda res: res.test_r2)
    lines.extend([
        '',
        '## Smoke Conclusion',
        '',
        f'- 使用 fold: `{split.eval_setting}/fold{split.fold}`',
        f'- valid/test 是否保持原始官方 split 不变: `yes`。脚本直接从 `data/folds/{split.eval_setting}` 读取 valid/test pair keys，并未对其做采样或 CD-HIT。',
        f'- random 是否更容易拟合: `{"yes" if mean_random_train_r2 > cdhit_result.train_r2 else "no"}`。CD-HIT train R2=`{cdhit_result.train_r2:.6f}`，random mean train R2=`{mean_random_train_r2:.6f}`。',
        f'- CD-HIT 是否更利于 valid/test 泛化: `{"yes" if (cdhit_result.valid_r2 > mean_random_valid_r2 and cdhit_result.test_r2 > mean_random_test_r2) else "no"}`。'
        f'CD-HIT valid/test R2=`{cdhit_result.valid_r2:.6f}`/`{cdhit_result.test_r2:.6f}`，'
        f'random mean valid/test R2=`{mean_random_valid_r2:.6f}`/`{mean_random_test_r2:.6f}`，'
        f'best random test R2 来自 `{best_random_test.name}` = `{best_random_test.test_r2:.6f}`。',
        '- 当前 smoke 结果主要验证了流程，不足以直接指导正式主线，因为 official pair aggregation 后 train/valid 存在大规模重叠，需要在下一轮明确是否接受这种 pair-level leakage，或改成更严格的 split 映射规则。',
    ])
    report_path.write_text('\n'.join(lines) + '\n')


def build_official_split_payload(split: OfficialSplit) -> Dict[str, object]:
    return {
        'eval_setting': split.eval_setting,
        'fold': split.fold,
        'original_train_pair_key_count': split.original_train_pair_key_count,
        'original_valid_pair_key_count': split.original_valid_pair_key_count,
        'original_test_pair_key_count': split.original_test_pair_key_count,
        'mapped_train_pair_count': len(split.mapped_train_pairs),
        'mapped_valid_pair_count': len(split.mapped_valid_pairs),
        'mapped_test_pair_count': len(split.mapped_test_pairs),
        'unmapped_train_pair_keys': split.unmapped_train_pair_keys,
        'unmapped_valid_pair_keys': split.unmapped_valid_pair_keys,
        'unmapped_test_pair_keys': split.unmapped_test_pair_keys,
        'mapped_train_pairs': split.mapped_train_pairs,
        'mapped_valid_pairs': split.mapped_valid_pairs,
        'mapped_test_pairs': split.mapped_test_pairs,
    }


def execute_fold(
    bundle,
    pair_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    sequence_map: Dict[str, str],
    eval_setting: str,
    fold: int,
    cdhit_threshold: float,
    num_random: int,
    seed_pool: int,
    output_root: Path,
    report_path: Path,
    verbose: bool,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    split = build_official_split(eval_setting, fold, pair_df)
    (output_root / 'official_split_audit.json').write_text(json.dumps(build_official_split_payload(split), indent=2, sort_keys=True))

    train_pair_df = pair_df[pair_df['pair_id'].isin(split.mapped_train_pairs)].copy()
    train_proteins = sorted(set(train_pair_df['proteinA_id']).union(set(train_pair_df['proteinB_id'])))

    train_fasta = output_root / f'{eval_setting}_fold{fold}_train_proteins.fasta'
    write_fasta(train_proteins, sequence_map, train_fasta)
    cdhit_prefix = output_root / f'{eval_setting}_fold{fold}_train_cdhit'
    retained_train_proteins, cluster_df = run_cdhit(train_fasta, cdhit_prefix, cdhit_threshold)
    cluster_df.to_csv(output_root / 'cdhit_clusters.csv', index=False)

    cdhit_plan = SamplingPlan(
        name=f'cdhit_trainonly_{eval_setting}_fold{fold}_c{str(cdhit_threshold).replace(".", "p")}',
        strategy='cdhit_train_only',
        cdhit_threshold=cdhit_threshold,
        random_seed=None,
        sampled_train_proteins=sorted(retained_train_proteins),
        sampled_train_pairs=induce_sampled_train_pairs(train_pair_df, set(retained_train_proteins)),
    )
    target_train_rows = int(train_pair_df[train_pair_df['pair_id'].isin(cdhit_plan.sampled_train_pairs)]['residue_rows'].sum())
    random_plans = choose_random_plans(
        train_pair_df=train_pair_df,
        train_proteins=train_proteins,
        target_protein_count=len(cdhit_plan.sampled_train_proteins),
        target_pair_count=len(cdhit_plan.sampled_train_pairs),
        target_row_count=target_train_rows,
        num_random=num_random,
        seed_pool=seed_pool,
    )

    results: List[ExperimentResult] = []
    for plan in [cdhit_plan, *random_plans]:
        if verbose:
            print(f'\n[experiment] {plan.name}')
            print(
                f'original_train_pairs={len(split.mapped_train_pairs)} sampled_train_proteins={len(plan.sampled_train_proteins)} '
                f'sampled_train_pairs={len(plan.sampled_train_pairs)} original_valid_pairs={len(split.mapped_valid_pairs)} '
                f'original_test_pairs={len(split.mapped_test_pairs)}'
            )
        results.append(run_plan(bundle, split, plan, pair_df, output_root))

    results_df = pd.DataFrame([dict(res.__dict__) for res in results])
    results_df.insert(0, 'fold', fold)
    results_df.insert(1, 'eval_setting', eval_setting)
    results_df.to_csv(output_root / 'experiment_results.csv', index=False)
    write_report(report_path, output_root, protein_df, pair_df, split, cdhit_plan, random_plans, cluster_df, results)

    return {
        'fold': fold,
        'split': split,
        'cdhit_plan': cdhit_plan,
        'random_plans': random_plans,
        'results_df': results_df,
        'output_root': output_root,
        'report_path': report_path,
    }


def format_mean_std(series: pd.Series) -> str:
    if len(series) == 0:
        return 'n/a'
    if len(series) == 1:
        return f'{series.iloc[0]:.6f} +- 0.000000'
    return f'{series.mean():.6f} +- {series.std(ddof=1):.6f}'


def write_aggregate_report(
    report_path: Path,
    root_output_dir: Path,
    eval_setting: str,
    folds: Sequence[int],
    cdhit_threshold: float,
    fold_payloads: Sequence[Dict[str, object]],
    combined_df: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for payload in fold_payloads:
        split: OfficialSplit = payload['split']
        split_rows.append({
            'fold': split.fold,
            'original_train_pair_key_count': split.original_train_pair_key_count,
            'original_valid_pair_key_count': split.original_valid_pair_key_count,
            'original_test_pair_key_count': split.original_test_pair_key_count,
            'mapped_train_pair_count': len(split.mapped_train_pairs),
            'mapped_valid_pair_count': len(split.mapped_valid_pairs),
            'mapped_test_pair_count': len(split.mapped_test_pairs),
            'unmapped_train_pair_key_count': len(split.unmapped_train_pair_keys),
            'unmapped_valid_pair_key_count': len(split.unmapped_valid_pair_keys),
            'unmapped_test_pair_key_count': len(split.unmapped_test_pair_keys),
        })
    split_df = pd.DataFrame(split_rows).sort_values('fold').reset_index(drop=True)
    split_df.to_csv(root_output_dir / 'split_summary.csv', index=False)

    cdhit_df = combined_df[combined_df['strategy'] == 'cdhit_train_only'].copy().sort_values('fold').reset_index(drop=True)
    random_df = combined_df[combined_df['strategy'] == 'random_match'].copy().sort_values(['fold', 'random_seed']).reset_index(drop=True)
    random_df.to_csv(root_output_dir / 'random_seed_results.csv', index=False)
    cdhit_df.to_csv(root_output_dir / 'cdhit_results.csv', index=False)

    random_fold_mean = random_df.groupby('fold', as_index=False).agg(
        random_seed_count=('random_seed', 'count'),
        sampled_train_protein_count_mean=('sampled_train_protein_count', 'mean'),
        sampled_train_pair_count_mean=('sampled_train_pair_count', 'mean'),
        sampled_train_residue_row_count_mean=('sampled_train_residue_row_count', 'mean'),
        train_mse_mean=('train_mse', 'mean'),
        train_r2_mean=('train_r2', 'mean'),
        valid_mse_mean=('valid_mse', 'mean'),
        valid_r2_mean=('valid_r2', 'mean'),
        test_mse_mean=('test_mse', 'mean'),
        test_r2_mean=('test_r2', 'mean'),
        train_valid_gap_mean=('train_valid_gap', 'mean'),
        train_test_gap_mean=('train_test_gap', 'mean'),
    )
    random_fold_std = random_df.groupby('fold', as_index=False).agg(
        train_r2_std=('train_r2', 'std'),
        valid_r2_std=('valid_r2', 'std'),
        test_r2_std=('test_r2', 'std'),
        train_valid_gap_std=('train_valid_gap', 'std'),
        train_test_gap_std=('train_test_gap', 'std'),
    )
    random_fold_summary = random_fold_mean.merge(random_fold_std, on='fold', how='left').sort_values('fold').reset_index(drop=True)
    random_fold_summary.to_csv(root_output_dir / 'random_fold_summary.csv', index=False)

    delta_df = cdhit_df[['fold', 'valid_r2', 'test_r2']].merge(
        random_fold_summary[['fold', 'valid_r2_mean', 'test_r2_mean']],
        on='fold',
        how='left',
    )
    delta_df['delta_valid_R2'] = delta_df['valid_r2'] - delta_df['valid_r2_mean']
    delta_df['delta_test_R2'] = delta_df['test_r2'] - delta_df['test_r2_mean']
    delta_df.to_csv(root_output_dir / 'cdhit_vs_random_delta.csv', index=False)

    cdhit_stats = {
        'train_r2': format_mean_std(cdhit_df['train_r2']),
        'valid_r2': format_mean_std(cdhit_df['valid_r2']),
        'test_r2': format_mean_std(cdhit_df['test_r2']),
    }
    random_stats = {
        'train_r2_mean': format_mean_std(random_fold_summary['train_r2_mean']),
        'valid_r2_mean': format_mean_std(random_fold_summary['valid_r2_mean']),
        'test_r2_mean': format_mean_std(random_fold_summary['test_r2_mean']),
    }
    delta_valid_overall = delta_df['delta_valid_R2'].mean()
    delta_test_overall = delta_df['delta_test_R2'].mean()

    lines = [
        '# Random vs CD-HIT Weighted RD Under Fixed Official Valid/Test (5-fold)',
        '',
        '## Experiment Definition',
        '',
        '- train-only sampling: yes',
        '- valid/test unchanged: yes',
        f'- transition: `{SCALE_IN} -> {SCALE_OUT}`',
        f'- eval setting: `{eval_setting}`',
        f'- folds: `{", ".join(f"fold{fold}" for fold in folds)}`',
        f'- CD-HIT threshold: `{cdhit_threshold}`',
        '- matched random seeds per fold: 3',
        f'- fixed weight index: `{WEIGHT_INDEX_CSV}`',
        f'- root_output_dir: `{root_output_dir}`',
        f'- report_path: `{report_path}`',
        '',
        '## Per-Fold Split Statistics',
        '',
        '| fold | original train pair keys | original valid pair keys | original test pair keys | runnable train pairs | runnable valid pairs | runnable test pairs | unmapped train | unmapped valid | unmapped test |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in split_df.itertuples(index=False):
        lines.append(
            f'| {row.fold} | {row.original_train_pair_key_count} | {row.original_valid_pair_key_count} | {row.original_test_pair_key_count} | '
            f'{row.mapped_train_pair_count} | {row.mapped_valid_pair_count} | {row.mapped_test_pair_count} | '
            f'{row.unmapped_train_pair_key_count} | {row.unmapped_valid_pair_key_count} | {row.unmapped_test_pair_key_count} |'
        )

    lines.extend([
        '',
        '## Per-Fold Results',
        '',
        '| fold | strategy | random seed | sampled train proteins | sampled train pairs | sampled train rows | train R2 | valid R2 | test R2 | train-valid gap | train-test gap |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in combined_df.sort_values(['fold', 'strategy', 'random_seed'], na_position='first').itertuples(index=False):
        seed = '-' if pd.isna(row.random_seed) else int(row.random_seed)
        lines.append(
            f'| {int(row.fold)} | {row.strategy} | {seed} | {int(row.sampled_train_protein_count)} | {int(row.sampled_train_pair_count)} | {int(row.sampled_train_residue_row_count)} | '
            f'{row.train_r2:.6f} | {row.valid_r2:.6f} | {row.test_r2:.6f} | {row.train_valid_gap:.6f} | {row.train_test_gap:.6f} |'
        )

    lines.extend([
        '',
        '## 5-fold Summary',
        '',
        '### CD-HIT 0.5',
        '',
        '| fold | train R2 | valid R2 | test R2 |',
        '| ---: | ---: | ---: | ---: |',
    ])
    for row in cdhit_df.itertuples(index=False):
        lines.append(f'| {int(row.fold)} | {row.train_r2:.6f} | {row.valid_r2:.6f} | {row.test_r2:.6f} |')
    lines.extend([
        '',
        f'- CD-HIT train R2 mean +- std: `{cdhit_stats["train_r2"]}`',
        f'- CD-HIT valid R2 mean +- std: `{cdhit_stats["valid_r2"]}`',
        f'- CD-HIT test R2 mean +- std: `{cdhit_stats["test_r2"]}`',
        '',
        '### Matched Random Per-Seed Results',
        '',
        '| fold | random seed | sampled train proteins | sampled train pairs | sampled train rows | train R2 | valid R2 | test R2 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in random_df.itertuples(index=False):
        lines.append(
            f'| {int(row.fold)} | {int(row.random_seed)} | {int(row.sampled_train_protein_count)} | {int(row.sampled_train_pair_count)} | {int(row.sampled_train_residue_row_count)} | '
            f'{row.train_r2:.6f} | {row.valid_r2:.6f} | {row.test_r2:.6f} |'
        )

    lines.extend([
        '',
        '### Matched Random Fold-Level Mean',
        '',
        '| fold | random seed count | mean train R2 | mean valid R2 | mean test R2 | std train R2 | std valid R2 | std test R2 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in random_fold_summary.itertuples(index=False):
        lines.append(
            f'| {int(row.fold)} | {int(row.random_seed_count)} | {row.train_r2_mean:.6f} | {row.valid_r2_mean:.6f} | {row.test_r2_mean:.6f} | '
            f'{row.train_r2_std:.6f} | {row.valid_r2_std:.6f} | {row.test_r2_std:.6f} |'
        )
    lines.extend([
        '',
        f'- Random fold-mean train R2 overall mean +- std: `{random_stats["train_r2_mean"]}`',
        f'- Random fold-mean valid R2 overall mean +- std: `{random_stats["valid_r2_mean"]}`',
        f'- Random fold-mean test R2 overall mean +- std: `{random_stats["test_r2_mean"]}`',
        '',
        '### Direct Comparison Against Mean Random',
        '',
        '| fold | CD-HIT valid R2 | mean random valid R2 | delta_valid_R2 | CD-HIT test R2 | mean random test R2 | delta_test_R2 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in delta_df.itertuples(index=False):
        lines.append(
            f'| {int(row.fold)} | {row.valid_r2:.6f} | {row.valid_r2_mean:.6f} | {row.delta_valid_R2:.6f} | '
            f'{row.test_r2:.6f} | {row.test_r2_mean:.6f} | {row.delta_test_R2:.6f} |'
        )

    supports_cdhit = delta_valid_overall > 0 and delta_test_overall > 0
    advantage_scope = []
    if delta_valid_overall > 0:
        advantage_scope.append('valid')
    if delta_test_overall > 0:
        advantage_scope.append('test')
    advantage_text = ', '.join(advantage_scope) if advantage_scope else 'none'
    lines.extend([
        '',
        f'- Overall mean delta_valid_R2: `{delta_valid_overall:.6f}`',
        f'- Overall mean delta_test_R2: `{delta_test_overall:.6f}`',
        '',
        '## Initial Conclusion',
        '',
        f'- CD-HIT 0.5 是否在多 fold 平均上优于 matched random: `{"yes" if supports_cdhit else "no"}`。',
        f'- 优势主要体现在 valid 还是 test: `{advantage_text}`。',
        f'- 是否值得继续扩到 full chain: `{"yes" if supports_cdhit else "no"}`。',
        f'- 是否值得继续测试 c=0.3: `{"yes" if not supports_cdhit else "optional"}`。',
    ])
    report_path.write_text('\n'.join(lines) + '\n')


def run_batch_folds(
    bundle,
    pair_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    sequence_map: Dict[str, str],
    eval_setting: str,
    folds: Sequence[int],
    cdhit_threshold: float,
    num_random: int,
    seed_pool: int,
    output_root: Path,
    report_path: Path,
    verbose: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    protein_df.to_csv(output_root / 'protein_audit.csv', index=False)
    pair_df[['pair_id', 'proteinA_id', 'proteinB_id', 'length_A', 'length_B', 'residue_rows']].to_csv(output_root / 'pair_audit.csv', index=False)

    fold_payloads = []
    combined_frames = []
    for fold in folds:
        fold_output_root = output_root / f'fold{fold}'
        fold_report_path = fold_output_root / 'report.md'
        payload = execute_fold(
            bundle=bundle,
            pair_df=pair_df,
            protein_df=protein_df,
            sequence_map=sequence_map,
            eval_setting=eval_setting,
            fold=fold,
            cdhit_threshold=cdhit_threshold,
            num_random=num_random,
            seed_pool=seed_pool,
            output_root=fold_output_root,
            report_path=fold_report_path,
            verbose=verbose,
        )
        fold_payloads.append(payload)
        combined_frames.append(payload['results_df'])

    combined_df = pd.concat(combined_frames, ignore_index=True)
    combined_df.to_csv(output_root / 'all_folds_experiment_results.csv', index=False)
    write_aggregate_report(report_path, output_root, eval_setting, folds, cdhit_threshold, fold_payloads, combined_df)

    delta_df = pd.read_csv(output_root / 'cdhit_vs_random_delta.csv')
    print(json.dumps(
        {
            'output_root': str(output_root),
            'report_path': str(report_path),
            'eval_setting': eval_setting,
            'folds': list(folds),
            'cdhit_threshold': cdhit_threshold,
            'overall_mean_delta_valid_R2': float(delta_df['delta_valid_R2'].mean()),
            'overall_mean_delta_test_R2': float(delta_df['delta_test_R2'].mean()),
            'final_valid_test_unchanged': True,
        },
        indent=2,
        default=str,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare CD-HIT and random train-only sampling under fixed official valid/test for weighted RD.')
    parser.add_argument('--output_root', type=Path, default=build_default_output_root())
    parser.add_argument('--report_path', type=Path, default=build_default_report_path())
    parser.add_argument('--eval_setting', type=str, default=DEFAULT_EVAL_SETTING)
    parser.add_argument('--fold', type=int, default=DEFAULT_FOLD)
    parser.add_argument('--folds', type=int, nargs='+', default=None)
    parser.add_argument('--cdhit_threshold', type=float, default=DEFAULT_CDHIT_THRESHOLD)
    parser.add_argument('--num_random', type=int, default=DEFAULT_NUM_RANDOM)
    parser.add_argument('--seed_pool', type=int, default=DEFAULT_SEED_POOL)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    add_rd_project_to_path(DEFAULT_RD_PROJECT_ROOT)

    pair_df = read_runnable_pair_df()
    protein_df, sequence_map = build_protein_audit(pair_df)
    bundle = load_fixed_weighted_residue_transition(SCALE_IN, SCALE_OUT, weight_index_csv=WEIGHT_INDEX_CSV, verbose=args.verbose)

    if args.folds:
        folds = args.folds
        run_batch_folds(
            bundle=bundle,
            pair_df=pair_df,
            protein_df=protein_df,
            sequence_map=sequence_map,
            eval_setting=args.eval_setting,
            folds=folds,
            cdhit_threshold=args.cdhit_threshold,
            num_random=args.num_random,
            seed_pool=args.seed_pool,
            output_root=args.output_root,
            report_path=args.report_path,
            verbose=args.verbose,
        )
        return

    protein_df.to_csv(args.output_root / 'protein_audit.csv', index=False)
    pair_df[['pair_id', 'proteinA_id', 'proteinB_id', 'length_A', 'length_B', 'residue_rows']].to_csv(args.output_root / 'pair_audit.csv', index=False)
    payload = execute_fold(
        bundle=bundle,
        pair_df=pair_df,
        protein_df=protein_df,
        sequence_map=sequence_map,
        eval_setting=args.eval_setting,
        fold=args.fold,
        cdhit_threshold=args.cdhit_threshold,
        num_random=args.num_random,
        seed_pool=args.seed_pool,
        output_root=args.output_root,
        report_path=args.report_path,
        verbose=args.verbose,
    )
    split: OfficialSplit = payload['split']
    random_plans: List[SamplingPlan] = payload['random_plans']
    results_df: pd.DataFrame = payload['results_df']
    print(json.dumps(
        {
            'output_root': str(args.output_root),
            'report_path': str(args.report_path),
            'eval_setting': args.eval_setting,
            'fold': args.fold,
            'cdhit_threshold': args.cdhit_threshold,
            'random_seeds': [plan.random_seed for plan in random_plans],
            'original_train_pair_count': len(split.mapped_train_pairs),
            'original_valid_pair_count': len(split.mapped_valid_pairs),
            'original_test_pair_count': len(split.mapped_test_pairs),
            'valid_test_unchanged': True,
            'results': results_df.to_dict(orient='records'),
        },
        indent=2,
        default=str,
    ))


if __name__ == '__main__':
    main()
