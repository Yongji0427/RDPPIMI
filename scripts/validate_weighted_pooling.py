from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from rdppimi.ppimi.datasets.PPIMI_datasets import (
    ModulatorPPIDataset,
    _load_pair_manifest_lookup,
    _load_prior_weight_map,
    _load_residue_level_embedding_paths,
    _normalize_id,
)
from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


def _stats(arr):
    arr = np.asarray(arr)
    return {
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'sum': float(arr.sum(dtype=np.float64)),
    }


def _topk(weights, k=5):
    n = int(min(k, len(weights)))
    if n <= 0:
        return []
    idx = np.argsort(-weights)[:n]
    return [(int(i), float(weights[i])) for i in idx]


def parse_args():
    parser = ArgumentParser(description='Validate ppi_weighted_sum / ppi_softmax_sum pooling paths')
    parser.add_argument('--project_root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--setting', type=str, default='S1')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_key', type=str, default='150M')
    parser.add_argument('--pooling_mode', type=str, default='ppi_weighted_sum', choices=['ppi_weighted_sum', 'ppi_softmax_sum'])
    parser.add_argument('--pooling_softmax_temperature', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=128)
    parser.add_argument('--sample_rows', type=int, default=5)
    parser.add_argument('--output_md', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pooling_mode == 'ppi_softmax_sum' and args.pooling_softmax_temperature <= 0:
        raise ValueError('--pooling_softmax_temperature must be > 0 for ppi_softmax_sum')

    project_root = args.project_root.resolve()
    manifest_path = project_root / 'multippimi_pair_manifest.csv'
    prior_path = project_root / 'multippimi_ppi_prior_index.csv'
    residue_index_path = project_root / 'multippimi_residue_embedding_index.csv'
    output_md = args.output_md or (project_root / 'pooling_validation.md')

    spec = get_embedding_spec(args.model_key)
    pair_lookup = _load_pair_manifest_lookup(manifest_path)
    prior_map = _load_prior_weight_map(prior_path)
    residue_path_map = _load_residue_level_embedding_paths(spec, residue_index_path)

    ds = ModulatorPPIDataset(
        mode='train',
        setting=args.setting,
        fold=args.fold,
        protein_embedding_model=args.model_key,
        protein_feature_source='residue_level_esm2',
        residue_embedding_index_path=str(residue_index_path),
        pooling_mode=args.pooling_mode,
        pair_manifest_path=str(manifest_path),
        ppi_prior_index_path=str(prior_path),
        pooling_softmax_temperature=args.pooling_softmax_temperature,
    )

    active_df = ds.active_df.reset_index(drop=True)
    assert len(active_df) == len(ds.sample_pair_ids), 'active_df and sample_pair_ids length mismatch'

    total_rows = len(active_df)
    eval_rows = min(total_rows, max(1, args.max_samples))

    emb_cache = {}
    raw_cache = {}
    validated_weight_cache = {}

    def load_emb(pid):
        p = residue_path_map[pid]
        if p not in emb_cache:
            emb_cache[p] = np.load(p).astype(np.float32)
        return emb_cache[p]

    def load_raw(path):
        if path not in raw_cache:
            raw_cache[path] = np.load(path).astype(np.float32)
        return raw_cache[path]

    alignment_errors = []
    length_mismatch_errors = []
    pooled_shape_errors = []
    softmax_sum_errors = []
    softmax_nonneg_errors = []
    softmax_nonfinite_errors = []

    sample_rows = []

    for idx in range(eval_rows):
        row = active_df.iloc[idx]
        p1 = _normalize_id(row['uniprot_id1'])
        p2 = _normalize_id(row['uniprot_id2'])
        pair_id, swapped = pair_lookup[(p1, p2)]

        if pair_id != ds.sample_pair_ids[idx]:
            alignment_errors.append((idx, pair_id, ds.sample_pair_ids[idx], p1, p2))

        prior = prior_map[pair_id]
        w1_path = prior['residue_weights_B_path'] if swapped else prior['residue_weights_A_path']
        w2_path = prior['residue_weights_A_path'] if swapped else prior['residue_weights_B_path']

        emb1 = load_emb(p1)
        emb2 = load_emb(p2)
        raw1 = load_raw(w1_path)
        raw2 = load_raw(w2_path)

        if emb1.shape[0] != raw1.shape[0] or emb2.shape[0] != raw2.shape[0]:
            length_mismatch_errors.append((idx, pair_id, p1, p2, emb1.shape, emb2.shape, raw1.shape, raw2.shape))
            continue

        if args.pooling_mode == 'ppi_softmax_sum':
            weights1 = ds._softmax_normalize_scores(raw1, args.pooling_softmax_temperature)
            weights2 = ds._softmax_normalize_scores(raw2, args.pooling_softmax_temperature)
        else:
            weights1 = ds._load_cached_weight(w1_path, validated_weight_cache)
            weights2 = ds._load_cached_weight(w2_path, validated_weight_cache)

        pooled1 = np.matmul(weights1, emb1)
        pooled2 = np.matmul(weights2, emb2)
        if pooled1.shape != (spec.esm_dim,) or pooled2.shape != (spec.esm_dim,):
            pooled_shape_errors.append((idx, pair_id, pooled1.shape, pooled2.shape))

        for side, ww, pid in [('A', weights1, pair_id), ('B', weights2, pair_id)]:
            if not np.isfinite(ww).all():
                softmax_nonfinite_errors.append((idx, pid, side, 'non-finite'))
            if (ww < -1e-7).any():
                softmax_nonneg_errors.append((idx, pid, side, float(ww.min())))
            ww_sum = float(ww.sum(dtype=np.float64))
            if not np.isclose(ww_sum, 1.0, atol=1e-5):
                softmax_sum_errors.append((idx, pid, side, ww_sum))

        if idx < args.sample_rows:
            sample_rows.append({
                'idx': idx,
                'pair_id': pair_id,
                'proteinA_id': p1,
                'proteinB_id': p2,
                'swapped': swapped,
                'embA_shape': tuple(emb1.shape),
                'embB_shape': tuple(emb2.shape),
                'rawA_shape': tuple(raw1.shape),
                'rawB_shape': tuple(raw2.shape),
                'pooledA_shape': tuple(pooled1.shape),
                'pooledB_shape': tuple(pooled2.shape),
                'rawA_stats': _stats(raw1),
                'rawB_stats': _stats(raw2),
                'normA_stats': _stats(weights1),
                'normB_stats': _stats(weights2),
                'topkA': _topk(weights1, 5),
                'topkB': _topk(weights2, 5),
            })

    lines = []
    lines.append('# Pooling Validation Report')
    lines.append('')
    lines.append('## Setup')
    lines.append(f'- setting/fold: `{args.setting}/fold{args.fold}`')
    lines.append(f'- model_key: `{args.model_key}` (esm_dim={spec.esm_dim})')
    lines.append(f'- pooling_mode: `{args.pooling_mode}`')
    if args.pooling_mode == 'ppi_softmax_sum':
        lines.append(f'- pooling_softmax_temperature: `{args.pooling_softmax_temperature}`')
    lines.append(f'- evaluated samples: `{eval_rows}` / kept samples `{total_rows}`')
    lines.append('')

    lines.append('## Softmax Definition')
    lines.append(r'- $\alpha_i = \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}$')
    lines.append(r'- $z = \sum_i \alpha_i e_i$')
    lines.append('')

    lines.append('## Sample-Level Checks')
    lines.append('| idx | pair_id | swapped | embA_shape | embB_shape | rawA_shape | rawB_shape | pooledA_shape | pooledB_shape | topkA(idx,val) | topkB(idx,val) |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in sample_rows:
        lines.append(
            f"| {r['idx']} | {r['pair_id']} | {r['swapped']} | `{r['embA_shape']}` | `{r['embB_shape']}` | "
            f"`{r['rawA_shape']}` | `{r['rawB_shape']}` | `{r['pooledA_shape']}` | `{r['pooledB_shape']}` | "
            f"`{r['topkA']}` | `{r['topkB']}` |"
        )
    lines.append('')

    lines.append('## Raw/Normalized Stats (sample rows)')
    lines.append('| idx | pair_id | rawA(min,max,mean,std,sum) | normA(min,max,mean,std,sum) | rawB(min,max,mean,std,sum) | normB(min,max,mean,std,sum) |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for r in sample_rows:
        ra = r['rawA_stats']
        rb = r['rawB_stats']
        na = r['normA_stats']
        nb = r['normB_stats']
        lines.append(
            f"| {r['idx']} | {r['pair_id']} | "
            f"`({ra['min']:.3e},{ra['max']:.3e},{ra['mean']:.3e},{ra['std']:.3e},{ra['sum']:.3e})` | "
            f"`({na['min']:.3e},{na['max']:.3e},{na['mean']:.3e},{na['std']:.3e},{na['sum']:.3e})` | "
            f"`({rb['min']:.3e},{rb['max']:.3e},{rb['mean']:.3e},{rb['std']:.3e},{rb['sum']:.3e})` | "
            f"`({nb['min']:.3e},{nb['max']:.3e},{nb['mean']:.3e},{nb['std']:.3e},{nb['sum']:.3e})` |"
        )
    lines.append('')

    lines.append('## Global Checks')
    lines.append(f'- pair_id alignment errors: **{len(alignment_errors)}**')
    lines.append(f'- length mismatch errors: **{len(length_mismatch_errors)}**')
    lines.append(f'- pooled output shape errors: **{len(pooled_shape_errors)}**')
    lines.append(f'- normalized weight non-finite errors: **{len(softmax_nonfinite_errors)}**')
    lines.append(f'- normalized weight negative errors: **{len(softmax_nonneg_errors)}**')
    lines.append(f'- normalized weight sum!=1 errors: **{len(softmax_sum_errors)}**')
    lines.append('')

    ok = (
        len(alignment_errors) == 0
        and len(length_mismatch_errors) == 0
        and len(pooled_shape_errors) == 0
        and len(softmax_nonfinite_errors) == 0
        and len(softmax_nonneg_errors) == 0
        and len(softmax_sum_errors) == 0
    )

    lines.append('## Conclusion')
    if ok:
        lines.append('- Validation passed for the evaluated subset: shape/alignment/normalization checks are all satisfied.')
    else:
        lines.append('- Validation failed: inspect error counters above.')

    output_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote: {output_md}')
    print(
        f'pooling_mode={args.pooling_mode} temp={args.pooling_softmax_temperature} '
        f'eval_rows={eval_rows} alignment_errors={len(alignment_errors)} '
        f'len_mismatch={len(length_mismatch_errors)} pooled_shape_errors={len(pooled_shape_errors)} '
        f'nonfinite={len(softmax_nonfinite_errors)} negative={len(softmax_nonneg_errors)} sum_errors={len(softmax_sum_errors)} '
        f'ok={ok}'
    )


if __name__ == '__main__':
    main()
