#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def load_infer_module(script_path: Path):
    spec = importlib.util.spec_from_file_location('ppi_pair_infer_mod', str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load inference module from {script_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_ranges(length: int, chunk_size: int):
    if length <= 0:
        return []
    starts = list(range(0, length, chunk_size))
    return [(s, min(s + chunk_size, length)) for s in starts]


def existing_success(pair_dir: Path, len_a: int, len_b: int) -> bool:
    meta_path = pair_dir / 'metadata.json'
    wA_path = pair_dir / 'residue_weights_A.npy'
    wB_path = pair_dir / 'residue_weights_B.npy'
    sA_path = pair_dir / 'residue_scores_A.npy'
    sB_path = pair_dir / 'residue_scores_B.npy'

    if not (meta_path.exists() and wA_path.exists() and wB_path.exists() and sA_path.exists() and sB_path.exists()):
        return False

    try:
        meta = json.loads(meta_path.read_text())
        if meta.get('status') != 'success':
            return False
        wA = np.load(wA_path)
        wB = np.load(wB_path)
        sA = np.load(sA_path)
        sB = np.load(sB_path)
        if len(wA) != len_a or len(wB) != len_b:
            return False
        if len(sA) != len_a or len(sB) != len_b:
            return False
        if not np.isclose(float(wA.sum(dtype=np.float64)), 1.0, atol=1e-6):
            return False
        if not np.isclose(float(wB.sum(dtype=np.float64)), 1.0, atol=1e-6):
            return False
        return True
    except Exception:
        return False


def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {'1', 'true', 't', 'yes', 'y'}


def main():
    parser = argparse.ArgumentParser(description='Batch-run PPI-seq prior inference for a pair manifest')
    parser.add_argument('--project_root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--manifest_csv', type=Path, default=None)
    parser.add_argument('--config_yaml', type=Path, default=Path('external/proteinshake/config/inference/ppi_seq_final.yaml'))
    parser.add_argument('--infer_script', type=Path, default=Path('external/proteinshake/experiments/infer_ppi_pair_matrix.py'))
    parser.add_argument('--cache_dir', type=Path, default=None)
    parser.add_argument('--index_csv', type=Path, default=None)
    parser.add_argument('--failures_csv', type=Path, default=None)
    parser.add_argument('--save_pair_matrices', action='store_true', help='Also save pair_probs.npy and pair_logits.npy')
    parser.add_argument('--matrix_max_cells', type=int, default=8000000, help='Skip matrix save when LA*LB exceeds this')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--eps', type=float, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_csv = (args.manifest_csv or (root / 'multippimi_pair_manifest.csv')).resolve()
    cache_dir = (args.cache_dir or (root / 'ppi_prior_cache')).resolve()
    index_csv = (args.index_csv or (root / 'multippimi_ppi_prior_index.csv')).resolve()
    failures_csv = (args.failures_csv or (root / 'multippimi_ppi_prior_failures.csv')).resolve()

    if not manifest_csv.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_csv}')
    if not args.config_yaml.exists():
        raise FileNotFoundError(f'Config yaml not found: {args.config_yaml}')
    if not args.infer_script.exists():
        raise FileNotFoundError(f'Inference script not found: {args.infer_script}')

    cache_dir.mkdir(parents=True, exist_ok=True)

    infer_mod = load_infer_module(args.infer_script)
    cfg = OmegaConf.load(str(args.config_yaml))

    if args.eps is not None:
        eps = float(args.eps)
    else:
        eps = float(cfg.inference.get('eps', 1e-8))

    device = infer_mod.select_device(cfg.inference.get('device', 'auto'))
    model = infer_mod.load_model(cfg, device)
    load_report = model.encoder.get_load_report() if hasattr(model.encoder, 'get_load_report') else {}
    infer_mod.enforce_fixed_constraints(cfg, load_report)

    manifest = pd.read_csv(manifest_csv)
    if args.max_pairs is not None:
        manifest = manifest.head(args.max_pairs).copy()

    max_len = int(cfg.representation.max_len)

    records = []
    failed_records = []

    total = len(manifest)
    t0_all = time.time()

    for i, row in manifest.iterrows():
        pair_id = str(row['pair_id']).strip()
        proteinA_id = str(row['proteinA_id']).strip()
        proteinB_id = str(row['proteinB_id']).strip()
        seqA = '' if pd.isna(row.get('seqA')) else str(row.get('seqA'))
        seqB = '' if pd.isna(row.get('seqB')) else str(row.get('seqB'))
        split = '' if pd.isna(row.get('split')) else str(row.get('split'))
        is_runnable_manifest = to_bool(row.get('is_runnable', True))

        len_a = len(seqA)
        len_b = len(seqB)
        pair_dir = cache_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        base_record = {
            'pair_id': pair_id,
            'proteinA_id': proteinA_id,
            'proteinB_id': proteinB_id,
            'split': split,
            'is_runnable_manifest': is_runnable_manifest,
            'lenA': len_a,
            'lenB': len_b,
            'cache_dir': str(pair_dir),
            'residue_weights_A_path': str(pair_dir / 'residue_weights_A.npy'),
            'residue_weights_B_path': str(pair_dir / 'residue_weights_B.npy'),
            'residue_scores_A_path': str(pair_dir / 'residue_scores_A.npy'),
            'residue_scores_B_path': str(pair_dir / 'residue_scores_B.npy'),
            'pair_probs_path': str(pair_dir / 'pair_probs.npy') if args.save_pair_matrices else '',
            'pair_logits_path': str(pair_dir / 'pair_logits.npy') if args.save_pair_matrices else '',
            'metadata_path': str(pair_dir / 'metadata.json'),
            'n_chunks_a': math.ceil(len_a / max_len) if len_a > 0 else 0,
            'n_chunks_b': math.ceil(len_b / max_len) if len_b > 0 else 0,
            'n_chunk_pairs': (math.ceil(len_a / max_len) * math.ceil(len_b / max_len)) if (len_a > 0 and len_b > 0) else 0,
            'status': '',
            'error': '',
            'elapsed_sec': 0.0,
        }

        start = time.time()
        print(f'[{len(records)+len(failed_records)+1}/{total}] pair_id={pair_id} lenA={len_a} lenB={len_b}')

        if (not args.force_recompute) and existing_success(pair_dir, len_a, len_b):
            rec = dict(base_record)
            rec['status'] = 'cached'
            rec['elapsed_sec'] = round(time.time() - start, 4)
            records.append(rec)
            print(f'  -> cached')
            continue

        if not is_runnable_manifest:
            rec = dict(base_record)
            rec['status'] = 'failed'
            rec['error'] = 'manifest is_runnable=false (missing sequence)'
            rec['elapsed_sec'] = round(time.time() - start, 4)
            failed_records.append(rec)
            print('  -> failed: is_runnable=false')
            continue

        if len_a == 0 or len_b == 0:
            rec = dict(base_record)
            rec['status'] = 'failed'
            rec['error'] = 'empty sequence'
            rec['elapsed_sec'] = round(time.time() - start, 4)
            failed_records.append(rec)
            print('  -> failed: empty sequence')
            continue

        try:
            chunks_a = split_ranges(len_a, max_len)
            chunks_b = split_ranges(len_b, max_len)

            residue_scores_a = np.full((len_a,), -np.inf, dtype=np.float32)
            residue_scores_b = np.full((len_b,), -np.inf, dtype=np.float32)

            save_mats = args.save_pair_matrices and (len_a * len_b <= args.matrix_max_cells)
            pair_probs = None
            pair_logits = None
            if save_mats:
                pair_probs = np.zeros((len_a, len_b), dtype=np.float32)
                pair_logits = np.zeros((len_a, len_b), dtype=np.float32)

            for a0, a1 in chunks_a:
                subA = seqA[a0:a1]
                for b0, b1 in chunks_b:
                    subB = seqB[b0:b1]
                    out = infer_mod.predict(model, subA, subB, max_len=max_len)
                    probs = out['probs']
                    logits = out['logits']

                    residue_scores_a[a0:a1] = np.maximum(residue_scores_a[a0:a1], probs.max(axis=1))
                    residue_scores_b[b0:b1] = np.maximum(residue_scores_b[b0:b1], probs.max(axis=0))

                    if save_mats:
                        pair_probs[a0:a1, b0:b1] = probs
                        pair_logits[a0:a1, b0:b1] = logits

            if not np.isfinite(residue_scores_a).all() or not np.isfinite(residue_scores_b).all():
                raise RuntimeError('non-finite residue scores after chunk aggregation')

            weights_a = infer_mod.normalize_weights(residue_scores_a, eps)
            weights_b = infer_mod.normalize_weights(residue_scores_b, eps)

            if len(weights_a) != len_a or len(weights_b) != len_b:
                raise RuntimeError('weight length mismatch')
            if not np.isclose(float(weights_a.sum(dtype=np.float64)), 1.0, atol=1e-6):
                raise RuntimeError(f'weights_A sum != 1, got {weights_a.sum()}')
            if not np.isclose(float(weights_b.sum(dtype=np.float64)), 1.0, atol=1e-6):
                raise RuntimeError(f'weights_B sum != 1, got {weights_b.sum()}')

            np.save(pair_dir / 'residue_scores_A.npy', residue_scores_a)
            np.save(pair_dir / 'residue_scores_B.npy', residue_scores_b)
            np.save(pair_dir / 'residue_weights_A.npy', weights_a)
            np.save(pair_dir / 'residue_weights_B.npy', weights_b)

            if save_mats:
                np.save(pair_dir / 'pair_probs.npy', pair_probs)
                np.save(pair_dir / 'pair_logits.npy', pair_logits)

            meta = {
                'status': 'success',
                'pair_id': pair_id,
                'proteinA_id': proteinA_id,
                'proteinB_id': proteinB_id,
                'lenA': len_a,
                'lenB': len_b,
                'split': split,
                'max_len': max_len,
                'eps': eps,
                'n_chunks_a': len(chunks_a),
                'n_chunks_b': len(chunks_b),
                'n_chunk_pairs': len(chunks_a) * len(chunks_b),
                'save_pair_matrices': bool(save_mats),
                'loaded_pretrained': bool(load_report.get('loaded_pretrained', False)),
                'used_random_init': bool(load_report.get('used_random_init', True)),
                'load_source': load_report.get('load_source', 'unknown'),
                'weight_sum_A': float(weights_a.sum(dtype=np.float64)),
                'weight_sum_B': float(weights_b.sum(dtype=np.float64)),
            }
            (pair_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

            rec = dict(base_record)
            rec['status'] = 'success'
            rec['elapsed_sec'] = round(time.time() - start, 4)
            if not save_mats:
                rec['pair_probs_path'] = ''
                rec['pair_logits_path'] = ''
            records.append(rec)
            print(f"  -> success (chunks={len(chunks_a)}x{len(chunks_b)})")

        except Exception as e:
            err_msg = f'{type(e).__name__}: {e}'
            tb = traceback.format_exc(limit=3)
            meta = {
                'status': 'failed',
                'pair_id': pair_id,
                'error': err_msg,
                'traceback': tb,
            }
            (pair_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

            rec = dict(base_record)
            rec['status'] = 'failed'
            rec['error'] = err_msg
            rec['elapsed_sec'] = round(time.time() - start, 4)
            failed_records.append(rec)
            print(f'  -> failed: {err_msg}')

    index_df = pd.DataFrame(records + failed_records)
    # keep original manifest order where possible
    if 'pair_id' in index_df.columns:
        order_map = {pid: i for i, pid in enumerate(manifest['pair_id'].astype(str).tolist())}
        index_df['__order'] = index_df['pair_id'].map(order_map)
        index_df = index_df.sort_values(['__order', 'status']).drop(columns=['__order'])

    index_df.to_csv(index_csv, index=False)

    failures_df = index_df[index_df['status'] == 'failed'].copy()
    failures_df.to_csv(failures_csv, index=False)

    elapsed_all = time.time() - t0_all
    print('\n=== Batch Prior Inference Done ===')
    print(f'index_csv={index_csv}')
    print(f'failures_csv={failures_csv}')
    print(f'cache_dir={cache_dir}')
    print(f'total_pairs={total}')
    print(f"success={int((index_df['status']=='success').sum())}")
    print(f"cached={int((index_df['status']=='cached').sum())}")
    print(f"failed={int((index_df['status']=='failed').sum())}")
    print(f'elapsed_sec={elapsed_all:.2f}')


if __name__ == '__main__':
    main()
