#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

from rdppimi.ppimi.compound_gnn_model import GNNComplete
from rdppimi.ppimi.datasets.PPIMI_datasets import ModulatorPPIDataset
from rdppimi.ppimi.MultiPPIMI import MultiPPIMI
from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for record in records:
            f.write(json.dumps(json_ready(record)) + '\n')


def read_sample_ids(value: str):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            if 'sample_id' in df.columns:
                return set(df['sample_id'].astype(str))
            if df.shape[1] != 1:
                raise ValueError(f'Sample id CSV must have sample_id or one column: {path}')
            return set(df.iloc[:, 0].astype(str))
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {item.strip() for item in value.split(',') if item.strip()}


def load_prior_setting(args):
    metadata = {}
    if args.prior_setting_metadata:
        p = Path(args.prior_setting_metadata)
        if not p.exists():
            raise FileNotFoundError(f'prior setting metadata file not found: {p}')
        if p.suffix.lower() == '.json':
            metadata = json.loads(p.read_text())
        else:
            metadata = {'metadata_path': str(p), 'text': p.read_text()}
    prior_setting = args.prior_setting or metadata.get('prior_setting') or metadata.get('name') or ''
    return prior_setting, metadata


def load_manifest(path: Path, sample_ids, sample_id_column: str):
    if not path.exists():
        raise FileNotFoundError(f'input manifest not found: {path}')
    df = pd.read_csv(path)
    required = {'SMILES', 'uniprot_id1', 'uniprot_id2'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'input manifest missing required columns {sorted(missing)}: {path}')
    df = df.copy()
    if sample_id_column not in df.columns:
        df[sample_id_column] = [f'sample_{i}' for i in range(len(df))]
    df[sample_id_column] = df[sample_id_column].astype(str)
    if sample_ids is not None:
        before = len(df)
        df = df[df[sample_id_column].isin(sample_ids)].copy()
        if df.empty:
            raise ValueError(f'No rows left after filtering {before} rows by sample IDs')
    if 'label' not in df.columns:
        df['label'] = 0
    return df.reset_index(drop=True)


def build_model(args, device, spec):
    modulator_model = GNNComplete(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    )
    return MultiPPIMI(
        modulator_model,
        modulator_emb_dim=310,
        ppi_emb_dim=spec.paired_dim,
        device=device,
        h_dim=args.h_dim,
        n_heads=args.n_heads,
    ).to(device)


def checkpoint_metadata(path: Path, raw_checkpoint):
    meta = {
        'checkpoint_path': str(path),
        'checkpoint_name': path.name,
        'checkpoint_size_bytes': int(path.stat().st_size),
        'checkpoint_mtime': float(path.stat().st_mtime),
    }
    if isinstance(raw_checkpoint, dict):
        meta['checkpoint_top_level_keys'] = sorted(str(k) for k in raw_checkpoint.keys())[:50]
    return meta


def extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return raw_checkpoint
    raise ValueError('Unsupported checkpoint format: expected a state_dict-like dict')


def normalize_state_dict_keys(state_dict):
    if all(str(k).startswith('module.') for k in state_dict.keys()):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path: Path, device, strict: bool):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    raw_checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(raw_checkpoint))
    load_result = model.load_state_dict(state_dict, strict=strict)
    meta = checkpoint_metadata(checkpoint_path, raw_checkpoint)
    meta['strict_checkpoint'] = bool(strict)
    meta['missing_keys'] = list(getattr(load_result, 'missing_keys', []))
    meta['unexpected_keys'] = list(getattr(load_result, 'unexpected_keys', []))
    return meta


def make_temp_dataset_csv(df: pd.DataFrame):
    token = f'{int(time.time())}_{os.getpid()}_{random.randint(1000, 9999)}'
    setting = f'__softmax_export_{token}'
    mode = 'export'
    fold = '0'
    fold_dir = REPO_ROOT / 'data' / 'folds' / setting
    fold_dir.mkdir(parents=True, exist_ok=False)
    csv_path = fold_dir / f'{mode}_fold{fold}.csv'
    df.to_csv(csv_path, index=False)
    return setting, mode, fold, fold_dir, csv_path


def enrich_record(record, args, sample_id_column, model_scale, prior_setting, prior_setting_metadata, ckpt_meta):
    sample_metadata = record.get('sample_metadata') or {}
    sample_id = sample_metadata.get(sample_id_column)
    if sample_id is None:
        sample_id = record.get('sample_index')
    record['sample_id'] = str(sample_id)
    record['model_scale'] = model_scale
    record['prior_setting'] = prior_setting
    record['prior_setting_metadata'] = prior_setting_metadata
    record['temperature'] = float(args.pooling_softmax_temperature)
    record['checkpoint_metadata'] = ckpt_meta
    record['checkpoint_path'] = str(args.checkpoint)
    record['input_manifest'] = str(args.input_manifest)
    record['protein_a_softmax_weights'] = record['proteinA_softmax_weights']
    record['protein_b_softmax_weights'] = record['proteinB_softmax_weights']
    record['protein_a_length'] = int(record['proteinA_residue_count'])
    record['protein_b_length'] = int(record['proteinB_residue_count'])
    record['prediction_logit'] = record.get('prediction_logit_positive')
    return record


def parse_args():
    parser = argparse.ArgumentParser(description='Export PPIMI ppi_softmax_sum residue pooling weights for interpretation')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model_scale', type=str, required=True, choices=['8M', '35M', '150M', '650M', '3B'])
    parser.add_argument('--input_manifest', type=Path, required=True)
    parser.add_argument('--sample_ids', type=str, default='', help='Comma-separated IDs, text file, or CSV file with sample_id column')
    parser.add_argument('--sample_id_column', type=str, default='sample_id')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--prior_setting', type=str, default='')
    parser.add_argument('--prior_setting_metadata', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pooling_softmax_temperature', type=float, default=1.0)
    parser.add_argument(
        '--protein_feature_source',
        type=str,
        default='residue_level_esm2',
        choices=['residue_level_esm2', 'rd_residue_level_esm2'],
    )
    parser.add_argument('--residue_embedding_index_path', type=str, default='./multippimi_residue_embedding_index.csv')
    parser.add_argument('--pair_manifest_path', type=str, default='./multippimi_pair_manifest.csv')
    parser.add_argument('--ppi_prior_index_path', type=str, default='./multippimi_ppi_prior_index.csv')
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.0)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--gnn_type', type=str, default='gin')
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--strict_checkpoint', action='store_true')
    parser.add_argument('--allow_filtered_samples', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pooling_softmax_temperature <= 0:
        raise ValueError('--pooling_softmax_temperature must be > 0')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {args.device}, but CUDA is not available')
    if args.protein_feature_source == 'rd_residue_level_esm2' and args.model_scale not in {'35M', '150M'}:
        raise ValueError(
            '--protein_feature_source=rd_residue_level_esm2 currently supports only '
            '--model_scale in {35M, 150M}'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = read_sample_ids(args.sample_ids)
    selected_manifest = load_manifest(args.input_manifest, sample_ids, args.sample_id_column)
    prior_setting, prior_setting_metadata = load_prior_setting(args)

    fold_dir = None
    try:
        setting, mode, fold, fold_dir, temp_csv = make_temp_dataset_csv(selected_manifest)
        spec = get_embedding_spec(args.model_scale)
        dataset = ModulatorPPIDataset(
            mode=mode,
            setting=setting,
            fold=fold,
            protein_embedding_model=args.model_scale,
            protein_feature_source=args.protein_feature_source,
            residue_embedding_index_path=args.residue_embedding_index_path,
            pooling_mode='ppi_softmax_sum',
            pair_manifest_path=args.pair_manifest_path,
            ppi_prior_index_path=args.ppi_prior_index_path,
            pooling_softmax_temperature=args.pooling_softmax_temperature,
            enable_softmax_pooling_export=True,
        )
        if len(dataset) != len(selected_manifest) and not args.allow_filtered_samples:
            raise RuntimeError(
                f'Dataset kept {len(dataset)} of {len(selected_manifest)} manifest rows. '
                'Missing prior/residue assets would make exports incomplete; pass --allow_filtered_samples to permit this.'
            )

        device = torch.device(args.device)
        model = build_model(args, device, spec)
        ckpt_meta = load_checkpoint(model, args.checkpoint, device, args.strict_checkpoint)
        model.eval()

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        records = []
        sample_offset = 0
        with torch.no_grad():
            for batch in dataloader:
                modulator, rdkit_descriptors, ppi_feats, _label = batch
                modulator = modulator.to(device)
                rdkit_descriptors = rdkit_descriptors.to(device)
                ppi_feats = ppi_feats.to(device)
                pred = model(modulator, rdkit_descriptors, ppi_feats).squeeze()
                if pred.ndim == 1:
                    pred = pred.unsqueeze(0)
                pred_cpu = pred.detach().cpu().numpy()
                for local_idx in range(pred_cpu.shape[0]):
                    record = dataset.get_softmax_pooling_export(sample_offset + local_idx, pred_cpu[local_idx])
                    records.append(enrich_record(
                        record,
                        args,
                        args.sample_id_column,
                        args.model_scale,
                        prior_setting,
                        prior_setting_metadata,
                        ckpt_meta,
                    ))
                sample_offset += pred_cpu.shape[0]

        output_jsonl = args.output_dir / 'softmax_pooling_weights.jsonl'
        write_jsonl(output_jsonl, records)
        config = {
            'checkpoint': str(args.checkpoint),
            'model_scale': args.model_scale,
            'input_manifest': str(args.input_manifest),
            'selected_manifest_rows': int(len(selected_manifest)),
            'exported_records': int(len(records)),
            'output_jsonl': str(output_jsonl),
            'sample_ids': sorted(sample_ids) if sample_ids is not None else None,
            'sample_id_column': args.sample_id_column,
            'pooling_mode': 'ppi_softmax_sum',
            'pooling_softmax_temperature': float(args.pooling_softmax_temperature),
            'protein_feature_source': args.protein_feature_source,
            'prior_setting': prior_setting,
            'prior_setting_metadata': prior_setting_metadata,
            'residue_embedding_index_path': args.residue_embedding_index_path,
            'pair_manifest_path': args.pair_manifest_path,
            'ppi_prior_index_path': args.ppi_prior_index_path,
            'allow_filtered_samples': bool(args.allow_filtered_samples),
            'checkpoint_metadata': ckpt_meta,
            'temporary_dataset_csv': str(temp_csv),
        }
        (args.output_dir / 'run_config.json').write_text(json.dumps(json_ready(config), indent=2))
        print(f'Wrote {len(records)} records to {output_jsonl}')
    finally:
        if fold_dir is not None and fold_dir.exists():
            shutil.rmtree(fold_dir)


if __name__ == '__main__':
    main()
