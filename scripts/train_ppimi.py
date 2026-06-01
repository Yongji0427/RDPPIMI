import argparse
import copy
import json
import math
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from rdppimi.ppimi.datasets.PPIMI_datasets import ModulatorPPIDataset, performance_evaluation
from rdppimi.ppimi.compound_gnn_model import GNNComplete
from rdppimi.ppimi.MultiPPIMI import MultiPPIMI
from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


def _metrics_to_dict(metrics_tuple):
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, _ = metrics_tuple
    return {
        'roc_auc': float(roc_auc),
        'aupr': float(aupr),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'mcc': float(mcc),
    }


def _effective_seed(args):
    return int(args.runseed)


def _seed_everything(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _make_worker_init_fn(base_seed):
    base_seed = int(base_seed)

    def _worker_init_fn(worker_id):
        worker_seed = base_seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def _build_dataloader(dataset, batch_size, shuffle, base_seed, seed_offset, name):
    generator_seed = int(base_seed) + int(seed_offset)
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    worker_init_fn = _make_worker_init_fn(generator_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    debug = {
        'name': str(name),
        'shuffle': bool(shuffle),
        'batch_size': int(batch_size),
        'num_workers': 0,
        'generator_seed': int(generator_seed),
        'worker_seed_base': int(generator_seed),
        'worker_init_policy': 'base_seed_plus_worker_id',
    }
    return dataloader, debug


def _dataset_debug_summary(dataset):
    if hasattr(dataset, 'get_debug_summary'):
        return dataset.get_debug_summary()
    return None


def _resolve_device(device_arg):
    device_str = str(device_arg).strip().lower()
    if device_str == 'cpu':
        return torch.device('cpu')
    if device_str.startswith('cuda'):
        return torch.device(device_str)
    return torch.device(f'cuda:{device_arg}')


def _build_ppimi_model(args, device, embedding_spec):
    modulator_model = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    if args.pretrained_model_file != '':
        print('========= Loading from {}'.format(args.pretrained_model_file))
        modulator_model.load_state_dict(torch.load(args.pretrained_model_file, map_location=device))
    model = MultiPPIMI(
        modulator_model,
        modulator_emb_dim=310,
        ppi_emb_dim=embedding_spec.paired_dim,
        protein_projector_dim=args.ppi_hidden_dim,
        protein_projector_mode=args.protein_projector_mode,
        protein_esm_dim=args.protein_projector_esm_dim or embedding_spec.esm_dim,
        protein_phy_dim=args.protein_projector_phy_dim,
        protein_projector_target_esm_dim=args.protein_projector_target_esm_dim,
        protein_projector_gate_init=args.protein_projector_gate_init,
        protein_projector_init_scale=args.protein_projector_init_scale,
        protein_projector_residual_init_scale=args.protein_projector_residual_init_scale,
        protein_projector_residual_dropout=args.protein_projector_residual_dropout,
        protein_projector_residual_norm_clip_ratio=args.protein_projector_residual_norm_clip_ratio,
        device=device,
        h_dim=args.h_dim,
        n_heads=args.n_heads,
        ).to(device)
    if args.protein_projector_prefix_checkpoint_path:
        print('========= Loading protein projector prefix from {}'.format(args.protein_projector_prefix_checkpoint_path))
        model.load_protein_projector_prefix_checkpoint(args.protein_projector_prefix_checkpoint_path, map_location=device)
    return model


def _build_classification_criterion(args, train_dataset, device):
    class_weights = None
    if args.class_weight_mode == 'balanced':
        label_counts = torch.bincount(train_dataset.label_list.cpu(), minlength=2).float()
        if (label_counts <= 0).any():
            raise ValueError(f'Cannot build balanced class weights with label counts: {label_counts.tolist()}')
        class_weights = label_counts.sum() / (label_counts.numel() * label_counts)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion, class_weights


def _current_learning_rates(optimizer):
    return [float(group['lr']) for group in optimizer.param_groups]


def _selection_score(metric_name, metrics_dict):
    if metric_name in {'roc_auc', 'late_roc_auc'}:
        return float(metrics_dict['roc_auc'])
    if metric_name == 'roc_plus_aupr':
        return float(metrics_dict['roc_auc']) + float(metrics_dict['aupr'])
    raise ValueError(f'Unsupported checkpoint selection metric: {metric_name}')


def _checkpoint_selection_allowed(metric_name, epoch, args):
    if metric_name != 'late_roc_auc':
        return True
    return int(epoch) >= int(args.late_checkpoint_start_epoch)


def _save_model_state(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _build_scheduler(args, optimizer):
    warmup_epochs = max(0, int(args.lr_warmup_epochs))
    total_epochs = max(1, int(args.epochs))
    warmup_start_factor = float(args.lr_warmup_start_factor)
    lr_min = float(args.lr_min)
    if warmup_start_factor <= 0 or warmup_start_factor > 1.0:
        raise ValueError('--lr_warmup_start_factor must be in (0, 1]')
    if lr_min < 0:
        raise ValueError('--lr_min must be >= 0')
    if lr_min > float(args.learning_rate):
        raise ValueError('--lr_min must be <= --learning_rate')
    if warmup_epochs >= total_epochs:
        warmup_epochs = max(0, total_epochs - 1)

    schedule_enabled = warmup_epochs > 0 or lr_min > 0.0
    scheduler_info = {
        'enabled': bool(schedule_enabled),
        'warmup_epochs': int(warmup_epochs),
        'warmup_start_factor': float(warmup_start_factor),
        'lr_min': float(lr_min),
    }
    if not schedule_enabled:
        return None, scheduler_info

    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=lr_min,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine
    return scheduler, scheduler_info


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_softmax_pooling_exports(output_path, records):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for record in records:
            f.write(json.dumps(_json_ready(record)) + '\n')


def _prediction_export_record(dataset, sample_idx, pred_row):
    if not hasattr(dataset, 'get_softmax_pooling_export'):
        raise RuntimeError('Dataset does not implement get_softmax_pooling_export')
    return dataset.get_softmax_pooling_export(sample_idx, prediction_logits=pred_row)


def train(PPIMI_model, device, dataloader, optimizer, criterion, grad_clip_norm=0.0):
    PPIMI_model.train()
    loss_accum = 0.0
    for step_idx, batch in enumerate(dataloader):
        modulator, rdkit_descriptors, ppi_esm, label = batch
        modulator = modulator.to(device)
        rdkit_descriptors = rdkit_descriptors.to(device)
        ppi_esm = ppi_esm.to(device)
        label = label.to(device)
        pred = PPIMI_model(modulator, rdkit_descriptors, ppi_esm).squeeze()

        optimizer.zero_grad()
        loss = criterion(pred, label)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(PPIMI_model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        loss_accum += float(loss.detach().item())
    mean_loss = loss_accum / max(1, len(dataloader))
    print('Loss:	{}'.format(mean_loss))
    return mean_loss


def predicting(PPIMI_model, device, dataloader, return_softmax_pooling_weights=False):
    PPIMI_model.eval()
    total_preds = []
    total_labels = []
    softmax_pooling_records = []
    sample_offset = 0
    with torch.no_grad():
        for batch in dataloader:
            modulator, rdkit_descriptors, ppi_esm, label = batch
            modulator = modulator.to(device)
            rdkit_descriptors = rdkit_descriptors.to(device)
            ppi_esm = ppi_esm.to(device)
            label = label.to(device)
            pred = PPIMI_model(modulator, rdkit_descriptors, ppi_esm).squeeze()
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            pred_cpu = pred.detach().cpu()
            total_preds.append(pred_cpu)
            total_labels.append(label.detach().cpu())
            if return_softmax_pooling_weights:
                for local_idx in range(pred_cpu.shape[0]):
                    softmax_pooling_records.append(
                        _prediction_export_record(dataloader.dataset, sample_offset + local_idx, pred_cpu[local_idx].numpy())
                    )
                sample_offset += pred_cpu.shape[0]

    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    if return_softmax_pooling_weights:
        return total_labels.numpy(), total_preds.numpy(), softmax_pooling_records
    return total_labels.numpy(), total_preds.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of MultiPPIMI')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--eval_setting', type=str, default='S1', choices=['S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--fold', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_warmup_epochs', type=int, default=0)
    parser.add_argument('--lr_warmup_start_factor', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--grad_clip_norm', type=float, default=0.0)
    parser.add_argument('--checkpoint_selection_metric', type=str, default='roc_auc', choices=['roc_auc', 'roc_plus_aupr', 'late_roc_auc'])
    parser.add_argument('--late_checkpoint_start_epoch', type=int, default=0)
    parser.add_argument('--late_checkpoint_interval', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--pretrained_model_file', type=str, default='./src/GraphMVP_C.model')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--out_path', type=str, default='.')
    ########## For compound embedding ##########
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--gnn_type', type=str, default='gin')
    ########## For protein embedding ##########
    parser.add_argument(
        '--protein_embedding_model',
        type=str,
        default='150M',
        choices=[
            '8M', '35M', '150M', '650M', '3B',
            'mean_concat_8M_35M',
            'mean_concat_8M_35M_150M',
            'mean_concat_8M_35M_150M_650M',
            'mean_concat_5scale',
            'softmax_concat_8M_35M',
            'softmax_concat_8M_35M_150M',
            'softmax_concat_8M_35M_150M_650M',
            'softmax_concat_5scale',
        ],
    )
    parser.add_argument(
        '--protein_feature_source',
        type=str,
        default='original_pooled_esm2',
        choices=['original_pooled_esm2', 'residue_level_esm2', 'rd_residue_level_esm2', 'weighted_pair_embedding'],
        help=(
            'Protein feature source. Use weighted_pair_embedding to consume precomputed pair-level pooled embeddings; '
            'in that branch --pooling_mode must stay mean as a compatibility placeholder.'
        ),
    )
    parser.add_argument(
        '--residue_embedding_index_path',
        type=str,
        default='./multippimi_residue_embedding_index.csv',
    )
    parser.add_argument(
        '--pooling_mode',
        type=str,
        default='mean',
        choices=['mean', 'ppi_weighted_sum', 'ppi_softmax_sum'],
        help=(
            'Pooling applied when residue-level inputs are used. For protein_feature_source=weighted_pair_embedding, '
            'keep this as mean; no additional mean pooling is performed because the pair-level embedding is already pooled.'
        ),
    )
    parser.add_argument(
        '--pooling_softmax_temperature',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--pair_manifest_path',
        type=str,
        default='./multippimi_pair_manifest.csv',
    )
    parser.add_argument(
        '--ppi_prior_index_path',
        type=str,
        default='./multippimi_ppi_prior_index.csv',
    )
    parser.add_argument(
        '--weighted_embedding_index_path',
        type=str,
        default='./multippimi_weighted_embedding_index.csv',
    )
    parser.add_argument(
        '--train_pair_allowlist_csv',
        type=str,
        default='',
        help='Optional canonical pair_id allowlist CSV. Applied only to the train split; valid/test ignore it.',
    )
    parser.add_argument(
        '--export_softmax_pooling_weights_path',
        type=str,
        default='',
        help='Optional JSONL path for test-set ppi_softmax_sum residue pooling weight export.',
    )
    parser.add_argument('--ppi_hidden_dim', type=int, default=None)
    parser.add_argument(
        '--protein_projector_mode',
        type=str,
        default='linear',
        choices=['none', 'linear', 'block_esm_to_35m', 'block_650m_prefix_residual_to35m'],
        help=(
            'Protein pair projector. linear preserves the historical global projector; '
            'block_esm_to_35m slices [A_esm, A_phy, B_esm, B_phy], projects only ESM, '
            'and preserves PHY. block_650m_prefix_residual_to35m keeps the inherited prefix path '
            'separate from a weakly gated residual path.'
        ),
    )
    parser.add_argument('--protein_projector_esm_dim', type=int, default=0)
    parser.add_argument('--protein_projector_phy_dim', type=int, default=19)
    parser.add_argument('--protein_projector_target_esm_dim', type=int, default=None)
    parser.add_argument('--protein_projector_gate_init', type=float, default=0.85)
    parser.add_argument('--protein_projector_init_scale', type=float, default=1.0)
    parser.add_argument('--protein_projector_residual_init_scale', type=float, default=0.05)
    parser.add_argument('--protein_projector_residual_dropout', type=float, default=0.2)
    parser.add_argument('--protein_projector_residual_norm_clip_ratio', type=float, default=0.35)
    parser.add_argument('--protein_projector_prefix_checkpoint_path', type=str, default='')
    parser.add_argument('--class_weight_mode', type=str, default='none', choices=['none', 'balanced'])
    parser.add_argument('--missing_sequence_pair_fallback_ids', type=str, default='')
    ########## For attention module ##########
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    args = parser.parse_args()

    embedding_spec = get_embedding_spec(args.protein_embedding_model)
    softmax_residue_sources = {'residue_level_esm2', 'rd_residue_level_esm2'}
    if args.pooling_mode in {'ppi_weighted_sum', 'ppi_softmax_sum'} and args.protein_feature_source not in softmax_residue_sources:
        raise ValueError(
            '--pooling_mode in {ppi_weighted_sum, ppi_softmax_sum} requires '
            '--protein_feature_source in {residue_level_esm2, rd_residue_level_esm2}'
        )
    if args.pooling_mode == 'ppi_softmax_sum' and args.pooling_softmax_temperature <= 0:
        raise ValueError('--pooling_softmax_temperature must be > 0 when --pooling_mode=ppi_softmax_sum')
    if args.protein_feature_source == 'rd_residue_level_esm2':
        if args.protein_embedding_model not in {'35M', '150M', '650M', '3B'}:
            raise ValueError(
                '--protein_feature_source=rd_residue_level_esm2 currently supports only '
                '--protein_embedding_model in {35M, 150M, 650M, 3B}'
            )
        if args.pooling_mode != 'ppi_softmax_sum':
            raise ValueError('--protein_feature_source=rd_residue_level_esm2 currently supports only --pooling_mode=ppi_softmax_sum')
    if args.protein_feature_source == 'weighted_pair_embedding' and args.pooling_mode != 'mean':
        raise ValueError(
            '--protein_feature_source=weighted_pair_embedding requires --pooling_mode=mean '
            '(compatibility placeholder only; no additional mean pooling is applied because the '
            'pair-level weighted/softmax pooling is already precomputed).'
        )
    if args.export_softmax_pooling_weights_path and args.pooling_mode != 'ppi_softmax_sum':
        raise ValueError('--export_softmax_pooling_weights_path requires --pooling_mode=ppi_softmax_sum')

    if args.ppi_hidden_dim is None:
        args.ppi_hidden_dim = embedding_spec.paired_dim
    elif args.ppi_hidden_dim <= 0:
        raise ValueError('--ppi_hidden_dim must be positive')
    elif args.ppi_hidden_dim > embedding_spec.paired_dim:
        raise ValueError(
            f'--ppi_hidden_dim={args.ppi_hidden_dim} exceeds '
            f'{args.protein_embedding_model} paired dim {embedding_spec.paired_dim}'
        )
    if args.protein_projector_mode == 'none' and args.ppi_hidden_dim != embedding_spec.paired_dim:
        raise ValueError('--protein_projector_mode=none requires --ppi_hidden_dim to equal the paired input dim')
    if args.protein_projector_mode in {'block_esm_to_35m', 'block_650m_prefix_residual_to35m'}:
        projector_esm_dim = args.protein_projector_esm_dim or embedding_spec.esm_dim
        expected_input_dim = 2 * (projector_esm_dim + args.protein_projector_phy_dim)
        if expected_input_dim != embedding_spec.paired_dim:
            raise ValueError(
                f'--protein_projector_mode={args.protein_projector_mode} expected input dim '
                f'2 * ({projector_esm_dim} + {args.protein_projector_phy_dim}) = '
                f'{expected_input_dim}, but embedding paired dim is {embedding_spec.paired_dim}'
            )
        if args.protein_projector_target_esm_dim is None:
            raise ValueError(f'--protein_projector_target_esm_dim is required for {args.protein_projector_mode}')
        expected_output_dim = 2 * (args.protein_projector_target_esm_dim + args.protein_projector_phy_dim)
        if args.ppi_hidden_dim != expected_output_dim:
            raise ValueError(
                f'--protein_projector_mode={args.protein_projector_mode} expected --ppi_hidden_dim '
                f'{expected_output_dim}, got {args.ppi_hidden_dim}'
            )
        if not 0.0 < args.protein_projector_gate_init < 1.0:
            raise ValueError('--protein_projector_gate_init must be in (0, 1)')
        if args.protein_projector_init_scale <= 0:
            raise ValueError('--protein_projector_init_scale must be positive')
        if args.protein_projector_mode == 'block_650m_prefix_residual_to35m':
            if projector_esm_dim % 2 != 0:
                raise ValueError('--protein_projector_mode=block_650m_prefix_residual_to35m requires an even ESM dim')
            if args.protein_projector_residual_init_scale <= 0:
                raise ValueError('--protein_projector_residual_init_scale must be positive')
            if not 0.0 <= args.protein_projector_residual_dropout < 1.0:
                raise ValueError('--protein_projector_residual_dropout must be in [0, 1)')
            if args.protein_projector_residual_norm_clip_ratio < 0.0:
                raise ValueError('--protein_projector_residual_norm_clip_ratio must be >= 0')
    if args.weight_decay < 0:
        raise ValueError('--weight_decay must be >= 0')
    if args.lr_warmup_epochs < 0:
        raise ValueError('--lr_warmup_epochs must be >= 0')
    if args.grad_clip_norm < 0:
        raise ValueError('--grad_clip_norm must be >= 0')
    if args.late_checkpoint_start_epoch < 0:
        raise ValueError('--late_checkpoint_start_epoch must be >= 0')
    if args.late_checkpoint_interval < 0:
        raise ValueError('--late_checkpoint_interval must be >= 0')
    missing_sequence_pair_fallback_ids = [
        value.strip() for value in str(args.missing_sequence_pair_fallback_ids).split(',') if value.strip()
    ]

    ### set random seeds
    effective_seed = _seed_everything(_effective_seed(args))
    device = _resolve_device(args.device)
    print(device)
    print('provided_seed:	{}'.format(args.seed))
    print('provided_runseed:	{}'.format(args.runseed))
    print('effective_seed:	{}'.format(effective_seed))
    print('cudnn_deterministic:	{}'.format(torch.backends.cudnn.deterministic))
    print('cudnn_benchmark:	{}'.format(torch.backends.cudnn.benchmark))
    print('protein_embedding_model:	{}'.format(args.protein_embedding_model))
    print('protein_embedding_csv:	{}'.format(embedding_spec.csv_path))
    print('protein_feature_source:	{}'.format(args.protein_feature_source))
    print('pooling_mode:	{}'.format(args.pooling_mode))
    if args.protein_feature_source in {'residue_level_esm2', 'rd_residue_level_esm2'}:
        print('residue_embedding_index_path:	{}'.format(args.residue_embedding_index_path))
    if args.protein_feature_source == 'weighted_pair_embedding':
        print('weighted_embedding_index_path:	{}'.format(args.weighted_embedding_index_path))
        print('pair_manifest_path:	{}'.format(args.pair_manifest_path))
    if args.train_pair_allowlist_csv:
        print('train_pair_allowlist_csv:	{}'.format(args.train_pair_allowlist_csv))
    if args.pooling_mode in {'ppi_weighted_sum', 'ppi_softmax_sum'}:
        print('pair_manifest_path:	{}'.format(args.pair_manifest_path))
        print('ppi_prior_index_path:	{}'.format(args.ppi_prior_index_path))
    if args.pooling_mode == 'ppi_softmax_sum':
        print('pooling_softmax_temperature:	{}'.format(args.pooling_softmax_temperature))
    print('esm_dim:	{}'.format(embedding_spec.esm_dim))
    print('phy_dim:	{}'.format(embedding_spec.phy_dim))
    print('ppi_hidden_dim:	{}'.format(args.ppi_hidden_dim))
    print('protein_projector_mode:	{}'.format(args.protein_projector_mode))
    print('protein_projector_esm_dim:	{}'.format(args.protein_projector_esm_dim or embedding_spec.esm_dim))
    print('protein_projector_phy_dim:	{}'.format(args.protein_projector_phy_dim))
    print('protein_projector_target_esm_dim:	{}'.format(args.protein_projector_target_esm_dim))
    print('protein_projector_gate_init:	{}'.format(args.protein_projector_gate_init))
    print('protein_projector_init_scale:	{}'.format(args.protein_projector_init_scale))
    print('class_weight_mode:	{}'.format(args.class_weight_mode))
    print('weight_decay:	{}'.format(args.weight_decay))
    print('lr_warmup_epochs:	{}'.format(args.lr_warmup_epochs))
    print('lr_warmup_start_factor:	{}'.format(args.lr_warmup_start_factor))
    print('lr_min:	{}'.format(args.lr_min))
    print('grad_clip_norm:	{}'.format(args.grad_clip_norm))
    print('checkpoint_selection_metric:	{}'.format(args.checkpoint_selection_metric))
    print('late_checkpoint_start_epoch:	{}'.format(args.late_checkpoint_start_epoch))
    print('late_checkpoint_interval:	{}'.format(args.late_checkpoint_interval))
    print('missing_sequence_pair_fallback_ids:	{}'.format(missing_sequence_pair_fallback_ids))

    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_path / f'setting_{args.eval_setting}_fold{args.fold}.model'
    best_model_path = checkpoint_dir / 'best_model.pt'
    last_model_path = checkpoint_dir / 'last_model.pt'

    ########## Set up dataset and dataloader ##########
    train_dataset = ModulatorPPIDataset(
        mode='train',
        setting=args.eval_setting,
        fold=args.fold,
        protein_embedding_model=args.protein_embedding_model,
        protein_feature_source=args.protein_feature_source,
        residue_embedding_index_path=args.residue_embedding_index_path,
        pooling_mode=args.pooling_mode,
        pair_manifest_path=args.pair_manifest_path,
        ppi_prior_index_path=args.ppi_prior_index_path,
        weighted_embedding_index_path=args.weighted_embedding_index_path,
        train_pair_allowlist_csv=args.train_pair_allowlist_csv,
        pooling_softmax_temperature=args.pooling_softmax_temperature,
        missing_sequence_pair_fallback_ids=missing_sequence_pair_fallback_ids,
    )
    valid_dataset = ModulatorPPIDataset(
        mode='valid',
        setting=args.eval_setting,
        fold=args.fold,
        protein_embedding_model=args.protein_embedding_model,
        protein_feature_source=args.protein_feature_source,
        residue_embedding_index_path=args.residue_embedding_index_path,
        pooling_mode=args.pooling_mode,
        pair_manifest_path=args.pair_manifest_path,
        ppi_prior_index_path=args.ppi_prior_index_path,
        weighted_embedding_index_path=args.weighted_embedding_index_path,
        train_pair_allowlist_csv=args.train_pair_allowlist_csv,
        pooling_softmax_temperature=args.pooling_softmax_temperature,
        missing_sequence_pair_fallback_ids=missing_sequence_pair_fallback_ids,
    )
    test_dataset = ModulatorPPIDataset(
        mode='test',
        setting=args.eval_setting,
        fold=args.fold,
        protein_embedding_model=args.protein_embedding_model,
        protein_feature_source=args.protein_feature_source,
        residue_embedding_index_path=args.residue_embedding_index_path,
        pooling_mode=args.pooling_mode,
        pair_manifest_path=args.pair_manifest_path,
        ppi_prior_index_path=args.ppi_prior_index_path,
        weighted_embedding_index_path=args.weighted_embedding_index_path,
        train_pair_allowlist_csv=args.train_pair_allowlist_csv,
        pooling_softmax_temperature=args.pooling_softmax_temperature,
        enable_softmax_pooling_export=bool(args.export_softmax_pooling_weights_path),
        missing_sequence_pair_fallback_ids=missing_sequence_pair_fallback_ids,
    )
    print('size of train: {}\tval: {}\ttest: {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    train_dataloader, train_dataloader_debug = _build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        base_seed=effective_seed,
        seed_offset=0,
        name='train',
    )
    valid_dataloader, valid_dataloader_debug = _build_dataloader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        base_seed=effective_seed,
        seed_offset=1,
        name='valid',
    )
    test_dataloader, test_dataloader_debug = _build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        base_seed=effective_seed,
        seed_offset=2,
        name='test',
    )

    ########## Set up model ##########
    PPIMI_model = _build_ppimi_model(args, device, embedding_spec)
    print('MultiPPIMI model\n', PPIMI_model)

    criterion, class_weights = _build_classification_criterion(args, train_dataset, device)
    if class_weights is not None:
        print('class_weights:	{}'.format(class_weights.tolist()))
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(PPIMI_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_name = 'AdamW'
    else:
        optimizer = torch.optim.Adam(PPIMI_model.parameters(), lr=args.learning_rate)
        optimizer_name = 'Adam'
    scheduler, scheduler_info = _build_scheduler(args, optimizer)
    scheduler_info['optimizer'] = optimizer_name
    scheduler_info['weight_decay'] = float(args.weight_decay)
    print('scheduler:	{}'.format(scheduler_info))

    best_model = None
    best_selection_score = float('-inf')
    best_epoch = 0
    best_validation_metrics = None
    best_metrics = None
    last_metrics = None
    last_validation_metrics = None
    validation_history = []
    saved_checkpoints = {
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'late_checkpoints': [],
    }

    train_start_time = time.time()
    for epoch in range(1, 1 + args.epochs):
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch))
        current_lrs = _current_learning_rates(optimizer)
        print('Current LR:	{}'.format(current_lrs))
        train_loss = train(
            PPIMI_model,
            device,
            train_dataloader,
            optimizer,
            criterion,
            grad_clip_norm=args.grad_clip_norm,
        )

        G, P = predicting(PPIMI_model, device, valid_dataloader)
        current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
        current_validation_metrics = _metrics_to_dict(
            (current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels)
        )
        current_selection_score = _selection_score(args.checkpoint_selection_metric, current_validation_metrics)
        last_validation_metrics = current_validation_metrics
        validation_history.append({
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'learning_rates': current_lrs,
            'selection_metric': args.checkpoint_selection_metric,
            'selection_score': float(current_selection_score),
            'validation_metrics': current_validation_metrics,
        })
        print('Val AUC:	{}'.format(current_roc_auc))
        print('Val AUPR:	{}'.format(current_aupr))
        print('Selection score ({})	{}'.format(args.checkpoint_selection_metric, current_selection_score))
        selection_allowed = _checkpoint_selection_allowed(args.checkpoint_selection_metric, epoch, args)
        if selection_allowed and current_selection_score > best_selection_score:
            best_selection_score = current_selection_score
            best_epoch = epoch
            best_model = copy.deepcopy(PPIMI_model)
            best_validation_metrics = current_validation_metrics
            _save_model_state(PPIMI_model, best_model_path)
            print(
                'Selection metric improved at epoch {}	best {}: {}'.format(
                    best_epoch,
                    args.checkpoint_selection_metric,
                    best_selection_score,
                )
            )
        elif not selection_allowed:
            print(
                'Checkpoint selection inactive until epoch {} for {}'.format(
                    args.late_checkpoint_start_epoch,
                    args.checkpoint_selection_metric,
                )
            )
        else:
            print(
                'No improvement since epoch {}	best {}: {}'.format(
                    best_epoch,
                    args.checkpoint_selection_metric,
                    best_selection_score,
                )
            )

        if args.late_checkpoint_interval > 0 and epoch >= args.late_checkpoint_start_epoch:
            if (epoch - args.late_checkpoint_start_epoch) % args.late_checkpoint_interval == 0:
                late_path = checkpoint_dir / f'epoch_{epoch}.pt'
                _save_model_state(PPIMI_model, late_path)
                saved_checkpoints['late_checkpoints'].append({
                    'epoch': int(epoch),
                    'path': str(late_path),
                    'validation_metrics': current_validation_metrics,
                    'selection_score': float(current_selection_score),
                })

        if scheduler is not None:
            scheduler.step()
        print('Took {:.5f}s.'.format(time.time() - start_time))
        print()

    print('Finish training!')
    print('Total training time: {:.5f} hours'.format((time.time() - train_start_time) / 3600))
    start_time = time.time()
    print('Last epoch test results: {}'.format(args.epochs))
    G, P = predicting(PPIMI_model, device, test_dataloader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
    last_metrics = _metrics_to_dict((roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels))
    print('AUC:	{}'.format(roc_auc))
    print('AUPR:	{}'.format(aupr))
    print('precision:	{}'.format(precision))
    print('accuracy:	{}'.format(accuracy))
    print('recall:	{}'.format(recall))
    print('f1:	{}'.format(f1))
    print('specificity:	{}'.format(specificity))
    print('mcc:	{}'.format(mcc))
    print('')
    print('Took {:.5f}s.'.format(time.time() - start_time))
    _save_model_state(PPIMI_model, last_model_path)

    start_time = time.time()
    print('Best epoch test results: {}'.format(best_epoch))
    softmax_pooling_records = []
    if args.export_softmax_pooling_weights_path:
        G, P, softmax_pooling_records = predicting(
            best_model, device, test_dataloader, return_softmax_pooling_weights=True
        )
    else:
        G, P = predicting(best_model, device, test_dataloader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
    best_metrics = _metrics_to_dict((roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels))
    print('AUC:	{}'.format(roc_auc))
    print('AUPR:	{}'.format(aupr))
    print('precision:	{}'.format(precision))
    print('accuracy:	{}'.format(accuracy))
    print('recall:	{}'.format(recall))
    print('f1:	{}'.format(f1))
    print('specificity:	{}'.format(specificity))
    print('mcc:	{}'.format(mcc))
    print('Took {:.5f}s.'.format(time.time() - start_time))
    if args.export_softmax_pooling_weights_path:
        _write_softmax_pooling_exports(args.export_softmax_pooling_weights_path, softmax_pooling_records)
        print('Wrote softmax pooling weight export: {}'.format(args.export_softmax_pooling_weights_path))

    _save_model_state(PPIMI_model, model_path)
    summary_path = out_path / 'run_summary.json'

    dataset_debug_by_split = {
        'train': _dataset_debug_summary(train_dataset),
        'valid': _dataset_debug_summary(valid_dataset),
        'test': _dataset_debug_summary(test_dataset),
    }
    sample_pair_ids_order_by_split = {
        split: (debug or {}).get('sample_pair_ids_order', [])
        for split, debug in dataset_debug_by_split.items()
    }
    sample_pair_id_checksums_by_split = {
        split: (debug or {}).get('sample_pair_ids_order_checksum')
        for split, debug in dataset_debug_by_split.items()
    }
    dataloader_debug = {
        'train': train_dataloader_debug,
        'valid': valid_dataloader_debug,
        'test': test_dataloader_debug,
    }

    run_summary = {
        'status': 'success',
        'protein_embedding_model': args.protein_embedding_model,
        'protein_feature_source': args.protein_feature_source,
        'residue_embedding_index_path': args.residue_embedding_index_path,
        'pooling_mode': args.pooling_mode,
        'pair_manifest_path': args.pair_manifest_path,
        'ppi_prior_index_path': args.ppi_prior_index_path,
        'train_pair_allowlist_csv': args.train_pair_allowlist_csv,
        'protein_embedding_csv': str(embedding_spec.csv_path),
        'provided_seed': int(args.seed),
        'provided_runseed': int(args.runseed),
        'effective_seed': int(effective_seed),
        'ppi_input_dim': int(embedding_spec.paired_dim),
        'ppi_hidden_dim': int(args.ppi_hidden_dim),
        'protein_projector_mode': args.protein_projector_mode,
        'protein_projector_esm_dim': int(args.protein_projector_esm_dim or embedding_spec.esm_dim),
        'protein_projector_phy_dim': int(args.protein_projector_phy_dim),
        'protein_projector_target_esm_dim': (
            None if args.protein_projector_target_esm_dim is None else int(args.protein_projector_target_esm_dim)
        ),
        'protein_projector_gate_init': float(args.protein_projector_gate_init),
        'protein_projector_init_scale': float(args.protein_projector_init_scale),
        'protein_projector_residual_init_scale': float(args.protein_projector_residual_init_scale),
        'protein_projector_residual_dropout': float(args.protein_projector_residual_dropout),
        'protein_projector_residual_norm_clip_ratio': float(args.protein_projector_residual_norm_clip_ratio),
        'protein_projector_prefix_checkpoint_path': args.protein_projector_prefix_checkpoint_path,
        'protein_projector_metadata': (
            PPIMI_model.protein_projector_metadata()
            if hasattr(PPIMI_model, 'protein_projector_metadata')
            else None
        ),
        'class_weight_mode': args.class_weight_mode,
        'class_weights': class_weights.tolist() if class_weights is not None else None,
        'optimizer': optimizer_name,
        'weight_decay': float(args.weight_decay),
        'eval_setting': args.eval_setting,
        'fold': args.fold,
        'best_epoch': int(best_epoch),
        'checkpoint_selection_metric': args.checkpoint_selection_metric,
        'best_selection_score': float(best_selection_score),
        'best_validation_metrics': best_validation_metrics,
        'last_validation_metrics': last_validation_metrics,
        'best_metrics': best_metrics,
        'last_metrics': last_metrics,
        'model_path': str(model_path),
        'scheduler': scheduler_info,
        'grad_clip_norm': float(args.grad_clip_norm),
        'missing_sequence_pair_fallback_ids': missing_sequence_pair_fallback_ids,
        'saved_checkpoints': saved_checkpoints,
        'validation_history': validation_history,
        'dataloader_debug': dataloader_debug,
        'dataset_debug_by_split': dataset_debug_by_split,
        'sample_pair_ids_order_by_split': sample_pair_ids_order_by_split,
        'sample_pair_id_checksums_by_split': sample_pair_id_checksums_by_split,
        'softmax_pooling_weights_export_path': args.export_softmax_pooling_weights_path,
    }
    with summary_path.open('w') as f:
        json.dump(run_summary, f, indent=2)

    if not model_path.exists():
        raise RuntimeError(f'Model file was not created: {model_path}')
    if not summary_path.exists():
        raise RuntimeError(f'Run summary was not created: {summary_path}')
    print('TRAINING_COMPLETED_SUCCESSFULLY')
