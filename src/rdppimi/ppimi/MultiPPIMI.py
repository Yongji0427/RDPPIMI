import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from rdppimi.ppimi.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm


def _logit(probability):
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(probability, dtype=torch.float32))


class BlockEsmTo35MProjector(nn.Module):
    def __init__(
        self,
        ppi_emb_dim,
        output_dim,
        esm_dim,
        phy_dim,
        target_esm_dim,
        gate_init=0.85,
        init_scale=1.0,
    ):
        super(BlockEsmTo35MProjector, self).__init__()
        self.ppi_emb_dim = int(ppi_emb_dim)
        self.output_dim = int(output_dim)
        self.esm_dim = int(esm_dim)
        self.phy_dim = int(phy_dim)
        self.target_esm_dim = int(target_esm_dim)
        self.gate_init = float(gate_init)
        self.init_scale = float(init_scale)
        self.expected_input_dim = 2 * (self.esm_dim + self.phy_dim)
        self.expected_output_dim = 2 * (self.target_esm_dim + self.phy_dim)

        if self.ppi_emb_dim != self.expected_input_dim:
            raise ValueError(
                'block_esm_to_35m expected input dim 2 * (esm_dim + phy_dim) = '
                f'{self.expected_input_dim}, got ppi_emb_dim={self.ppi_emb_dim}'
            )
        if self.output_dim != self.expected_output_dim:
            raise ValueError(
                'block_esm_to_35m expected output dim 2 * (target_esm_dim + phy_dim) = '
                f'{self.expected_output_dim}, got output_dim={self.output_dim}'
            )
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError(f'gate_init must be in (0, 1), got {self.gate_init}')
        if self.init_scale <= 0:
            raise ValueError(f'init_scale must be positive, got {self.init_scale}')

        self.esm_norm = nn.LayerNorm(self.esm_dim)
        self.esm_projector = nn.Linear(self.esm_dim, self.target_esm_dim)
        nn.init.orthogonal_(self.esm_projector.weight)
        self.esm_projector.weight.data.mul_(self.init_scale)
        nn.init.zeros_(self.esm_projector.bias)
        self.gate_a_logit = nn.Parameter(_logit(self.gate_init))
        self.gate_b_logit = nn.Parameter(_logit(self.gate_init))

    def forward(self, ppi_feats):
        if ppi_feats.size(-1) != self.expected_input_dim:
            raise ValueError(
                'block_esm_to_35m input width mismatch: '
                f'expected {self.expected_input_dim}, got {ppi_feats.size(-1)}'
            )
        a_esm_end = self.esm_dim
        a_phy_end = a_esm_end + self.phy_dim
        b_esm_end = a_phy_end + self.esm_dim
        b_phy_end = b_esm_end + self.phy_dim

        a_esm = ppi_feats[..., :a_esm_end]
        a_phy = ppi_feats[..., a_esm_end:a_phy_end]
        b_esm = ppi_feats[..., a_phy_end:b_esm_end]
        b_phy = ppi_feats[..., b_esm_end:b_phy_end]

        gate_a = torch.sigmoid(self.gate_a_logit)
        gate_b = torch.sigmoid(self.gate_b_logit)
        a_projected = gate_a * self.esm_projector(self.esm_norm(a_esm))
        b_projected = gate_b * self.esm_projector(self.esm_norm(b_esm))
        return torch.cat((a_projected, a_phy, b_projected, b_phy), dim=-1)

    def metadata(self):
        return {
            'mode': 'block_esm_to_35m',
            'input_dim': self.ppi_emb_dim,
            'output_dim': self.output_dim,
            'esm_dim': self.esm_dim,
            'phy_dim': self.phy_dim,
            'target_esm_dim': self.target_esm_dim,
            'expected_input_layout': '[A_esm, A_phy, B_esm, B_phy]',
            'expected_input_slices': {
                'A_esm': [0, self.esm_dim],
                'A_phy': [self.esm_dim, self.esm_dim + self.phy_dim],
                'B_esm': [self.esm_dim + self.phy_dim, 2 * self.esm_dim + self.phy_dim],
                'B_phy': [2 * self.esm_dim + self.phy_dim, 2 * (self.esm_dim + self.phy_dim)],
            },
            'shared_ab_projector': True,
            'preserve_phy': True,
            'activation': 'none',
            'projector_dropout': 0.0,
            'gate_type': 'scalar_per_role',
            'gate_init': self.gate_init,
            'gate_a': float(torch.sigmoid(self.gate_a_logit.detach()).cpu()),
            'gate_b': float(torch.sigmoid(self.gate_b_logit.detach()).cpu()),
            'init': 'orthogonal_scaled',
            'init_scale': self.init_scale,
        }


class PrefixResidualEsmTo35MProjector(nn.Module):
    def __init__(
        self,
        ppi_emb_dim,
        output_dim,
        esm_dim,
        phy_dim,
        target_esm_dim,
        gate_init=0.15,
        prefix_init_scale=1.0,
        residual_init_scale=0.05,
        residual_dropout=0.2,
        residual_norm_clip_ratio=0.35,
    ):
        super(PrefixResidualEsmTo35MProjector, self).__init__()
        self.ppi_emb_dim = int(ppi_emb_dim)
        self.output_dim = int(output_dim)
        self.esm_dim = int(esm_dim)
        self.phy_dim = int(phy_dim)
        self.target_esm_dim = int(target_esm_dim)
        self.gate_init = float(gate_init)
        self.prefix_init_scale = float(prefix_init_scale)
        self.residual_init_scale = float(residual_init_scale)
        self.residual_dropout_rate = float(residual_dropout)
        self.residual_norm_clip_ratio = float(residual_norm_clip_ratio)
        self.expected_input_dim = 2 * (self.esm_dim + self.phy_dim)
        self.expected_output_dim = 2 * (self.target_esm_dim + self.phy_dim)

        if self.esm_dim % 2 != 0:
            raise ValueError(f'prefix_residual projector requires even esm_dim, got {self.esm_dim}')
        self.prefix_dim = self.esm_dim // 2
        self.residual_dim = self.esm_dim - self.prefix_dim
        if self.ppi_emb_dim != self.expected_input_dim:
            raise ValueError(
                'block_650m_prefix_residual_to35m expected input dim '
                f'2 * (esm_dim + phy_dim) = {self.expected_input_dim}, got ppi_emb_dim={self.ppi_emb_dim}'
            )
        if self.output_dim != self.expected_output_dim:
            raise ValueError(
                'block_650m_prefix_residual_to35m expected output dim '
                f'2 * (target_esm_dim + phy_dim) = {self.expected_output_dim}, got output_dim={self.output_dim}'
            )
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError(f'gate_init must be in (0, 1), got {self.gate_init}')
        if self.prefix_init_scale <= 0:
            raise ValueError(f'prefix_init_scale must be positive, got {self.prefix_init_scale}')
        if self.residual_init_scale <= 0:
            raise ValueError(f'residual_init_scale must be positive, got {self.residual_init_scale}')
        if not 0.0 <= self.residual_dropout_rate < 1.0:
            raise ValueError(f'residual_dropout must be in [0, 1), got {self.residual_dropout_rate}')
        if self.residual_norm_clip_ratio < 0.0:
            raise ValueError(f'residual_norm_clip_ratio must be >= 0, got {self.residual_norm_clip_ratio}')

        self.prefix_norm = nn.LayerNorm(self.prefix_dim)
        self.residual_norm = nn.LayerNorm(self.residual_dim)
        self.prefix_projector = nn.Linear(self.prefix_dim, self.target_esm_dim)
        self.residual_projector = nn.Linear(self.residual_dim, self.target_esm_dim)
        nn.init.orthogonal_(self.prefix_projector.weight)
        self.prefix_projector.weight.data.mul_(self.prefix_init_scale)
        nn.init.zeros_(self.prefix_projector.bias)
        nn.init.orthogonal_(self.residual_projector.weight)
        self.residual_projector.weight.data.mul_(self.residual_init_scale)
        nn.init.zeros_(self.residual_projector.bias)
        self.residual_dropout = nn.Dropout(self.residual_dropout_rate)
        self.gate_a_logit = nn.Parameter(_logit(self.gate_init))
        self.gate_b_logit = nn.Parameter(_logit(self.gate_init))

    def _split_side(self, side_esm):
        prefix = side_esm[..., :self.prefix_dim]
        residual = side_esm[..., self.prefix_dim:self.esm_dim]
        return prefix, residual

    def _clip_residual(self, residual_projection, prefix_projection):
        if self.residual_norm_clip_ratio == 0.0:
            return residual_projection
        eps = 1e-6
        residual_norm = residual_projection.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        prefix_norm = prefix_projection.detach().norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        max_norm = self.residual_norm_clip_ratio * prefix_norm
        scale = (max_norm / residual_norm).clamp(max=1.0)
        return residual_projection * scale

    def _project_side(self, side_esm, gate):
        prefix, residual = self._split_side(side_esm)
        prefix_projection = self.prefix_projector(self.prefix_norm(prefix))
        residual_projection = self.residual_projector(self.residual_norm(residual))
        residual_projection = self.residual_dropout(residual_projection)
        residual_projection = self._clip_residual(residual_projection, prefix_projection)
        return prefix_projection + gate * residual_projection

    def forward(self, ppi_feats):
        if ppi_feats.size(-1) != self.expected_input_dim:
            raise ValueError(
                'block_650m_prefix_residual_to35m input width mismatch: '
                f'expected {self.expected_input_dim}, got {ppi_feats.size(-1)}'
            )
        a_esm_end = self.esm_dim
        a_phy_end = a_esm_end + self.phy_dim
        b_esm_end = a_phy_end + self.esm_dim
        b_phy_end = b_esm_end + self.phy_dim

        a_esm = ppi_feats[..., :a_esm_end]
        a_phy = ppi_feats[..., a_esm_end:a_phy_end]
        b_esm = ppi_feats[..., a_phy_end:b_esm_end]
        b_phy = ppi_feats[..., b_esm_end:b_phy_end]

        gate_a = torch.sigmoid(self.gate_a_logit)
        gate_b = torch.sigmoid(self.gate_b_logit)
        a_projected = self._project_side(a_esm, gate_a)
        b_projected = self._project_side(b_esm, gate_b)
        return torch.cat((a_projected, a_phy, b_projected, b_phy), dim=-1)

    def load_prefix_projector_from_state_dict(self, state_dict):
        key_map = {
            'protein_projector.esm_norm.weight': self.prefix_norm.weight,
            'protein_projector.esm_norm.bias': self.prefix_norm.bias,
            'protein_projector.esm_projector.weight': self.prefix_projector.weight,
            'protein_projector.esm_projector.bias': self.prefix_projector.bias,
        }
        missing = [key for key in key_map if key not in state_dict]
        if missing:
            raise KeyError(f'prefix checkpoint missing keys: {missing}')
        with torch.no_grad():
            for key, target in key_map.items():
                value = state_dict[key]
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f'prefix checkpoint shape mismatch for {key}: expected {tuple(target.shape)}, got {tuple(value.shape)}'
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def metadata(self):
        return {
            'mode': 'block_650m_prefix_residual_to35m',
            'input_dim': self.ppi_emb_dim,
            'output_dim': self.output_dim,
            'esm_dim': self.esm_dim,
            'phy_dim': self.phy_dim,
            'prefix_dim': self.prefix_dim,
            'residual_dim': self.residual_dim,
            'target_esm_dim': self.target_esm_dim,
            'expected_input_layout': '[A_esm(prefix,residual), A_phy, B_esm(prefix,residual), B_phy]',
            'expected_input_slices': {
                'A_prefix': [0, self.prefix_dim],
                'A_residual': [self.prefix_dim, self.esm_dim],
                'A_phy': [self.esm_dim, self.esm_dim + self.phy_dim],
                'B_prefix': [self.esm_dim + self.phy_dim, self.esm_dim + self.phy_dim + self.prefix_dim],
                'B_residual': [self.esm_dim + self.phy_dim + self.prefix_dim, 2 * self.esm_dim + self.phy_dim],
                'B_phy': [2 * self.esm_dim + self.phy_dim, 2 * (self.esm_dim + self.phy_dim)],
            },
            'shared_prefix_projector': True,
            'shared_residual_projector': True,
            'preserve_phy': True,
            'activation': 'none',
            'residual_dropout': self.residual_dropout_rate,
            'residual_norm_clip_ratio': self.residual_norm_clip_ratio,
            'gate_type': 'scalar_per_role_residual_only',
            'gate_init': self.gate_init,
            'gate_a': float(torch.sigmoid(self.gate_a_logit.detach()).cpu()),
            'gate_b': float(torch.sigmoid(self.gate_b_logit.detach()).cpu()),
            'prefix_init': 'orthogonal_scaled_or_checkpoint',
            'prefix_init_scale': self.prefix_init_scale,
            'residual_init': 'orthogonal_scaled',
            'residual_init_scale': self.residual_init_scale,
        }


class MultiPPIMI(nn.Module):
    def __init__(self, modulator_model, modulator_emb_dim, ppi_emb_dim,
                 h_dim, n_heads,
                 protein_projector_dim=None,
                 protein_projector_mode='linear',
                 protein_esm_dim=None,
                 protein_phy_dim=19,
                 protein_projector_target_esm_dim=None,
                 protein_projector_gate_init=0.85,
                 protein_projector_init_scale=1.0,
                 protein_projector_residual_init_scale=0.05,
                 protein_projector_residual_dropout=0.2,
                 protein_projector_residual_norm_clip_ratio=0.35,
                 output_dim=2, dropout=0.2, device=None, attention=False):
        super(MultiPPIMI, self).__init__()
        self.attention = attention
        self.modulator_emb_dim = modulator_emb_dim
        self.ppi_emb_dim = ppi_emb_dim
        self.modulator_model = modulator_model
        self.protein_projector_mode = str(protein_projector_mode or 'linear')
        self.protein_projector_dim = protein_projector_dim or ppi_emb_dim
        self.protein_esm_dim = protein_esm_dim
        self.protein_phy_dim = protein_phy_dim
        self.protein_projector_target_esm_dim = protein_projector_target_esm_dim
        self.protein_projector_gate_init = protein_projector_gate_init
        self.protein_projector_init_scale = protein_projector_init_scale
        self.protein_projector_residual_init_scale = protein_projector_residual_init_scale
        self.protein_projector_residual_dropout = protein_projector_residual_dropout
        self.protein_projector_residual_norm_clip_ratio = protein_projector_residual_norm_clip_ratio

        self.protein_projector = None
        ban_q_dim = ppi_emb_dim
        if self.protein_projector_mode == 'none':
            if self.protein_projector_dim != ppi_emb_dim:
                raise ValueError(
                    f'protein_projector_mode=none requires protein_projector_dim={ppi_emb_dim}, '
                    f'got {self.protein_projector_dim}'
                )
        elif self.protein_projector_mode == 'linear':
            if self.protein_projector_dim != ppi_emb_dim:
                if self.protein_projector_dim <= 0 or self.protein_projector_dim > ppi_emb_dim:
                    raise ValueError(
                        f'protein_projector_dim must be in [1, {ppi_emb_dim}], got {self.protein_projector_dim}'
                    )
                self.protein_projector = nn.Sequential(
                    nn.LayerNorm(ppi_emb_dim),
                    nn.Linear(ppi_emb_dim, self.protein_projector_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                ban_q_dim = self.protein_projector_dim
        elif self.protein_projector_mode == 'block_esm_to_35m':
            if self.protein_projector_dim <= 0 or self.protein_projector_dim > ppi_emb_dim:
                raise ValueError(
                    f'protein_projector_dim must be in [1, {ppi_emb_dim}], got {self.protein_projector_dim}'
                )
            if protein_esm_dim is None or protein_projector_target_esm_dim is None:
                raise ValueError(
                    'protein_projector_mode=block_esm_to_35m requires '
                    'protein_esm_dim and protein_projector_target_esm_dim'
                )
            self.protein_projector = BlockEsmTo35MProjector(
                ppi_emb_dim=ppi_emb_dim,
                output_dim=self.protein_projector_dim,
                esm_dim=protein_esm_dim,
                phy_dim=protein_phy_dim,
                target_esm_dim=protein_projector_target_esm_dim,
                gate_init=protein_projector_gate_init,
                init_scale=protein_projector_init_scale,
            )
            ban_q_dim = self.protein_projector_dim
        elif self.protein_projector_mode == 'block_650m_prefix_residual_to35m':
            if self.protein_projector_dim <= 0 or self.protein_projector_dim > ppi_emb_dim:
                raise ValueError(
                    f'protein_projector_dim must be in [1, {ppi_emb_dim}], got {self.protein_projector_dim}'
                )
            if protein_esm_dim is None or protein_projector_target_esm_dim is None:
                raise ValueError(
                    'protein_projector_mode=block_650m_prefix_residual_to35m requires '
                    'protein_esm_dim and protein_projector_target_esm_dim'
                )
            self.protein_projector = PrefixResidualEsmTo35MProjector(
                ppi_emb_dim=ppi_emb_dim,
                output_dim=self.protein_projector_dim,
                esm_dim=protein_esm_dim,
                phy_dim=protein_phy_dim,
                target_esm_dim=protein_projector_target_esm_dim,
                gate_init=protein_projector_gate_init,
                prefix_init_scale=protein_projector_init_scale,
                residual_init_scale=protein_projector_residual_init_scale,
                residual_dropout=protein_projector_residual_dropout,
                residual_norm_clip_ratio=protein_projector_residual_norm_clip_ratio,
            )
            ban_q_dim = self.protein_projector_dim
        else:
            raise ValueError(
                "protein_projector_mode must be one of {'none', 'linear', 'block_esm_to_35m', 'block_650m_prefix_residual_to35m'}, "
                f'got {self.protein_projector_mode}'
            )

        ##### bilinear attention #####
        self.bcn = weight_norm(
            BANLayer(v_dim=modulator_emb_dim, q_dim=ban_q_dim, h_dim=h_dim, h_out=n_heads, k=3),
            name='h_mat', dim=None)

        self.fc1 = nn.Linear(h_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, output_dim)
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def load_protein_projector_prefix_checkpoint(self, checkpoint_path, map_location=None):
        if not hasattr(self.protein_projector, 'load_prefix_projector_from_state_dict'):
            raise ValueError(
                f'protein_projector_mode={self.protein_projector_mode} does not support prefix checkpoint loading'
            )
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.protein_projector.load_prefix_projector_from_state_dict(state_dict)

    def forward(self, modulator, rdkit_descriptors, ppi_feats):
        modulator_node_repr = self.modulator_model(modulator)
        modulator_repr = self.pool(modulator_node_repr, modulator.batch)
        modulator_repr = torch.cat((modulator_repr, rdkit_descriptors), 1)

        if self.protein_projector is not None:
            ppi_feats = self.protein_projector(ppi_feats)

        x, att = self.bcn(modulator_repr, ppi_feats)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x

    def protein_projector_metadata(self):
        if self.protein_projector is None:
            return {
                'mode': self.protein_projector_mode,
                'input_dim': self.ppi_emb_dim,
                'output_dim': self.ppi_emb_dim,
                'enabled': False,
            }
        if hasattr(self.protein_projector, 'metadata'):
            metadata = self.protein_projector.metadata()
            metadata['enabled'] = True
            return metadata
        return {
            'mode': self.protein_projector_mode,
            'input_dim': self.ppi_emb_dim,
            'output_dim': self.protein_projector_dim,
            'enabled': True,
            'activation': 'ReLU',
            'projector_dropout': float(self.dropout.p),
        }
