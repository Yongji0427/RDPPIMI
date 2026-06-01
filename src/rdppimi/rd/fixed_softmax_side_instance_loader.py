from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rdppimi.ppimi.protein_embedding_specs import get_embedding_spec


CANONICAL_FIXED_SOFTMAX_ASSET_ROOT = Path(
    "./fixed_softmax_score_pooled_assets/t0p7"
)
FIXED_SOFTMAX_SCALES: Tuple[str, ...] = ("8M", "35M", "150M", "650M", "3B")
EXPECTED_PAIR_COUNT = 118
EXPECTED_SIDE_INSTANCE_COUNT = 236
VALID_STATUSES = {"success", "cached"}


@dataclass(frozen=True)
class SideInstanceRowKey:
    pair_id: str
    side: str
    protein_id: str
    partner_protein_id: str
    registry_proteinA_id: str
    registry_proteinB_id: str

    def as_tuple(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.pair_id,
            self.side,
            self.protein_id,
            self.partner_protein_id,
            self.registry_proteinA_id,
            self.registry_proteinB_id,
        )

    def as_dict(self, row_index: int) -> Dict[str, object]:
        return {
            "row_index": row_index,
            "pair_id": self.pair_id,
            "side": self.side,
            "protein_id": self.protein_id,
            "partner_protein_id": self.partner_protein_id,
            "registry_proteinA_id": self.registry_proteinA_id,
            "registry_proteinB_id": self.registry_proteinB_id,
        }


@dataclass(frozen=True)
class FixedSoftmaxSideInstanceBundle:
    asset_root: Path
    row_keys: List[SideInstanceRowKey]
    matrices: Dict[str, np.ndarray]
    registry_paths: Dict[str, Path]

    @property
    def pair_count(self) -> int:
        return len(self.row_keys) // 2

    @property
    def side_instance_count(self) -> int:
        return len(self.row_keys)

    def shape_map(self) -> Dict[str, Tuple[int, int]]:
        return {scale: tuple(matrix.shape) for scale, matrix in self.matrices.items()}

    def row_key_format(self) -> str:
        return "row_index,pair_id,side,protein_id,partner_protein_id,registry_proteinA_id,registry_proteinB_id"

    def row_keys_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([row_key.as_dict(idx) for idx, row_key in enumerate(self.row_keys)])


def _registry_path(asset_root: Path, scale: str) -> Path:
    return asset_root / "indices" / f"fixed_softmax_scores_registry_t0p7_{scale}.csv"


def _validate_required_columns(df: pd.DataFrame, registry_path: Path) -> None:
    required_columns = {
        "pair_id",
        "proteinA_id",
        "proteinB_id",
        "weighted_embedding_A_path",
        "weighted_embedding_B_path",
        "status",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Registry missing required columns {sorted(missing)}: {registry_path}")


def _load_registry(asset_root: Path, scale: str, expected_pair_count: int) -> pd.DataFrame:
    registry_path = _registry_path(asset_root, scale)
    if not registry_path.exists():
        raise FileNotFoundError(f"Fixed softmax registry not found for scale={scale}: {registry_path}")

    registry_df = pd.read_csv(registry_path)
    _validate_required_columns(registry_df, registry_path)

    selected = registry_df[registry_df["status"].astype(str).isin(VALID_STATUSES)].copy()
    if "validation_status" in selected.columns:
        selected = selected[selected["validation_status"].fillna("").astype(str) == "ok"].copy()
    if "model_scale" in selected.columns:
        selected = selected[selected["model_scale"].astype(str) == scale].copy()
    elif "model_key" in selected.columns:
        selected = selected[selected["model_key"].astype(str) == scale].copy()

    if selected.empty:
        raise ValueError(f"No usable fixed softmax rows found for scale={scale}: {registry_path}")

    if len(selected) != expected_pair_count:
        raise ValueError(
            f"Expected {expected_pair_count} pair rows for scale={scale}, got {len(selected)}: {registry_path}"
        )

    return selected.reset_index(drop=True)


def _load_vector(vector_path: Path, scale: str, side_label: str) -> np.ndarray:
    if not vector_path.exists():
        raise FileNotFoundError(f"Missing fixed softmax vector for {scale} {side_label}: {vector_path}")

    vector = np.load(vector_path)
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D fixed softmax vector for {scale} {side_label}, got shape={vector.shape}: {vector_path}")

    expected_dim = get_embedding_spec(scale).esm_dim
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"Unexpected fixed softmax vector dim for {scale} {side_label}: expected {expected_dim}, got {vector.shape[0]}: {vector_path}"
        )

    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite values detected in fixed softmax vector for {scale} {side_label}: {vector_path}")

    return vector.astype(np.float32, copy=False)


def _build_scale_side_instances(registry_df: pd.DataFrame, scale: str) -> Tuple[List[SideInstanceRowKey], np.ndarray]:
    row_keys: List[SideInstanceRowKey] = []
    vectors: List[np.ndarray] = []

    for row in registry_df.itertuples(index=False):
        pair_id = str(row.pair_id).strip()
        protein_a = str(row.proteinA_id).strip()
        protein_b = str(row.proteinB_id).strip()

        vector_a = _load_vector(Path(str(row.weighted_embedding_A_path).strip()), scale, f"{pair_id}:A")
        vector_b = _load_vector(Path(str(row.weighted_embedding_B_path).strip()), scale, f"{pair_id}:B")

        row_keys.append(
            SideInstanceRowKey(
                pair_id=pair_id,
                side="A",
                protein_id=protein_a,
                partner_protein_id=protein_b,
                registry_proteinA_id=protein_a,
                registry_proteinB_id=protein_b,
            )
        )
        vectors.append(vector_a)

        row_keys.append(
            SideInstanceRowKey(
                pair_id=pair_id,
                side="B",
                protein_id=protein_b,
                partner_protein_id=protein_a,
                registry_proteinA_id=protein_a,
                registry_proteinB_id=protein_b,
            )
        )
        vectors.append(vector_b)

    matrix = np.stack(vectors, axis=0)
    expected_shape = (EXPECTED_SIDE_INSTANCE_COUNT, get_embedding_spec(scale).esm_dim)
    if matrix.shape != expected_shape:
        raise ValueError(f"Unexpected side-instance matrix shape for {scale}: expected {expected_shape}, got {matrix.shape}")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D side-instance matrix for {scale}, got ndim={matrix.ndim}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values detected in side-instance matrix for {scale}")

    return row_keys, matrix


def load_fixed_softmax_side_instance_bundle(
    asset_root: Path = CANONICAL_FIXED_SOFTMAX_ASSET_ROOT,
    scales: Sequence[str] = FIXED_SOFTMAX_SCALES,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
) -> FixedSoftmaxSideInstanceBundle:
    asset_root = Path(asset_root).resolve()
    if not asset_root.exists():
        raise FileNotFoundError(f"Canonical fixed softmax asset root does not exist: {asset_root}")

    matrices: Dict[str, np.ndarray] = {}
    registry_paths: Dict[str, Path] = {}
    base_row_keys: List[SideInstanceRowKey] | None = None

    for scale in scales:
        registry_df = _load_registry(asset_root, scale, expected_pair_count=expected_pair_count)
        row_keys, matrix = _build_scale_side_instances(registry_df, scale)

        if base_row_keys is None:
            base_row_keys = row_keys
        else:
            if [row_key.as_tuple() for row_key in row_keys] != [row_key.as_tuple() for row_key in base_row_keys]:
                raise ValueError(f"Row key mismatch detected between scales; failed at scale={scale}")

        matrices[scale] = matrix
        registry_paths[scale] = _registry_path(asset_root, scale)

    if base_row_keys is None:
        raise RuntimeError("No fixed softmax side-instance rows were loaded")
    if len(base_row_keys) != EXPECTED_SIDE_INSTANCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SIDE_INSTANCE_COUNT} side-instances, got {len(base_row_keys)}"
        )

    return FixedSoftmaxSideInstanceBundle(
        asset_root=asset_root,
        row_keys=base_row_keys,
        matrices=matrices,
        registry_paths=registry_paths,
    )
