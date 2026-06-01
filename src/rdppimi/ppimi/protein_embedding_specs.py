from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    csv_path: Path
    ready_marker_path: Path
    raw_embedding_dir: Path
    esm_dim: int
    phy_dim: int
    per_protein_dim: int
    paired_dim: int


PROTEIN_PHY_CSV_PATH = repo_path("data", "features", "protein_phy.csv")


def _make_spec(key: str, csv_relpath: str, raw_dir_relpath: str, esm_dim: int, phy_dim: int = 19) -> EmbeddingSpec:
    csv_path = repo_path(*csv_relpath.split("/"))
    ready_marker_path = csv_path.with_suffix(".ready")
    raw_embedding_dir = repo_path(*raw_dir_relpath.split("/"))
    per_protein_dim = esm_dim + phy_dim
    paired_dim = 2 * per_protein_dim
    return EmbeddingSpec(
        key=key,
        csv_path=csv_path,
        ready_marker_path=ready_marker_path,
        raw_embedding_dir=raw_embedding_dir,
        esm_dim=esm_dim,
        phy_dim=phy_dim,
        per_protein_dim=per_protein_dim,
        paired_dim=paired_dim,
    )


PROTEIN_EMBEDDING_SPECS = {
    "8M": _make_spec(
        "8M",
        "data/features_multiscale/protein_esm2_8M.csv",
        "data/esm2_multiscale_embeddings/esm2_t6_8M_UR50D",
        320,
    ),
    "35M": _make_spec(
        "35M",
        "data/features_multiscale/protein_esm2_35M.csv",
        "data/esm2_multiscale_embeddings/esm2_t12_35M_UR50D",
        480,
    ),
    "150M": _make_spec(
        "150M",
        "data/features/protein_esm2.csv",
        "data/esm2_multiscale_embeddings/esm2_t30_150M_UR50D",
        640,
    ),
    "650M": _make_spec(
        "650M",
        "data/features_multiscale/protein_esm2_650M.csv",
        "data/esm2_multiscale_embeddings/esm2_t33_650M_UR50D",
        1280,
    ),
    "3B": _make_spec(
        "3B",
        "data/features_multiscale/protein_esm2_3B.csv",
        "data/esm2_multiscale_embeddings/esm2_t36_3B_UR50D",
        2560,
    ),
    "mean_concat_8M_35M": _make_spec(
        "mean_concat_8M_35M",
        "data/features_multiscale/protein_esm2_mean_concat_8M_35M.csv",
        "data/esm2_multiscale_embeddings",
        800,
    ),
    "mean_concat_8M_35M_150M": _make_spec(
        "mean_concat_8M_35M_150M",
        "data/features_multiscale/protein_esm2_mean_concat_8M_35M_150M.csv",
        "data/esm2_multiscale_embeddings",
        1440,
    ),
    "mean_concat_8M_35M_150M_650M": _make_spec(
        "mean_concat_8M_35M_150M_650M",
        "data/features_multiscale/protein_esm2_mean_concat_8M_35M_150M_650M.csv",
        "data/esm2_multiscale_embeddings",
        2720,
    ),
    "mean_concat_5scale": _make_spec(
        "mean_concat_5scale",
        "data/features_multiscale/protein_esm2_mean_concat_5scale.csv",
        "data/esm2_multiscale_embeddings",
        5280,
    ),
    "softmax_concat_8M_35M": _make_spec(
        "softmax_concat_8M_35M",
        "data/features_multiscale/softmax_concat_registry.csv",
        "data/features_multiscale/softmax_concat_cache",
        800,
    ),
    "softmax_concat_8M_35M_150M": _make_spec(
        "softmax_concat_8M_35M_150M",
        "data/features_multiscale/softmax_concat_registry.csv",
        "data/features_multiscale/softmax_concat_cache",
        1440,
    ),
    "softmax_concat_8M_35M_150M_650M": _make_spec(
        "softmax_concat_8M_35M_150M_650M",
        "data/features_multiscale/softmax_concat_registry.csv",
        "data/features_multiscale/softmax_concat_cache",
        2720,
    ),
    "softmax_concat_5scale": _make_spec(
        "softmax_concat_5scale",
        "data/features_multiscale/softmax_concat_registry.csv",
        "data/features_multiscale/softmax_concat_cache",
        5280,
    ),
}


def get_embedding_spec(model_key: str) -> EmbeddingSpec:
    try:
        return PROTEIN_EMBEDDING_SPECS[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(PROTEIN_EMBEDDING_SPECS))
        raise KeyError(f"Unknown protein embedding model {model_key!r}; expected one of: {valid}") from exc


def load_canonical_protein_ids() -> List[str]:
    if not PROTEIN_PHY_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing protein phy feature file: {PROTEIN_PHY_CSV_PATH}")

    with PROTEIN_PHY_CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "uniprot_id" not in reader.fieldnames:
            raise ValueError(f"{PROTEIN_PHY_CSV_PATH} missing required uniprot_id column")
        ids = [row["uniprot_id"] for row in reader]

    if len(ids) != len(set(ids)):
        raise ValueError(f"{PROTEIN_PHY_CSV_PATH} contains duplicate uniprot_id values")

    return ids
