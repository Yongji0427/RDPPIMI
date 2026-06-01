from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_MODEL_SCALES: Tuple[str, ...] = ("8M", "35M", "150M", "650M", "3B")
MODEL_SCALE_ORDER: Dict[str, int] = {scale: idx for idx, scale in enumerate(EXPECTED_MODEL_SCALES)}

CANONICAL_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "canonical_pair_id",
    "model_scale",
    "proteinA_id",
    "proteinB_id",
    "sample_protein1_id",
    "sample_protein2_id",
    "sample_to_canonical_swapped",
    "ab_mapping_status",
    "ab_mapping_basis",
    "proteinA_scores_path",
    "proteinB_scores_path",
    "proteinA_embedding_metadata_path",
    "proteinB_embedding_metadata_path",
    "proteinA_scores_exists",
    "proteinB_scores_exists",
    "proteinA_embedding_exists",
    "proteinB_embedding_exists",
    "availability_status",
    "fixed_validation_status",
    "fixed_pooling_mode",
    "fixed_pooling_input_semantics",
    "fixed_temperature",
)

STEP3_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "canonical_pair_id",
    "model_scale",
    "pooling_mode",
    "protein_side",
    "canonical_protein_id",
    "alignment_mode",
    "residue_index",
    "residue_score_raw",
    "residue_importance",
)

STEP4_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "canonical_pair_id",
    "model_scale",
    "pooling_mode",
    "pooling_temperature",
    "protein_side",
    "canonical_protein_id",
    "alignment_mode",
    "residue_index",
    "residue_score_raw",
    "residue_importance",
    "ppi_score",
    "ppi_mask_topk",
    "ppi_mask_threshold",
)

PROTEINSHAKE_INDEX_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "proteinA_id",
    "proteinB_id",
    "lenA",
    "lenB",
    "cache_dir",
    "residue_scores_A_path",
    "residue_scores_B_path",
    "residue_weights_A_path",
    "residue_weights_B_path",
    "metadata_path",
    "status",
    "error",
)


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def alignment_mode_from_swapped(swapped_value) -> str:
    return "aligned_via_swap" if to_bool(swapped_value) else "direct"


def normalize_model_scale_key(value: str) -> Tuple[int, str]:
    scale = str(value)
    return MODEL_SCALE_ORDER.get(scale, len(MODEL_SCALE_ORDER)), scale


def softmax_normalize(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax temperature must be > 0, got {temperature}")
    logits = scores.astype(np.float64) / float(temperature)
    max_logit = float(np.max(logits))
    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    denom = float(exp_shifted.sum(dtype=np.float64))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(f"invalid softmax denominator: {denom}")
    weights = exp_shifted / denom
    if not np.isfinite(weights).all():
        raise ValueError("softmax produced non-finite weights")
    if (weights < -1e-7).any():
        raise ValueError("softmax produced negative weights")
    weights = np.clip(weights, 0.0, None)
    clipped_sum = float(weights.sum(dtype=np.float64))
    if clipped_sum <= 0.0 or not np.isfinite(clipped_sum):
        raise ValueError(f"invalid clipped softmax sum: {clipped_sum}")
    weights = weights / clipped_sum
    correction = 1.0 - float(weights.sum(dtype=np.float64))
    weights[-1] += correction
    weights = weights.astype(np.float32, copy=False)
    weight_sum = float(weights.sum(dtype=np.float64))
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(f"softmax weight sum != 1, got {weight_sum}")
    return weights


def read_exportable_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        manifest_path,
        usecols=list(CANONICAL_MANIFEST_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    df = df.drop_duplicates(subset=["sample_id", "model_scale"], keep="first").copy()
    df["alignment_mode"] = df["sample_to_canonical_swapped"].map(alignment_mode_from_swapped)
    df["proteinA_scores_exists_bool"] = df["proteinA_scores_exists"].map(to_bool)
    df["proteinB_scores_exists_bool"] = df["proteinB_scores_exists"].map(to_bool)
    df["proteinA_embedding_exists_bool"] = df["proteinA_embedding_exists"].map(to_bool)
    df["proteinB_embedding_exists_bool"] = df["proteinB_embedding_exists"].map(to_bool)
    return df


def filter_analyzable_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Counter]:
    skipped = Counter()
    keep_rows: List[dict] = []
    for row in df.to_dict(orient="records"):
        reason = None
        if row["canonical_pair_id"] == "":
            reason = "missing_canonical_pair_id"
        elif row["availability_status"] != "ready_fixed_softmax":
            reason = f"availability_status={row['availability_status'] or 'missing'}"
        elif row["fixed_validation_status"] not in {"", "ok"}:
            reason = f"fixed_validation_status={row['fixed_validation_status']}"
        elif row["fixed_pooling_mode"] not in {"", "ppi_softmax_sum"}:
            reason = f"fixed_pooling_mode={row['fixed_pooling_mode']}"
        elif row["fixed_pooling_input_semantics"] not in {"", "residue_scores"}:
            reason = f"fixed_pooling_input_semantics={row['fixed_pooling_input_semantics']}"
        elif not row["proteinA_scores_exists_bool"] or not row["proteinB_scores_exists_bool"]:
            reason = "missing_residue_scores"
        elif not row["proteinA_embedding_exists_bool"] or not row["proteinB_embedding_exists_bool"]:
            reason = "missing_embedding_metadata"

        if reason is None:
            keep_rows.append(row)
        else:
            skipped[reason] += 1

    return pd.DataFrame(keep_rows), skipped


def parse_sample_ids(sample_ids_arg: str) -> Optional[List[str]]:
    if not sample_ids_arg:
        return None
    if "," in sample_ids_arg:
        values = [item.strip() for item in sample_ids_arg.split(",") if item.strip()]
    else:
        path = Path(sample_ids_arg)
        try:
            path_exists = path.exists()
        except OSError:
            path_exists = False
        if path_exists:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, dtype=str)
                if "sample_id" in df.columns:
                    values = df["sample_id"].astype(str).tolist()
                elif df.shape[1] == 1:
                    values = df.iloc[:, 0].astype(str).tolist()
                else:
                    raise ValueError(f"Sample-id CSV must have a sample_id column or exactly one column: {path}")
            else:
                values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            values = [sample_ids_arg.strip()] if sample_ids_arg.strip() else []
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def select_sample_ids(
    analyzable_df: pd.DataFrame,
    sample_ids: Optional[Sequence[str]],
    sample_limit: Optional[int],
    require_all_scales: bool,
    expected_model_scales: Sequence[str],
) -> Tuple[List[str], Counter]:
    skipped = Counter()
    expected_scales = set(expected_model_scales)
    grouped = analyzable_df.groupby("sample_id")["model_scale"].apply(lambda s: set(map(str, s))).to_dict()

    if sample_ids is not None:
        ordered_ids = [sample_id for sample_id in sample_ids if sample_id in grouped]
        for sample_id in sample_ids:
            if sample_id not in grouped:
                skipped["sample_id_not_exportable"] += 1
        if require_all_scales:
            filtered: List[str] = []
            for sample_id in ordered_ids:
                if grouped[sample_id] >= expected_scales:
                    filtered.append(sample_id)
                else:
                    skipped["missing_model_scale"] += 1
            ordered_ids = filtered
    else:
        ordered_ids = sorted(grouped)
        if require_all_scales:
            filtered = []
            for sample_id in ordered_ids:
                if grouped[sample_id] >= expected_scales:
                    filtered.append(sample_id)
                else:
                    skipped["missing_model_scale"] += 1
            ordered_ids = filtered

    if sample_limit is not None:
        ordered_ids = ordered_ids[:sample_limit]
    return ordered_ids, skipped


class ArrayCache:
    def __init__(self) -> None:
        self.score_cache: Dict[str, np.ndarray] = {}
        self.weight_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self.meta_length_cache: Dict[str, int] = {}

    def load_scores(self, path: str) -> np.ndarray:
        if path not in self.score_cache:
            array = np.load(path)
            if array.ndim != 1:
                raise ValueError(f"Residue score must be 1D: {path}")
            if not np.isfinite(array).all():
                raise ValueError(f"Residue score contains non-finite values: {path}")
            self.score_cache[path] = array.astype(np.float32, copy=False)
        return self.score_cache[path]

    def load_weights(self, path: str, temperature: float) -> np.ndarray:
        key = (path, float(temperature))
        if key not in self.weight_cache:
            self.weight_cache[key] = softmax_normalize(self.load_scores(path), temperature)
        return self.weight_cache[key]

    def load_sequence_length(self, metadata_path: str) -> int:
        if metadata_path not in self.meta_length_cache:
            data = json.loads(Path(metadata_path).read_text())
            length = data.get("sequence_length")
            if length is None:
                raise KeyError(f"sequence_length missing from embedding metadata: {metadata_path}")
            self.meta_length_cache[metadata_path] = int(length)
        return self.meta_length_cache[metadata_path]


def warning_record(
    sample_id: str,
    canonical_pair_id: str,
    model_scale: str,
    protein_side: str,
    warning_type: str,
    expected_length,
    observed_length,
    action: str,
    details: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "canonical_pair_id": canonical_pair_id,
        "model_scale": model_scale,
        "protein_side": protein_side,
        "warning_type": warning_type,
        "expected_length": "" if expected_length is None else expected_length,
        "observed_length": "" if observed_length is None else observed_length,
        "action": action,
        "details": details,
    }


def build_side_frame(
    *,
    sample_id: str,
    canonical_pair_id: str,
    model_scale: str,
    pooling_mode: str,
    pooling_temperature: float,
    protein_side: str,
    canonical_protein_id: str,
    alignment_mode: str,
    scores: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    residue_count = int(scores.shape[0])
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "canonical_pair_id": canonical_pair_id,
            "model_scale": model_scale,
            "pooling_mode": pooling_mode,
            "pooling_temperature": float(pooling_temperature),
            "protein_side": protein_side,
            "canonical_protein_id": canonical_protein_id,
            "alignment_mode": alignment_mode,
            "residue_index": np.arange(residue_count, dtype=np.int64),
            "residue_score_raw": scores.astype(np.float32, copy=False),
            "residue_importance": weights.astype(np.float32, copy=False),
        }
    )


def run_residue_importance_export(
    *,
    manifest_path: Path,
    output_root: Path,
    warnings_csv_path: Path,
    sample_ids: Optional[Sequence[str]],
    sample_limit: Optional[int],
    require_all_scales: bool,
    expected_model_scales: Sequence[str],
    clean_output: bool,
) -> dict:
    if clean_output and output_root.exists():
        for child in output_root.glob("*"):
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            elif child.is_file():
                child.unlink()

    output_root.mkdir(parents=True, exist_ok=True)
    warnings_csv_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = read_exportable_manifest(manifest_path)
    analyzable_df, filter_skips = filter_analyzable_rows(manifest_df)
    selected_sample_ids, selection_skips = select_sample_ids(
        analyzable_df,
        sample_ids=sample_ids,
        sample_limit=sample_limit,
        require_all_scales=require_all_scales,
        expected_model_scales=expected_model_scales,
    )

    selected_df = analyzable_df[analyzable_df["sample_id"].isin(selected_sample_ids)].copy()
    if selected_df.empty:
        raise ValueError("No analyzable sample rows selected for residue export.")

    selected_df["__model_sort"] = selected_df["model_scale"].map(lambda value: normalize_model_scale_key(value)[0])
    selected_df = selected_df.sort_values(["sample_id", "__model_sort", "model_scale"]).drop(columns=["__model_sort"])

    array_cache = ArrayCache()
    warnings: List[dict] = []
    skip_reasons = Counter()
    exported_csv_count = 0
    exported_sample_scale_cases = 0
    exported_sample_ids: List[str] = []
    exported_by_scale = Counter()
    exported_by_alignment = Counter()

    for sample_id, sample_group in selected_df.groupby("sample_id", sort=False):
        sample_rows = list(sample_group.to_dict(orient="records"))
        sample_invalid = False
        row_payloads: List[dict] = []

        for row in sample_rows:
            canonical_pair_id = row["canonical_pair_id"]
            model_scale = row["model_scale"]
            try:
                score_a = array_cache.load_scores(row["proteinA_scores_path"])
                score_b = array_cache.load_scores(row["proteinB_scores_path"])
                meta_len_a = array_cache.load_sequence_length(row["proteinA_embedding_metadata_path"])
                meta_len_b = array_cache.load_sequence_length(row["proteinB_embedding_metadata_path"])
            except Exception as exc:
                warnings.append(
                    warning_record(
                        sample_id,
                        canonical_pair_id,
                        model_scale,
                        "both",
                        "load_error",
                        None,
                        None,
                        "skip_sample",
                        str(exc),
                    )
                )
                skip_reasons["load_error"] += 1
                sample_invalid = True
                continue

            if len(score_a) != meta_len_a:
                warnings.append(
                    warning_record(
                        sample_id,
                        canonical_pair_id,
                        model_scale,
                        "A",
                        "embedding_score_length_mismatch",
                        meta_len_a,
                        len(score_a),
                        "skip_sample",
                        "proteinA score length does not match residue embedding metadata length",
                    )
                )
                skip_reasons["embedding_score_length_mismatch"] += 1
                sample_invalid = True

            if len(score_b) != meta_len_b:
                warnings.append(
                    warning_record(
                        sample_id,
                        canonical_pair_id,
                        model_scale,
                        "B",
                        "embedding_score_length_mismatch",
                        meta_len_b,
                        len(score_b),
                        "skip_sample",
                        "proteinB score length does not match residue embedding metadata length",
                    )
                )
                skip_reasons["embedding_score_length_mismatch"] += 1
                sample_invalid = True

            row_payloads.append(
                {
                    "row": row,
                    "score_a": score_a,
                    "score_b": score_b,
                    "len_a": int(len(score_a)),
                    "len_b": int(len(score_b)),
                }
            )

        if sample_invalid:
            continue

        len_a_set = {payload["len_a"] for payload in row_payloads}
        len_b_set = {payload["len_b"] for payload in row_payloads}
        if len(len_a_set) > 1:
            for payload in row_payloads:
                warnings.append(
                    warning_record(
                        sample_id,
                        payload["row"]["canonical_pair_id"],
                        payload["row"]["model_scale"],
                        "A",
                        "cross_scale_length_mismatch",
                        min(len_a_set),
                        payload["len_a"],
                        "skip_sample",
                        "proteinA residue counts differ across model scales for the same sample_id",
                    )
                )
            skip_reasons["cross_scale_length_mismatch"] += 1
            continue

        if len(len_b_set) > 1:
            for payload in row_payloads:
                warnings.append(
                    warning_record(
                        sample_id,
                        payload["row"]["canonical_pair_id"],
                        payload["row"]["model_scale"],
                        "B",
                        "cross_scale_length_mismatch",
                        min(len_b_set),
                        payload["len_b"],
                        "skip_sample",
                        "proteinB residue counts differ across model scales for the same sample_id",
                    )
                )
            skip_reasons["cross_scale_length_mismatch"] += 1
            continue

        exported_sample_ids.append(sample_id)
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for payload in row_payloads:
            row = payload["row"]
            model_scale = row["model_scale"]
            canonical_pair_id = row["canonical_pair_id"]
            alignment_mode = row["alignment_mode"]
            pooling_mode = row["fixed_pooling_mode"] or "ppi_softmax_sum"
            temperature_text = row["fixed_temperature"] or "0.7"
            temperature = float(temperature_text)
            score_a = payload["score_a"]
            score_b = payload["score_b"]
            weights_a = array_cache.load_weights(row["proteinA_scores_path"], temperature)
            weights_b = array_cache.load_weights(row["proteinB_scores_path"], temperature)

            df_a = build_side_frame(
                sample_id=sample_id,
                canonical_pair_id=canonical_pair_id,
                model_scale=model_scale,
                pooling_mode=pooling_mode,
                pooling_temperature=temperature,
                protein_side="A",
                canonical_protein_id=row["proteinA_id"],
                alignment_mode=alignment_mode,
                scores=score_a,
                weights=weights_a,
            )
            df_b = build_side_frame(
                sample_id=sample_id,
                canonical_pair_id=canonical_pair_id,
                model_scale=model_scale,
                pooling_mode=pooling_mode,
                pooling_temperature=temperature,
                protein_side="B",
                canonical_protein_id=row["proteinB_id"],
                alignment_mode=alignment_mode,
                scores=score_b,
                weights=weights_b,
            )

            out_a = sample_dir / f"{model_scale}_A.csv"
            out_b = sample_dir / f"{model_scale}_B.csv"
            df_a.to_csv(out_a, index=False)
            df_b.to_csv(out_b, index=False)

            exported_sample_scale_cases += 1
            exported_csv_count += 2
            exported_by_scale[model_scale] += 1
            exported_by_alignment[alignment_mode] += 1

    warning_columns = [
        "sample_id",
        "canonical_pair_id",
        "model_scale",
        "protein_side",
        "warning_type",
        "expected_length",
        "observed_length",
        "action",
        "details",
    ]
    warnings_df = pd.DataFrame(warnings, columns=warning_columns)
    warnings_df.to_csv(warnings_csv_path, index=False)

    return {
        "selected_sample_ids": selected_sample_ids,
        "selected_sample_count": len(selected_sample_ids),
        "selected_sample_scale_rows": int(len(selected_df)),
        "exported_unique_sample_count": len(exported_sample_ids),
        "exported_sample_scale_cases": int(exported_sample_scale_cases),
        "exported_csv_count": int(exported_csv_count),
        "exported_by_scale": dict(sorted(exported_by_scale.items(), key=lambda item: normalize_model_scale_key(item[0]))),
        "exported_by_alignment": dict(sorted(exported_by_alignment.items())),
        "filter_skip_counts": dict(sorted(filter_skips.items())),
        "selection_skip_counts": dict(sorted(selection_skips.items())),
        "runtime_skip_counts": dict(sorted(skip_reasons.items())),
        "warning_count": int(len(warnings_df)),
        "warning_csv_path": str(warnings_csv_path),
    }


def write_export_check_md(
    *,
    path: Path,
    manifest_path: Path,
    output_root: Path,
    warnings_csv_path: Path,
    sample_limit: Optional[int],
    require_all_scales: bool,
    expected_model_scales: Sequence[str],
    summary: dict,
) -> None:
    lines: List[str] = []
    lines.append("# Residue Importance Export Check")
    lines.append("")
    lines.append("## Run Scope")
    lines.append("")
    lines.append(f"- manifest: `{manifest_path}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- warnings_csv: `{warnings_csv_path}`")
    lines.append(f"- sample_limit: `{sample_limit if sample_limit is not None else 'all'}`")
    lines.append(f"- require_all_scales: `{require_all_scales}`")
    lines.append(f"- expected_model_scales: `{list(expected_model_scales)}`")
    lines.append("- residue_index_base: `0`")
    lines.append("- residue_importance definition: `softmax(residue_score_raw / temperature)` from current `ppi_softmax_sum` residue scores.")
    lines.append("")
    lines.append("## Canonical Order Guarantee")
    lines.append("")
    lines.append("- Export uses canonical fields from `canonical_manifest.csv`: `canonical_pair_id`, `proteinA_id`, `proteinB_id`, `proteinA_scores_path`, `proteinB_scores_path`.")
    lines.append("- `alignment_mode` is derived from `sample_to_canonical_swapped`.")
    lines.append("- For `aligned_via_swap` samples, exported `A/B` files still come from canonical `proteinA_scores_path` / `proteinB_scores_path`, never from raw sample protein order.")
    lines.append("- Therefore every exported `*_A.csv` corresponds to canonical `proteinA_id`, and every exported `*_B.csv` corresponds to canonical `proteinB_id`.")
    lines.append("")
    lines.append("## Export Summary")
    lines.append("")
    lines.append(f"- selected_sample_count: **{summary['selected_sample_count']}**")
    lines.append(f"- selected_sample_scale_rows: **{summary['selected_sample_scale_rows']}**")
    lines.append(f"- exported_unique_sample_count: **{summary['exported_unique_sample_count']}**")
    lines.append(f"- exported_sample_scale_cases: **{summary['exported_sample_scale_cases']}**")
    lines.append(f"- exported_csv_count: **{summary['exported_csv_count']}**")
    lines.append(f"- exported_by_scale: `{summary['exported_by_scale']}`")
    lines.append(f"- exported_by_alignment: `{summary['exported_by_alignment']}`")
    lines.append(f"- warning_count: **{summary['warning_count']}**")
    lines.append("")
    lines.append("## Skip Summary")
    lines.append("")
    lines.append(f"- filter_skip_counts: `{summary['filter_skip_counts']}`")
    lines.append(f"- selection_skip_counts: `{summary['selection_skip_counts']}`")
    lines.append(f"- runtime_skip_counts: `{summary['runtime_skip_counts']}`")
    lines.append("")
    lines.append("## Selected Sample IDs")
    lines.append("")
    for sample_id in summary["selected_sample_ids"]:
        lines.append(f"- `{sample_id}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This step exports residue-level importance only.")
    lines.append("- It does not merge ProteinShake proxy fields into the exported CSV schema.")
    lines.append("- It does not compute overlap.")
    lines.append("- It does not produce final figures.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_proteinshake_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        index_path,
        usecols=list(PROTEINSHAKE_INDEX_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    return df.drop_duplicates(subset=["pair_id"], keep="first").copy()


def proteinshake_failure_record(
    *,
    sample_id: str,
    canonical_pair_id: str,
    model_scale: str,
    protein_side: str,
    canonical_protein_id: str,
    alignment_mode: str,
    failure_reason: str,
    proteinshake_status: str,
    proteinshake_source_field: str,
    proteinshake_source_path: str,
    len_residue_table,
    len_proteinshake_score,
    notes: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "canonical_pair_id": canonical_pair_id,
        "model_scale": model_scale,
        "protein_side": protein_side,
        "canonical_protein_id": canonical_protein_id,
        "alignment_mode": alignment_mode,
        "failure_reason": failure_reason,
        "proteinshake_status": proteinshake_status,
        "proteinshake_source_field": proteinshake_source_field,
        "proteinshake_source_path": proteinshake_source_path,
        "len_residue_table": "" if len_residue_table is None else len_residue_table,
        "len_proteinshake_score": "" if len_proteinshake_score is None else len_proteinshake_score,
        "notes": notes,
    }


def discover_step3_cases(
    step3_root: Path,
    sample_ids: Optional[Sequence[str]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, Path]], Counter]:
    if not step3_root.exists():
        raise FileNotFoundError(f"step3 residue-importance root not found: {step3_root}")

    sample_filter = set(sample_ids) if sample_ids is not None else None
    cases: Dict[Tuple[str, str], Dict[str, Path]] = {}
    skipped = Counter()

    for sample_dir in sorted(step3_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        if sample_filter is not None and sample_id not in sample_filter:
            continue
        for csv_path in sorted(sample_dir.glob("*.csv")):
            stem = csv_path.stem
            if "_" not in stem:
                skipped["invalid_step3_filename"] += 1
                continue
            model_scale, protein_side = stem.rsplit("_", 1)
            if protein_side not in {"A", "B"}:
                skipped["invalid_step3_filename"] += 1
                continue
            cases.setdefault((sample_id, model_scale), {})[protein_side] = csv_path

    if sample_filter is not None:
        discovered_sample_ids = {sample_id for sample_id, _ in cases}
        for sample_id in sample_filter:
            if sample_id not in discovered_sample_ids:
                skipped["sample_id_not_found_in_step3"] += 1

    return cases, skipped


def load_step3_side_frame(path: Path) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    missing = [column for column in STEP3_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"step3 residue table missing required columns {missing}: {path}")
    if df.empty:
        raise ValueError(f"step3 residue table is empty: {path}")

    df = df.copy()
    df["residue_index"] = df["residue_index"].astype(np.int64)
    df = df.sort_values("residue_index").reset_index(drop=True)
    observed_index = df["residue_index"].to_numpy(dtype=np.int64)
    expected_index = np.arange(df.shape[0], dtype=np.int64)
    if not np.array_equal(observed_index, expected_index):
        raise ValueError(f"step3 residue_index must be contiguous and zero-based: {path}")

    metadata = {}
    for column in (
        "sample_id",
        "canonical_pair_id",
        "model_scale",
        "pooling_mode",
        "protein_side",
        "canonical_protein_id",
        "alignment_mode",
    ):
        unique_values = df[column].astype(str).unique()
        if len(unique_values) != 1:
            raise ValueError(f"step3 residue table must have exactly one {column}: {path}")
        metadata[column] = unique_values[0]

    if "pooling_temperature" in df.columns:
        unique_values = df["pooling_temperature"].astype(str).unique()
        if len(unique_values) != 1:
            raise ValueError(f"step3 residue table must have exactly one pooling_temperature: {path}")
        metadata["pooling_temperature"] = unique_values[0]
    else:
        metadata["pooling_temperature"] = ""

    metadata["length"] = int(df.shape[0])
    return df, metadata


def standardize_proteinshake_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("ProteinShake residue score must be 1D")
    if not np.isfinite(scores).all():
        raise ValueError("ProteinShake residue score contains non-finite values")
    standardized = np.clip(scores.astype(np.float32, copy=False), 0.0, 1.0)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized ProteinShake residue score contains non-finite values")
    return standardized


def build_ppi_mask_topk(scores: np.ndarray, topk_fraction: float) -> np.ndarray:
    if not 0.0 < float(topk_fraction) <= 1.0:
        raise ValueError(f"topk_fraction must be in (0, 1], got {topk_fraction}")
    if scores.ndim != 1:
        raise ValueError("topk mask expects a 1D score array")
    residue_count = int(scores.shape[0])
    if residue_count == 0:
        raise ValueError("topk mask cannot be built for an empty score array")
    topk_count = max(1, int(math.ceil(residue_count * float(topk_fraction))))
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(residue_count, dtype=np.int8)
    mask[order[:topk_count]] = 1
    return mask


def build_ppi_mask_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if scores.ndim != 1:
        raise ValueError("threshold mask expects a 1D score array")
    return (scores >= float(threshold)).astype(np.int8, copy=False)


def compute_global_threshold_fraction(index_df: pd.DataFrame, threshold: float, array_cache: ArrayCache) -> float:
    total_residues = 0
    total_above = 0
    for _, row in index_df[index_df["status"] == "cached"].iterrows():
        for column in ("residue_scores_A_path", "residue_scores_B_path"):
            scores = standardize_proteinshake_scores(array_cache.load_scores(row[column]))
            total_residues += int(scores.shape[0])
            total_above += int((scores >= float(threshold)).sum())
    if total_residues == 0:
        return 0.0
    return float(total_above) / float(total_residues)


def clean_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*"):
        if child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_dir():
                    nested.rmdir()
            child.rmdir()
        elif child.is_file():
            child.unlink()


def run_proteinshake_proxy_merge(
    *,
    step3_root: Path,
    proteinshake_index_path: Path,
    output_root: Path,
    failures_csv_path: Path,
    sample_ids: Optional[Sequence[str]],
    ppi_topk_fraction: float,
    ppi_threshold: float,
    clean_output: bool,
) -> dict:
    if clean_output:
        clean_directory_contents(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    failures_csv_path.parent.mkdir(parents=True, exist_ok=True)

    index_df = read_proteinshake_index(proteinshake_index_path)
    index_rows = {row["pair_id"]: row for row in index_df.to_dict(orient="records")}
    cases, discovery_skips = discover_step3_cases(step3_root, sample_ids=sample_ids)
    if not cases:
        raise ValueError("No step3 residue-importance cases found for ProteinShake proxy merge.")

    array_cache = ArrayCache()
    failures: List[dict] = []
    merged_csv_count = 0
    merged_sample_scale_cases = 0
    merged_sample_ids = set()
    merged_by_scale = Counter()
    merged_by_alignment = Counter()
    ppi_equals_residue_score_raw_csv_count = 0
    merge_skip_counts = Counter()
    used_pairs: Dict[str, dict] = {}

    ordered_cases = sorted(
        cases.items(),
        key=lambda item: (item[0][0], normalize_model_scale_key(item[0][1])),
    )

    for (sample_id, model_scale), side_paths in ordered_cases:
        missing_sides = sorted({"A", "B"} - set(side_paths))
        if missing_sides:
            merge_skip_counts["missing_step3_side"] += 1
            failures.append(
                proteinshake_failure_record(
                    sample_id=sample_id,
                    canonical_pair_id="",
                    model_scale=model_scale,
                    protein_side="both",
                    canonical_protein_id="",
                    alignment_mode="",
                    failure_reason="missing_step3_side",
                    proteinshake_status="not_attempted",
                    proteinshake_source_field="",
                    proteinshake_source_path="",
                    len_residue_table=None,
                    len_proteinshake_score=None,
                    notes=f"Missing step3 residue table(s): {','.join(missing_sides)}",
                )
            )
            continue

        try:
            df_a, meta_a = load_step3_side_frame(side_paths["A"])
            df_b, meta_b = load_step3_side_frame(side_paths["B"])
        except Exception as exc:
            merge_skip_counts["step3_load_error"] += 1
            failures.append(
                proteinshake_failure_record(
                    sample_id=sample_id,
                    canonical_pair_id="",
                    model_scale=model_scale,
                    protein_side="both",
                    canonical_protein_id="",
                    alignment_mode="",
                    failure_reason="step3_load_error",
                    proteinshake_status="not_attempted",
                    proteinshake_source_field="",
                    proteinshake_source_path="",
                    len_residue_table=None,
                    len_proteinshake_score=None,
                    notes=str(exc),
                )
            )
            continue

        metadata_mismatches = []
        for column in ("sample_id", "canonical_pair_id", "model_scale", "alignment_mode"):
            if meta_a[column] != meta_b[column]:
                metadata_mismatches.append(f"{column}: A={meta_a[column]!r}, B={meta_b[column]!r}")
        if meta_a["protein_side"] != "A" or meta_b["protein_side"] != "B":
            metadata_mismatches.append(
                f"protein_side mismatch: A_file={meta_a['protein_side']!r}, B_file={meta_b['protein_side']!r}"
            )
        if metadata_mismatches:
            merge_skip_counts["step3_metadata_mismatch"] += 1
            failures.append(
                proteinshake_failure_record(
                    sample_id=sample_id,
                    canonical_pair_id=meta_a["canonical_pair_id"],
                    model_scale=model_scale,
                    protein_side="both",
                    canonical_protein_id="",
                    alignment_mode=meta_a["alignment_mode"],
                    failure_reason="step3_metadata_mismatch",
                    proteinshake_status="not_attempted",
                    proteinshake_source_field="",
                    proteinshake_source_path="",
                    len_residue_table=None,
                    len_proteinshake_score=None,
                    notes=" | ".join(metadata_mismatches),
                )
            )
            continue

        canonical_pair_id = meta_a["canonical_pair_id"]
        alignment_mode = meta_a["alignment_mode"]
        pair_row = index_rows.get(canonical_pair_id)
        if pair_row is None:
            merge_skip_counts["proteinshake_pair_not_found"] += 1
            failures.append(
                proteinshake_failure_record(
                    sample_id=sample_id,
                    canonical_pair_id=canonical_pair_id,
                    model_scale=model_scale,
                    protein_side="both",
                    canonical_protein_id="",
                    alignment_mode=alignment_mode,
                    failure_reason="proteinshake_pair_not_found",
                    proteinshake_status="missing",
                    proteinshake_source_field="",
                    proteinshake_source_path="",
                    len_residue_table=None,
                    len_proteinshake_score=None,
                    notes="canonical_pair_id missing from multippimi_ppi_prior_index.csv",
                )
            )
            continue

        if pair_row["status"] != "cached":
            merge_skip_counts["proteinshake_status_not_cached"] += 1
            failures.append(
                proteinshake_failure_record(
                    sample_id=sample_id,
                    canonical_pair_id=canonical_pair_id,
                    model_scale=model_scale,
                    protein_side="both",
                    canonical_protein_id="",
                    alignment_mode=alignment_mode,
                    failure_reason="proteinshake_status_not_cached",
                    proteinshake_status=pair_row["status"],
                    proteinshake_source_field="",
                    proteinshake_source_path="",
                    len_residue_table=None,
                    len_proteinshake_score=None,
                    notes=pair_row["error"] or "ProteinShake prior is not cached for this canonical pair",
                )
            )
            continue

        pair_failed = False
        merged_frames: Dict[str, pd.DataFrame] = {}

        for protein_side, residue_df, metadata in (("A", df_a, meta_a), ("B", df_b, meta_b)):
            source_field = f"residue_scores_{protein_side}_path"
            source_path = pair_row[source_field]
            expected_protein_id = pair_row[f"protein{protein_side}_id"]
            if metadata["canonical_protein_id"] != expected_protein_id:
                merge_skip_counts["canonical_protein_id_mismatch"] += 1
                failures.append(
                    proteinshake_failure_record(
                        sample_id=sample_id,
                        canonical_pair_id=canonical_pair_id,
                        model_scale=model_scale,
                        protein_side=protein_side,
                        canonical_protein_id=metadata["canonical_protein_id"],
                        alignment_mode=alignment_mode,
                        failure_reason="canonical_protein_id_mismatch",
                        proteinshake_status=pair_row["status"],
                        proteinshake_source_field=source_field,
                        proteinshake_source_path=source_path,
                        len_residue_table=metadata["length"],
                        len_proteinshake_score=None,
                        notes=f"step3 canonical_protein_id does not match ProteinShake index protein{protein_side}_id",
                    )
                )
                pair_failed = True
                continue

            if not source_path or not Path(source_path).is_file():
                merge_skip_counts["proteinshake_source_missing"] += 1
                failures.append(
                    proteinshake_failure_record(
                        sample_id=sample_id,
                        canonical_pair_id=canonical_pair_id,
                        model_scale=model_scale,
                        protein_side=protein_side,
                        canonical_protein_id=metadata["canonical_protein_id"],
                        alignment_mode=alignment_mode,
                        failure_reason="proteinshake_source_missing",
                        proteinshake_status=pair_row["status"],
                        proteinshake_source_field=source_field,
                        proteinshake_source_path=source_path,
                        len_residue_table=metadata["length"],
                        len_proteinshake_score=None,
                        notes="ProteinShake residue-score source file is missing",
                    )
                )
                pair_failed = True
                continue

            try:
                ppi_scores = standardize_proteinshake_scores(array_cache.load_scores(source_path))
            except Exception as exc:
                merge_skip_counts["proteinshake_load_error"] += 1
                failures.append(
                    proteinshake_failure_record(
                        sample_id=sample_id,
                        canonical_pair_id=canonical_pair_id,
                        model_scale=model_scale,
                        protein_side=protein_side,
                        canonical_protein_id=metadata["canonical_protein_id"],
                        alignment_mode=alignment_mode,
                        failure_reason="proteinshake_load_error",
                        proteinshake_status=pair_row["status"],
                        proteinshake_source_field=source_field,
                        proteinshake_source_path=source_path,
                        len_residue_table=metadata["length"],
                        len_proteinshake_score=None,
                        notes=str(exc),
                    )
                )
                pair_failed = True
                continue

            if int(ppi_scores.shape[0]) != int(metadata["length"]):
                merge_skip_counts["proteinshake_length_mismatch"] += 1
                failures.append(
                    proteinshake_failure_record(
                        sample_id=sample_id,
                        canonical_pair_id=canonical_pair_id,
                        model_scale=model_scale,
                        protein_side=protein_side,
                        canonical_protein_id=metadata["canonical_protein_id"],
                        alignment_mode=alignment_mode,
                        failure_reason="proteinshake_length_mismatch",
                        proteinshake_status=pair_row["status"],
                        proteinshake_source_field=source_field,
                        proteinshake_source_path=source_path,
                        len_residue_table=metadata["length"],
                        len_proteinshake_score=int(ppi_scores.shape[0]),
                        notes="step3 residue count does not match ProteinShake residue-score length",
                    )
                )
                pair_failed = True
                continue

            merged_df = residue_df.copy()
            merged_df["ppi_score"] = ppi_scores
            merged_df["ppi_mask_topk"] = build_ppi_mask_topk(ppi_scores, ppi_topk_fraction)
            merged_df["ppi_mask_threshold"] = build_ppi_mask_threshold(ppi_scores, ppi_threshold)
            if np.array_equal(merged_df["residue_score_raw"].to_numpy(dtype=np.float32), ppi_scores):
                ppi_equals_residue_score_raw_csv_count += 1
            remaining_columns = [column for column in merged_df.columns if column not in STEP4_OUTPUT_COLUMNS]
            merged_frames[protein_side] = merged_df[list(STEP4_OUTPUT_COLUMNS) + remaining_columns]

        if pair_failed:
            continue

        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for protein_side in ("A", "B"):
            merged_frames[protein_side].to_csv(sample_dir / f"{model_scale}_{protein_side}.csv", index=False)
            merged_csv_count += 1

        merged_sample_scale_cases += 1
        merged_sample_ids.add(sample_id)
        merged_by_scale[model_scale] += 1
        merged_by_alignment[alignment_mode] += 1
        used_pairs[canonical_pair_id] = {
            "proteinA_id": pair_row["proteinA_id"],
            "proteinB_id": pair_row["proteinB_id"],
            "cache_dir": pair_row["cache_dir"],
            "residue_scores_A_path": pair_row["residue_scores_A_path"],
            "residue_scores_B_path": pair_row["residue_scores_B_path"],
            "metadata_path": pair_row["metadata_path"],
        }

    failure_columns = [
        "sample_id",
        "canonical_pair_id",
        "model_scale",
        "protein_side",
        "canonical_protein_id",
        "alignment_mode",
        "failure_reason",
        "proteinshake_status",
        "proteinshake_source_field",
        "proteinshake_source_path",
        "len_residue_table",
        "len_proteinshake_score",
        "notes",
    ]
    failures_df = pd.DataFrame(failures, columns=failure_columns)
    failures_df.to_csv(failures_csv_path, index=False)

    global_threshold_fraction = compute_global_threshold_fraction(index_df, ppi_threshold, array_cache)

    return {
        "selected_sample_ids": sorted({sample_id for sample_id, _ in cases}),
        "selected_sample_count": len({sample_id for sample_id, _ in cases}),
        "discovered_sample_scale_cases": len(cases),
        "merged_unique_sample_count": len(merged_sample_ids),
        "merged_sample_scale_cases": int(merged_sample_scale_cases),
        "merged_csv_count": int(merged_csv_count),
        "merged_by_scale": dict(sorted(merged_by_scale.items(), key=lambda item: normalize_model_scale_key(item[0]))),
        "merged_by_alignment": dict(sorted(merged_by_alignment.items())),
        "discovery_skip_counts": dict(sorted(discovery_skips.items())),
        "merge_skip_counts": dict(sorted(merge_skip_counts.items())),
        "failure_count": int(len(failures_df)),
        "failure_csv_path": str(failures_csv_path),
        "ppi_score_source_field": "residue_scores_A_path/residue_scores_B_path",
        "ppi_score_standardization": "direct ProteinShake 35M residue_scores clipped to [0,1]",
        "ppi_equals_residue_score_raw_csv_count": int(ppi_equals_residue_score_raw_csv_count),
        "ppi_topk_fraction": float(ppi_topk_fraction),
        "ppi_threshold": float(ppi_threshold),
        "global_fraction_ge_threshold": float(global_threshold_fraction),
        "used_pair_count": len(used_pairs),
        "used_pairs": dict(sorted(used_pairs.items())),
    }


def write_proteinshake_merge_log_md(
    *,
    path: Path,
    step3_root: Path,
    proteinshake_index_path: Path,
    output_root: Path,
    failures_csv_path: Path,
    summary: dict,
) -> None:
    lines: List[str] = []
    lines.append("# Step 4 ProteinShake Proxy Merge Log")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- step3_input_root: `{step3_root}`")
    lines.append(f"- proteinshake_index: `{proteinshake_index_path}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- failures_csv: `{failures_csv_path}`")
    lines.append("- This step merges predicted PPI region from ProteinShake 35M into the residue-level analysis tables.")
    lines.append("- ProteinShake fields are treated as an external predicted interface proxy, not as ground truth.")
    lines.append("- This step does not compute overlap and does not produce final figures.")
    lines.append("")
    lines.append("## PPI Proxy Definition")
    lines.append("")
    lines.append(f"- `ppi_score` source field: `{summary['ppi_score_source_field']}` from `multippimi_ppi_prior_index.csv`.")
    lines.append(f"- `ppi_score` standardization: `{summary['ppi_score_standardization']}`.")
    lines.append("- Canonical ProteinShake source mapping: `residue_scores_A_path -> canonical proteinA_id`, `residue_scores_B_path -> canonical proteinB_id`.")
    topk_percent = 100.0 * float(summary["ppi_topk_fraction"])
    lines.append(
        f"- `ppi_mask_topk`: exact top `{topk_percent:.1f}%` residues by `ppi_score` "
        f"with `k = ceil(L * {summary['ppi_topk_fraction']:.2f})`, ties resolved by canonical residue index order."
    )
    lines.append(f"- `ppi_mask_threshold`: `ppi_score >= {summary['ppi_threshold']:.2f}`.")
    lines.append(f"- Global cached ProteinShake residue fraction above threshold `{summary['ppi_threshold']:.2f}`: `{summary['global_fraction_ge_threshold']:.6f}`.")
    lines.append(
        f"- Processed step4 CSVs where `ppi_score` is numerically identical to step3 `residue_score_raw`: "
        f"`{summary['ppi_equals_residue_score_raw_csv_count']}/{summary['merged_csv_count']}`."
    )
    lines.append("")
    lines.append("## Canonical Alignment Guarantee")
    lines.append("")
    lines.append("- Step4 merge reads step3 residue tables that are already frozen to canonical A/B order.")
    lines.append("- Join key is `canonical_pair_id`, never raw sample protein order.")
    lines.append("- For `aligned_via_swap` samples, exported `*_A.csv` still uses canonical `proteinA_id`, and exported `*_B.csv` still uses canonical `proteinB_id`.")
    lines.append("- ProteinShake A/B source files are resolved only through canonical `residue_scores_A_path` / `residue_scores_B_path` from the pair index.")
    lines.append("")
    lines.append("## Merge Summary")
    lines.append("")
    lines.append(f"- selected_sample_count: **{summary['selected_sample_count']}**")
    lines.append(f"- discovered_sample_scale_cases: **{summary['discovered_sample_scale_cases']}**")
    lines.append(f"- merged_unique_sample_count: **{summary['merged_unique_sample_count']}**")
    lines.append(f"- merged_sample_scale_cases: **{summary['merged_sample_scale_cases']}**")
    lines.append(f"- merged_csv_count: **{summary['merged_csv_count']}**")
    lines.append(f"- merged_by_scale: `{summary['merged_by_scale']}`")
    lines.append(f"- merged_by_alignment: `{summary['merged_by_alignment']}`")
    lines.append(f"- discovery_skip_counts: `{summary['discovery_skip_counts']}`")
    lines.append(f"- merge_skip_counts: `{summary['merge_skip_counts']}`")
    lines.append(f"- failure_count: **{summary['failure_count']}**")
    lines.append("")
    lines.append("## ProteinShake Pairs Used")
    lines.append("")
    if summary["used_pairs"]:
        for pair_id, payload in summary["used_pairs"].items():
            lines.append(
                f"- `{pair_id}`: A=`{payload['proteinA_id']}` from `{payload['residue_scores_A_path']}`, "
                f"B=`{payload['proteinB_id']}` from `{payload['residue_scores_B_path']}`."
            )
    else:
        lines.append("- No ProteinShake pairs were merged successfully.")
    lines.append("")
    lines.append("## Failure Notes")
    lines.append("")
    if summary["failure_count"] == 0:
        lines.append("- No ProteinShake merge failures were recorded for the processed step3 batch.")
    else:
        lines.append(f"- Detailed failures are recorded in `{failures_csv_path}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
