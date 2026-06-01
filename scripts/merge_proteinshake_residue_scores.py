#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rdppimi.proteinshake.attention_overlap import (  # noqa: E402
    parse_sample_ids,
    run_proteinshake_proxy_merge,
    write_proteinshake_merge_log_md,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge predicted PPI region from ProteinShake 35M into canonical residue-level attention tables."
    )
    parser.add_argument(
        "--step3_root",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/attention_overlap/step3_residue_importance",
    )
    parser.add_argument(
        "--proteinshake_index_path",
        type=Path,
        default=REPO_ROOT / "multippimi_ppi_prior_index.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/attention_overlap/step4_residue_tables",
    )
    parser.add_argument(
        "--merge_log_path",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/attention_overlap/step4_residue_tables/merge_log.md",
    )
    parser.add_argument(
        "--failures_csv_path",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/attention_overlap/logs/proteinshake_merge_failures.csv",
    )
    parser.add_argument("--sample_ids", type=str, default="", help="Comma-separated sample_ids or a text/CSV file path.")
    parser.add_argument("--ppi_topk_fraction", type=float, default=0.20)
    parser.add_argument("--ppi_threshold", type=float, default=0.88)
    parser.add_argument("--clean_output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = parse_sample_ids(args.sample_ids)
    summary = run_proteinshake_proxy_merge(
        step3_root=args.step3_root.resolve(),
        proteinshake_index_path=args.proteinshake_index_path.resolve(),
        output_root=args.output_root.resolve(),
        failures_csv_path=args.failures_csv_path.resolve(),
        sample_ids=sample_ids,
        ppi_topk_fraction=float(args.ppi_topk_fraction),
        ppi_threshold=float(args.ppi_threshold),
        clean_output=bool(args.clean_output),
    )
    write_proteinshake_merge_log_md(
        path=args.merge_log_path.resolve(),
        step3_root=args.step3_root.resolve(),
        proteinshake_index_path=args.proteinshake_index_path.resolve(),
        output_root=args.output_root.resolve(),
        failures_csv_path=args.failures_csv_path.resolve(),
        summary=summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
