from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_rd_inputs import endpoint_protein_sets, induce_sidewise_sampled_train_pairs
import train_rd_scalers


class SidewiseCdhitPairSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = pd.DataFrame(
            [
                {'pair_id': 'p1', 'proteinA_id': 'A1', 'proteinB_id': 'B1'},
                {'pair_id': 'p2', 'proteinA_id': 'A2', 'proteinB_id': 'B2'},
                {'pair_id': 'p3', 'proteinA_id': 'A3', 'proteinB_id': 'B3'},
                {'pair_id': 'p4', 'proteinA_id': 'A4', 'proteinB_id': 'B4'},
            ]
        )

    def test_endpoint_sets_are_built_independently(self) -> None:
        proteins_a, proteins_b = endpoint_protein_sets(self.pairs)
        self.assertEqual(proteins_a, ['A1', 'A2', 'A3', 'A4'])
        self.assertEqual(proteins_b, ['B1', 'B2', 'B3', 'B4'])

    def test_pair_intersection_requires_both_endpoint_representatives(self) -> None:
        pairs_a, pairs_b, intersection = induce_sidewise_sampled_train_pairs(
            self.pairs,
            retained_protein_a_set={'A1', 'A2'},
            retained_protein_b_set={'B1', 'B3'},
        )
        self.assertEqual(pairs_a, ['p1', 'p2'])
        self.assertEqual(pairs_b, ['p1', 'p3'])
        self.assertEqual(intersection, ['p1'])

    def test_fold_plan_runs_cdhit_once_per_endpoint_and_intersects_pairs(self) -> None:
        pair_df = self.pairs.assign(
            seqA=['AAAA', 'CCCC', 'DDDD', 'EEEE'],
            seqB=['FFFF', 'GGGG', 'HHHH', 'IIII'],
            length_A=4,
            length_B=4,
            residue_rows=8,
        )
        sequence_map = {
            **dict(zip(pair_df['proteinA_id'], pair_df['seqA'])),
            **dict(zip(pair_df['proteinB_id'], pair_df['seqB'])),
        }
        split = SimpleNamespace(
            mapped_train_pairs=['p1', 'p2', 'p3', 'p4'],
            mapped_valid_pairs=[],
            mapped_test_pairs=[],
        )
        cluster_df = pd.DataFrame(
            [{'cluster_id': 0, 'representative_protein': 'x', 'member_count': 1, 'members': 'x'}]
        )

        def fake_run_cdhit(fasta_path: Path, output_prefix: Path, threshold: float):
            if '_A.' in fasta_path.name:
                return ['A1', 'A2'], cluster_df
            return ['B1', 'B3'], cluster_df

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                output_root=Path(tmpdir),
                eval_setting='S4',
                cdhit_threshold=0.5,
                cdhit_pair_mode='sidewise_intersection',
            )
            with (
                patch.object(train_rd_scalers, 'build_official_split', return_value=split),
                patch.object(train_rd_scalers, 'run_cdhit', side_effect=fake_run_cdhit) as run_mock,
            ):
                plan = train_rd_scalers.build_fold_plan(args, 1, pair_df, sequence_map)

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(plan['sampled_pair_ids'], ['p1'])
        self.assertEqual(plan['rd_fit_pair_count_A'], 2)
        self.assertEqual(plan['rd_fit_pair_count_B'], 2)
        self.assertEqual(plan['cdhit_pair_mode'], 'sidewise_intersection')


if __name__ == '__main__':
    unittest.main()
