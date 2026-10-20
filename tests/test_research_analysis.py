"""Check scoring, multiplicity correction, and the published aggregate report."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from research_analysis import TRAITS, adjust_bh, analyze, composite_scores, render_page


class ResearchAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(61)
        self.frame = pd.DataFrame({
            f"{code}{part}": rng.uniform(0.8, 4.8, 24)
            for code, _, count in TRAITS
            for part in range(1, count + 1)
        })
        self.frame.insert(0, "Identifier", [f"record-{n:03d}" for n in range(len(self.frame))])

    def test_composite_totals(self) -> None:
        scores = composite_scores(self.frame)
        self.assertEqual(scores.shape, (24, 7))
        self.assertAlmostEqual(
            scores.loc[0, "NS"],
            sum(self.frame.loc[0, f"NS{part}"] for part in range(1, 5)),
        )

    def test_fdr_adjustment(self) -> None:
        result = adjust_bh([0.01, 0.02, 0.06])
        self.assertTrue(np.allclose(result, [0.03, 0.03, 0.06]))

    def test_rejects_incomplete_scores(self) -> None:
        bad = self.frame.drop(columns="NS2")
        with self.assertRaisesRegex(ValueError, "NS2"):
            composite_scores(bad)
        bad = self.frame.copy()
        bad.loc[0, "HA1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            composite_scores(bad)

    def test_summary_does_not_export_profiles(self) -> None:
        summary = analyze(composite_scores(self.frame))
        self.assertEqual(summary["record_count"], 24)
        self.assertEqual(summary["profile_pairs"], 276)
        self.assertEqual(summary["subscale_count"], 29)
        self.assertEqual(summary["correlation_tests"], 21)
        self.assertEqual(len(summary["strongest_pairs"]), 5)
        self.assertEqual(len(summary["projection"]["distance_bins"]), 13)
        self.assertNotIn("record-000", str(summary))
        self.assertIn("24", render_page(summary))


if __name__ == "__main__":
    unittest.main()
