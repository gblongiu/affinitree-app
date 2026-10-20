"""Check the published atlas, its interactive controls, and data boundaries."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
sys.path.insert(0, str(ROOT / "scripts"))

from research_analysis import render_page


class AtlasPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = (SITE / "template.html").read_text(encoding="utf-8")
        cls.page = (SITE / "index.html").read_text(encoding="utf-8")
        cls.script = (SITE / "app.js").read_text(encoding="utf-8")
        cls.results = json.loads((SITE / "results.json").read_text(encoding="utf-8"))

    def test_published_html_matches_data_and_template(self) -> None:
        self.assertEqual(self.page, render_page(self.results))
        self.assertEqual(self.template.count('href="atlas.css"'), 1)
        self.assertEqual(self.page.count('id="shape"'), 1)

    def test_shape_lab_has_seven_controls_and_two_valid_endpoints(self) -> None:
        for control in ('shape-sliders', 'shape-svg', 'shape-distance-value',
                        'shape-match', 'shape-reset', 'shape-contributions'):
            self.assertIn(f'id="{control}"', self.template)
        for trait in ('NS', 'HA', 'RD', 'P', 'SD', 'CO', 'ST'):
            self.assertRegex(self.script, rf'\b{trait}:\s*\{{ color:')
        self.assertIn('const fixed = [', self.script)
        self.assertIn('const initial = [', self.script)
        self.assertIn("sum === 0 ? 0", self.script)

    def test_palette_preserves_original_seven_colors(self) -> None:
        for color in ('#FFA500', '#FF0000', '#FF69B4', '#8B4513',
                      '#0000FF', '#4CAF50', '#7E2F94'):
            self.assertIn(color, self.script)

    def test_report_remains_aggregate_and_exploratory(self) -> None:
        self.assertEqual(self.results['record_count'], 50)
        self.assertEqual(self.results['profile_pairs'], 1225)
        self.assertEqual(self.results['correlation_tests'], 21)
        self.assertEqual(self.results['fdr_significant'], 9)
        self.assertIn('ILLUSTRATIVE', self.template)
        self.assertIn('no individual', self.template.lower())
        self.assertNotIn('record-000', self.page)

    def test_site_has_no_external_runtime_dependencies(self) -> None:
        self.assertNotIn('<script src="http', self.page)
        self.assertNotIn('fonts.googleapis', self.page)
        self.assertNotIn('google-analytics', self.page.lower())
        for name in ('app.js', 'styles.css', 'atlas.css'):
            self.assertTrue((SITE / name).is_file())


if __name__ == '__main__':
    unittest.main()
