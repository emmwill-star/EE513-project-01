import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper"))

import utils


class ColorSummaryTests(unittest.TestCase):
    def test_population_std_uses_only_selected_patches(self):
        summary = utils.summarize_delta_e(
            np.array([1., 3., 100.]), ["a", "b", "clipped"], [True, True, False]
        )
        self.assertEqual(summary["meanDeltaE"], 2.)
        self.assertEqual(summary["stdDeltaE"], 1.)
        self.assertEqual(summary["maxDeltaE"], 3.)
        self.assertEqual(summary["worstPatches"][0], (2, "b", 3.))

    def test_single_patch_has_zero_population_std(self):
        summary = utils.summarize_delta_e(np.array([2.]), ["a"])
        self.assertEqual(summary["stdDeltaE"], 0.)

    def test_plot_reports_before_and_after_std(self):
        ax, summary = utils.plot_delta_e_before_after(
            np.array([1., 3., 100.]),
            np.array([0., 1., 100.]),
            ["a", "b", "clipped"],
            statsMask=[True, True, False],
        )
        try:
            self.assertEqual(summary["before"]["stdDeltaE"], 1.)
            self.assertEqual(summary["after"]["stdDeltaE"], .5)
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            self.assertIn("std 1.00", labels[0])
            self.assertIn("std 0.50", labels[1])
        finally:
            utils.plt.close(ax.figure)
