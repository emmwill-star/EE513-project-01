import importlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper"))

import utils


class MtfBatchTests(unittest.TestCase):
    def test_measures_all_five_images_for_each_corner(self):
        greenPlanes = {
            f"edge{index}.DNG": np.full((16, 16), index, dtype=float)
            for index in range(1, 6)
        }
        selection = {"roi": (0, 0, 16, 16)}
        for target in ("top-left", "top-right", "bottom-left", "bottom-right"):
            with self.subTest(target=target), patch.object(
                utils, "select_square_edge", return_value=selection
            ) as selectSquare, patch.object(
                utils,
                "measure_slanted_edge_mtf",
                side_effect=[
                    {"mtf50": index / 100, "frequency": [0, 0.125], "mtf": [1, 0.2]}
                    for index in range(1, 6)
                ],
            ) as measureMtf:
                selections, mtf50, nyquistMtf = utils.calculate_mtf50_from_mtf(
                    greenPlanes, 2, 0.125, target=target
                )
                self.assertEqual(set(selections), set(greenPlanes))
                self.assertEqual(
                    mtf50, {f"edge{index}.DNG": index / 100 for index in range(1, 6)}
                )
                self.assertEqual(nyquistMtf, dict.fromkeys(greenPlanes, 0.2))
                self.assertEqual(selectSquare.call_count, 5)
                for call in selectSquare.call_args_list:
                    self.assertEqual(call.kwargs, {"target": target})
                self.assertEqual(measureMtf.call_count, 5)
                for call in measureMtf.call_args_list:
                    self.assertEqual(
                        call.kwargs,
                        {"roi": selection["roi"], "sample_spacing": 4, "oversampling": 4},
                    )

    def test_default_target_remains_largest(self):
        with patch.object(
            utils, "select_square_edge", side_effect=ValueError("No square")
        ) as selectSquare:
            with self.assertRaisesRegex(ValueError, r"edge1.DNG \(largest\): No square"):
                utils.calculate_mtf50_from_mtf({"edge1.DNG": np.ones((16, 16))}, 2, 0.125)
            self.assertEqual(selectSquare.call_args.kwargs, {"target": "largest"})

    def test_measurement_failure_identifies_image_and_corner(self):
        with patch.object(
            utils, "select_square_edge", return_value={"roi": (0, 0, 16, 16)}
        ), patch.object(
            utils, "measure_slanted_edge_mtf", side_effect=ValueError("Invalid edge")
        ):
            with self.assertRaisesRegex(
                ValueError, r"edge2.DNG \(bottom-left\): Invalid edge"
            ):
                utils.calculate_mtf50_from_mtf(
                    {"edge2.DNG": np.ones((16, 16))}, 2, 0.125, target="bottom-left"
                )

    def run_corner_report(self, values, missingImage=None):
        projectFolder = Path(__file__).resolve().parents[1]
        notebook = json.loads((projectFolder / "project_main.ipynb").read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "81c01e99")
        greenPlanes = dict.fromkeys(f"edge{index}.DNG" for index in range(1, 6))
        if missingImage is not None:
            del greenPlanes[missingImage]
        context = {
            "np": np, "importlib": importlib, "utils": utils,
            "greenPlanes": greenPlanes, "p": 2, "nF": 0.125,
        }

        def measure_batch(planes, p, nF, *, target):
            return {}, dict(zip(planes, values[target], strict=True)), {}

        output = io.StringIO()
        with patch.object(importlib, "reload"), patch.object(
            utils, "calculate_mtf50_from_mtf", side_effect=measure_batch
        ), redirect_stdout(output):
            exec(compile("".join(cell["source"]), "<corner report>", "exec"), context)
        return context, output.getvalue()

    def test_corner_report_means_and_worst_mean_not_mean_of_minima(self):
        values = {
            "top-left": [0.01, 0.09, 0.09, 0.09, 0.09],
            "top-right": [0.08, 0.02, 0.08, 0.08, 0.08],
            "bottom-left": [0.08] * 5,
            "bottom-right": [0.09] * 5,
        }
        context, output = self.run_corner_report(values)
        for target in values:
            self.assertAlmostEqual(
                context["edgeSquareAverageMTF50"][target], np.mean(values[target])
            )
            self.assertAlmostEqual(
                context["edgeSquareAverageCyclesPerPixel"][target], 2 * np.mean(values[target])
            )
            for index, value in enumerate(values[target], start=1):
                self.assertAlmostEqual(
                    context["edgeSquareMTF50CyclesPerPixel"][target][f"edge{index}.DNG"],
                    2 * value,
                )
        self.assertAlmostEqual(context["worstCornerAverageCyclesPerPixel"], 0.136)
        self.assertIn("cycles/original sensor pixel", output)
        self.assertEqual(context["worstCorner"], "top-right")
        self.assertAlmostEqual(context["worstCornerAverageMTF50"], 0.068)
        self.assertNotAlmostEqual(
            context["worstCornerAverageMTF50"],
            np.mean(np.min(list(values.values()), axis=0)),
        )
        for index in range(1, 6):
            self.assertIn(f"edge{index}.DNG", output)
        self.assertIn("Average", output)
        self.assertIn("Worst corner average: top-right = 0.068000", output)

    def test_corner_report_rejects_missing_image(self):
        with self.assertRaisesRegex(KeyError, "edge5.DNG"):
            self.run_corner_report({}, missingImage="edge5.DNG")

    def test_saved_mtf_fields_use_sensor_pixel_units(self):
        values = {
            "top-left": [0.07, 0.08, 0.09, 0.10, 0.11],
            "top-right": [0.04, 0.05, 0.06, 0.07, 0.08],
            "bottom-left": [0.09] * 5,
            "bottom-right": [0.10] * 5,
        }
        context, _ = self.run_corner_report(values)
        context.update(Path=Path, centerMTF50={"edge1.DNG": 0.08, "edge2.DNG": 0.10})
        notebookPath = Path(__file__).resolve().parents[1] / "project_main.ipynb"
        notebook = json.loads(notebookPath.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "b8c875ce")
        with patch.object(importlib, "reload"), patch.object(utils, "update_json_values") as save:
            exec(compile("".join(cell["source"]), "<save MTF>", "exec"), context)
        saved = save.call_args.args[1]
        np.testing.assert_allclose(saved["mtf50_center_cy_per_px"], [0.18, 0.02])
        allCorners = np.array(list(values.values())) * 2
        np.testing.assert_allclose(
            saved["mtf50_corner_mean_cy_per_px"], [allCorners.mean(), allCorners.std()]
        )
        np.testing.assert_allclose(
            saved["mtf50_corner_worst_cy_per_px"],
            [0.12, np.std(np.array(values["top-right"]) * 2)],
        )

    def test_corner_report_rejects_unresolved_crossing(self):
        with self.assertRaisesRegex(ValueError, "top-left: unresolved MTF50.*edge3.DNG"):
            self.run_corner_report({"top-left": [0.08, 0.08, np.nan, 0.08, 0.08]})


if __name__ == "__main__":
    unittest.main()
