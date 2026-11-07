import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper"))

from utils import select_square_edge


class SquareSelectionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((800, 900), 1000, dtype=np.float64)
        self.centers = {
            "largest": (450, 400),
            "top-left": (150, 150),
            "top-right": (750, 150),
            "bottom-left": (150, 650),
            "bottom-right": (750, 650),
        }
        for index, (target, center) in enumerate(self.centers.items()):
            side = 200 if target == "largest" else 85 + index * 5
            corners = cv2.boxPoints((center, (side, side), 5)).astype(np.int32)
            cv2.fillConvexPoly(self.image, corners, 100)

    def test_selects_all_five_targets_and_preserves_input(self):
        original = self.image.copy()
        for target, center in self.centers.items():
            with self.subTest(target=target):
                result = select_square_edge(self.image, target=target)
                self.assertEqual(
                    set(result), {"squareCorners", "edgeEndpoints", "roi", "edgeAngleDeg"}
                )
                np.testing.assert_allclose(result["squareCorners"].mean(axis=0), center, atol=2)
                self.assertGreater(result["edgeEndpoints"][:, 0].mean(), center[0])
                x, y, width, height = result["roi"]
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + width, self.image.shape[1])
                self.assertLessEqual(y + height, self.image.shape[0])
                roi = self.image[y:y + height, x:x + width]
                self.assertLess(np.median(roi[:, :5]), np.median(roi[:, -5:]))
        np.testing.assert_array_equal(self.image, original)

    def test_default_is_largest(self):
        default = select_square_edge(self.image)
        explicit = select_square_edge(self.image, target="largest")
        for key in default:
            np.testing.assert_equal(default[key], explicit[key])

    def test_missing_corner_raises_instead_of_selecting_another_square(self):
        self.image[50:250, 50:250] = 1000
        with self.assertRaisesRegex(ValueError, "No top-left square"):
            select_square_edge(self.image, target="top-left")

    def test_invalid_target_raises(self):
        with self.assertRaisesRegex(ValueError, "target must be"):
            select_square_edge(self.image, target="center")

    def test_area_filter_remains_configurable_for_small_corners(self):
        with self.assertRaisesRegex(ValueError, "No top-left square"):
            select_square_edge(self.image, target="top-left", minAreaFraction=0.02)
        result = select_square_edge(
            self.image, target="top-left", minAreaFraction=0.005
        )
        np.testing.assert_allclose(
            result["squareCorners"].mean(axis=0), self.centers["top-left"], atol=2
        )

    def test_selects_corners_under_uneven_illumination(self):
        shadedImage = self.image * np.linspace(0.3, 1, self.image.shape[1])
        for target, center in self.centers.items():
            if target == "largest":
                continue
            with self.subTest(target=target):
                result = select_square_edge(shadedImage, target=target)
                np.testing.assert_allclose(
                    result["squareCorners"].mean(axis=0), center, atol=2
                )


if __name__ == "__main__":
    unittest.main()
