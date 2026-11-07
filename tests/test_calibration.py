import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper"))

import utils


class CalibrationTests(unittest.TestCase):
    def test_straight_grid_has_zero_line_residual(self):
        points = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(float)
        metrics = utils.measure_checkerboard_straightness(points, (9, 6))
        self.assertAlmostEqual(metrics["rmsPx"], 0.)
        self.assertAlmostEqual(metrics["maxPx"], 0.)
        self.assertEqual(len(metrics["lines"]), 15)

    def test_undistortion_reduces_synthetic_grid_curvature(self):
        grid = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(np.float32)
        objectPoints = np.column_stack(((grid - [4, 2.5]) * .15, np.ones(54)))
        cameraMatrix = np.array([[800., 0., 500.], [0., 800., 400.], [0., 0., 1.]])
        distortion = np.array([.15, -.02, .001, -.001, 0.])
        observed, _ = cv2.projectPoints(
            objectPoints, np.zeros(3), np.zeros(3), cameraMatrix, distortion
        )
        corrected = cv2.undistortPoints(
            observed, cameraMatrix, distortion, P=cameraMatrix
        )
        before = utils.measure_checkerboard_straightness(observed, (9, 6))
        after = utils.measure_checkerboard_straightness(corrected, (9, 6))
        self.assertGreater(before["rmsPx"], 1.)
        self.assertLess(after["rmsPx"], .01)

    def test_straightness_rejects_invalid_points(self):
        for points in (np.zeros((4, 2)), np.full((54, 2), np.nan), np.zeros((54, 2))):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                utils.measure_checkerboard_straightness(points, (9, 6))

    def test_calibration_preserves_width_height_order(self):
        imagePoints = [np.zeros((4, 1, 2), dtype=np.float32)]
        objectPoints = [np.zeros((4, 3), dtype=np.float32)]
        result = (0.5, np.eye(3), np.zeros(5), [], [], [], [], [])
        with patch.object(utils.cv2, "calibrateCameraExtended", return_value=result) as calibrate:
            actual = utils.calibrate_camera(imagePoints, objectPoints, (3024, 4032))
        self.assertEqual(calibrate.call_args.args[2], (3024, 4032))
        self.assertIs(actual[0], result[1])

    def test_refinement_uses_unannotated_pixels(self):
        rgb = np.full((40, 60, 3), 80, dtype=np.uint8)
        corners = np.array([[[20.0, 20.0]]], dtype=np.float32)
        refined = corners + 0.25
        with (
            patch.object(utils.rawpy, "imread") as readRaw,
            patch.object(utils.cv2, "findChessboardCorners", return_value=(True, corners)),
            patch.object(utils.cv2, "cornerSubPix", return_value=refined) as refine,
            patch.object(utils.cv2, "drawChessboardCorners") as draw,
        ):
            readRaw.return_value.__enter__.return_value.postprocess.return_value = rgb
            draw.side_effect = lambda image, *args: image.fill(255)
            imagePoints, objectPoints, annotated = utils.checkerboard_found(
                Path("checkerboard.dng"), (1, 1)
            )
            readRaw.return_value.__enter__.return_value.postprocess.assert_called_once_with(
                user_flip=0
            )
        np.testing.assert_array_equal(refine.call_args.args[0], np.full((40, 60), 80))
        self.assertIs(imagePoints[0], refined)
        self.assertEqual(objectPoints[0].shape, (1, 3))
        np.testing.assert_array_equal(annotated, np.full_like(rgb, 255))

    def test_reprojection_error_is_euclidean_rms_per_view(self):
        objectPoints = [np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)] * 2
        cameraMatrix = np.eye(3)
        distCoeffs = np.zeros(5)
        poses = [np.zeros(3)] * 2
        projected, _ = cv2.projectPoints(
            objectPoints[0], poses[0], poses[0], cameraMatrix, distCoeffs
        )
        imagePoints = [
            projected + np.array([3, 4], dtype=np.float32),
            projected + np.array([0, 2], dtype=np.float32),
        ]
        meanError, errors = utils.calculate_mean_reprojection_error(
            imagePoints, objectPoints, cameraMatrix, distCoeffs, poses, poses
        )
        np.testing.assert_allclose(errors, [5, 2])
        self.assertAlmostEqual(meanError, 3.5)

    def test_perfect_projection_has_zero_error(self):
        objectPoints = [np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)]
        poses = [np.zeros(3)]
        projected, _ = cv2.projectPoints(
            objectPoints[0], poses[0], poses[0], np.eye(3), np.zeros(5)
        )
        meanError, errors = utils.calculate_mean_reprojection_error(
            [projected], objectPoints, np.eye(3), np.zeros(5), poses, poses
        )
        self.assertEqual(meanError, 0)
        np.testing.assert_array_equal(errors, [0])


if __name__ == "__main__":
    unittest.main()
