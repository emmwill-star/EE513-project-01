import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper"))

from utils import add_to_results


class ResultsTests(unittest.TestCase):
    def test_creates_missing_and_empty_files(self):
        with tempfile.TemporaryDirectory() as tempDir:
            for isEmpty in (False, True):
                with self.subTest(isEmpty=isEmpty):
                    yamlPath = Path(tempDir) / f"results_{isEmpty}.yml"
                    if isEmpty:
                        yamlPath.touch()
                    results = {"mean": 12.5, "samples": [1, 2], "metadata": {"unit": "DN"}}
                    add_to_results(results, str(yamlPath), "noise")
                    self.assertEqual(yaml.safe_load(yamlPath.read_text()), {"noise": results})

    def test_merges_section_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tempDir:
            yamlPath = Path(tempDir) / "results.yml"
            yamlPath.write_text(
                "noise:\n  mean: 1\n  std: 2\n  metadata:\n    old: true\n"
                "camera:\n  model: iPhone\n",
                encoding="utf-8",
            )
            results = {"mean": 3, "metadata": {"new": True}}
            add_to_results(results, yamlPath, "noise")
            add_to_results({"gain": 4}, yamlPath, "calibration")
            self.assertEqual(
                yaml.safe_load(yamlPath.read_text()),
                {
                    "noise": {"mean": 3, "std": 2, "metadata": {"new": True}},
                    "camera": {"model": "iPhone"},
                    "calibration": {"gain": 4},
                },
            )
            self.assertEqual(results, {"mean": 3, "metadata": {"new": True}})

    def test_invalid_yaml_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tempDir:
            yamlPath = Path(tempDir) / "results.yml"
            for original, errorType in (
                ("- not a mapping\n", ValueError),
                ("noise: 42\n", ValueError),
                ("noise: null\n", ValueError),
                ("noise: [\n", yaml.YAMLError),
            ):
                with self.subTest(original=original):
                    yamlPath.write_text(original, encoding="utf-8")
                    with self.assertRaises(errorType):
                        add_to_results({"mean": 3}, yamlPath, "noise")
                    self.assertEqual(yamlPath.read_text(), original)

    def test_serialization_failure_does_not_truncate_file(self):
        with tempfile.TemporaryDirectory() as tempDir:
            yamlPath = Path(tempDir) / "results.yml"
            original = "noise:\n  mean: 1\n"
            yamlPath.write_text(original, encoding="utf-8")
            with self.assertRaises(yaml.representer.RepresenterError):
                add_to_results({"value": object()}, yamlPath, "noise")
            self.assertEqual(yamlPath.read_text(), original)

    def test_invalid_arguments_do_not_create_file(self):
        with tempfile.TemporaryDirectory() as tempDir:
            yamlPath = Path(tempDir) / "results.yml"
            for results, measurement in (([], "noise"), ({}, ""), ({}, "  "), ({}, None)):
                with self.subTest(results=results, measurement=measurement):
                    with self.assertRaises(TypeError):
                        add_to_results(results, yamlPath, measurement)
                    self.assertFalse(yamlPath.exists())


if __name__ == "__main__":
    unittest.main()
