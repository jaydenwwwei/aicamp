from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


DAY3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY3))

from d3 import annotate_and_save, readable_image  # noqa: E402
from label_dataset import grouped_splits  # noqa: E402
from lego_model import LegoMultiTaskModel  # noqa: E402
from train_model import build_parser as build_training_parser, metrics_from_predictions  # noqa: E402


class ProgramTests(unittest.TestCase):
    def test_invalid_path_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            readable_image("definitely-missing-image.jpg")

    def test_non_image_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-image.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a readable image"):
                readable_image(str(path))

    def test_model_has_expected_output_shapes(self):
        model = LegoMultiTaskModel(num_directions=8, num_colors=10, pretrained=False).eval()
        with torch.inference_mode():
            direction, color = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(direction.shape), (2, 8))
        self.assertEqual(tuple(color.shape), (2, 10))

    def test_group_split_does_not_leak(self):
        classes = [str(value) for value in range(1, 51)]
        split = grouped_splits(classes)
        self.assertEqual(set(split), set(classes))
        self.assertEqual(set(split.values()), {"train", "val", "test"})

    def test_metrics_include_confusion_matrix(self):
        metrics = metrics_from_predictions([0, 1, 1], [0, 0, 1], 2)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 1]])

    def test_training_parser_supports_resume(self):
        args = build_training_parser().parse_args(["--resume", "--epochs", "3"])
        self.assertTrue(args.resume)
        self.assertEqual(args.epochs, 3)

    def test_annotation_creates_image(self):
        with tempfile.TemporaryDirectory() as directory:
            result = annotate_and_save(
                Image.new("RGB", (300, 300), "red"),
                {
                    "direction": "front",
                    "direction_confidence": 0.9,
                    "direction_uncertain": False,
                    "head_color": "red",
                    "color_confidence": 0.5,
                    "color_uncertain": True,
                },
                Path("sample.jpg"),
                Path(directory),
            )
            self.assertTrue(result.is_file())
            with Image.open(result) as image:
                self.assertEqual(image.size, (300, 300))


if __name__ == "__main__":
    unittest.main()
