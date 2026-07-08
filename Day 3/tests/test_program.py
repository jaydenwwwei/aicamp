from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image


DAY3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY3))

from d3 import annotate_and_save, readable_image  # noqa: E402
from label_dataset import grouped_splits  # noqa: E402
from lego_model import LegoMultiTaskModel  # noqa: E402
from train_model import build_parser as build_training_parser, metrics_from_predictions  # noqa: E402
from live_matcher import (  # noqa: E402
    CameraError,
    LatestFrameSlot,
    Match,
    aggregate_prototypes,
    deduplicate_records,
    disambiguated_names,
    key_action,
    open_camera,
    rank_matches,
    read_camera_frame,
    render_panel,
    save_composite,
)


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

    def test_match_index_creates_38_normalized_prototypes(self):
        class_ids = [str(index) for index in range(1, 39) for _ in range(2)]
        paths = [f"{class_id}/{copy}.jpg" for class_id in range(1, 39) for copy in range(2)]
        embeddings = torch.arange(76 * 8, dtype=torch.float32).reshape(76, 8) + 1
        prototypes, ordered_ids, representatives = aggregate_prototypes(embeddings, class_ids, paths)
        self.assertEqual(tuple(prototypes.shape), (38, 8))
        self.assertEqual(len(ordered_ids), 38)
        self.assertEqual(len(representatives), 38)
        self.assertTrue(torch.allclose(prototypes.norm(dim=1), torch.ones(38), atol=1e-5))

    def test_exact_duplicates_are_removed(self):
        records = [{"path": "a.jpg"}, {"path": "b.jpg"}, {"path": "c.jpg"}]
        digests = {"a.jpg": "same", "b.jpg": "same", "c.jpg": "different"}
        unique, count = deduplicate_records(
            records,
            Path("."),
            digest_fn=lambda path: digests[path.name],
        )
        self.assertEqual([row["path"] for row in unique], ["a.jpg", "c.jpg"])
        self.assertEqual(count, 1)

    def test_cosine_ranking_returns_top_three(self):
        prototypes = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]])
        matches = rank_matches(torch.tensor([1.0, 0.0]), prototypes, top_k=3)
        self.assertEqual([match.index for match in matches], [0, 1, 2])
        self.assertGreater(matches[0].score, matches[1].score)

    def test_duplicate_character_names_include_set(self):
        metadata = {
            "1": {"minifigure_name": "SPIDER-MAN", "set_name": "Set A"},
            "2": {"minifigure_name": "SPIDER-MAN", "set_name": "Set B"},
            "3": {"minifigure_name": "YODA", "set_name": "Set C"},
        }
        names = disambiguated_names(metadata)
        self.assertEqual(names["1"], "SPIDER-MAN - Set A")
        self.assertEqual(names["2"], "SPIDER-MAN - Set B")
        self.assertEqual(names["3"], "YODA")

    def test_latest_frame_slot_replaces_stale_frame(self):
        slot = LatestFrameSlot()
        slot.put(np.full((2, 2, 3), 1, dtype=np.uint8))
        slot.put(np.full((2, 2, 3), 2, dtype=np.uint8))
        latest = slot.take()
        slot.close()
        self.assertTrue(np.all(latest == 2))

    def test_camera_failures_are_clear(self):
        class ClosedCamera:
            def isOpened(self):
                return False

            def release(self):
                pass

        with self.assertRaisesRegex(CameraError, "Could not open"):
            open_camera(0, capture_factory=lambda _: ClosedCamera())

        class EmptyCamera:
            def read(self):
                return False, None

        with self.assertRaisesRegex(CameraError, "did not return"):
            read_camera_frame(EmptyCamera())

    def test_match_panel_and_save(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metadata = {
            "entries": [
                {"display_name": "YODA", "reference_path": "unused.jpg"},
                {"display_name": "DARTH VADER", "reference_path": "unused.jpg"},
                {"display_name": "IRON MAN", "reference_path": "unused.jpg"},
            ]
        }
        references = [np.full((112, 112, 3), value, dtype=np.uint8) for value in (40, 80, 120)]
        matches = [Match(0, 0.4), Match(1, 0.3), Match(2, 0.2)]
        panel = render_panel(frame, matches, metadata, references)
        self.assertEqual(panel.shape, (480, 1070, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = save_composite(panel, Path(directory))
            self.assertTrue(path.is_file())

    def test_live_matcher_keys(self):
        self.assertEqual(key_action(ord("q")), "quit")
        self.assertEqual(key_action(ord("s")), "save")
        self.assertEqual(key_action(ord("r")), "rebuild")
        self.assertIsNone(key_action(ord("x")))


if __name__ == "__main__":
    unittest.main()
