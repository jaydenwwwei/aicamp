from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


DAY3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY3))

from sportballs import (  # noqa: E402
    EXPECTED_CLASS_COUNT,
    ImageRecord,
    SessionState,
    SessionStore,
    balanced_warmup_records,
    build_round,
    choose_replay_records,
    discover_dataset,
    odd_one_index,
    open_rgb_image,
    predict_round,
    round_result_message,
    train_online_round,
    warmup_model,
)
from sportsball_model import SportsBallModel  # noqa: E402


class TinyImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 3),
        )

    def forward(self, images):
        return self.network(images)


def make_image(path: Path, color: tuple[int, int, int] = (120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


class SportsBallTests(unittest.TestCase):
    def test_discovers_all_train_and_test_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_index in range(EXPECTED_CLASS_COUNT):
                class_name = f"class_{class_index:02d}"
                make_image(root / "train" / class_name / "train.jpg")
                for image_index in range(3):
                    make_image(root / "test" / class_name / f"test_{image_index}.jpg")
            layout = discover_dataset(root)
            self.assertEqual(len(layout.class_names), EXPECTED_CLASS_COUNT)
            self.assertEqual(len(layout.train_records), EXPECTED_CLASS_COUNT)
            self.assertEqual(len(layout.game_records), EXPECTED_CLASS_COUNT * 3)

    def test_round_has_three_matching_and_one_odd_without_repeats(self):
        records = [
            ImageRecord(Path(f"{label}_{index}.jpg"), f"{label}_{index}", label, str(label))
            for label in range(3)
            for index in range(8)
        ]
        used: set[str] = set()
        first = build_round(records, used, seed=7, round_number=0)
        second = build_round(records, used, seed=7, round_number=1)
        for game_round in (first, second):
            labels = [record.label for record in game_round.records]
            self.assertEqual(labels.count(game_round.majority_label), 3)
            self.assertEqual(labels.count(game_round.odd_label), 1)
            self.assertEqual(labels[game_round.odd_index], game_round.odd_label)
        self.assertTrue(set(record.key for record in first.records).isdisjoint(
            record.key for record in second.records
        ))

    def test_round_is_seeded_and_recycles_an_exhausted_pool(self):
        records = [
            ImageRecord(Path(f"{label}_{index}.jpg"), f"{label}_{index}", label, str(label))
            for label in range(2)
            for index in range(4)
        ]
        left = build_round(records, set(), seed=99, round_number=2)
        right = build_round(records, set(), seed=99, round_number=2)
        self.assertEqual([item.key for item in left.records], [item.key for item in right.records])
        used = {record.key for record in records}
        recycled = build_round(records, used, seed=99, round_number=3)
        self.assertTrue(recycled.recycled_pool)
        self.assertEqual(len(used), 4)

    def test_hard_round_uses_a_confusing_class_pair(self):
        class_names = ["american_football", "rugby_ball", "shuttlecock"]
        records = [
            ImageRecord(
                Path(f"{class_name}_{index}.jpg"),
                f"{class_name}_{index}",
                label,
                class_name,
            )
            for label, class_name in enumerate(class_names)
            for index in range(6)
        ]
        game_round = build_round(
            records,
            set(),
            seed=11,
            round_number=0,
            hard_mode=True,
        )
        chosen_classes = {record.class_name for record in game_round.records}
        self.assertEqual(chosen_classes, {"american_football", "rugby_ball"})
        self.assertTrue(game_round.hard_mode)

    def test_odd_scoring_finds_distinct_probability_distribution(self):
        probabilities = torch.tensor(
            [
                [0.90, 0.08, 0.02],
                [0.88, 0.10, 0.02],
                [0.91, 0.07, 0.02],
                [0.02, 0.06, 0.92],
            ]
        )
        prediction, scores = odd_one_index(probabilities)
        self.assertEqual(prediction, 3)
        self.assertGreater(scores[3], max(scores[:3]))

    def test_result_message_shows_player_and_model_answers(self):
        self.assertEqual(
            round_result_message(player_choice=2, model_choice=3, correct_choice=3),
            "You: Wrong    |    Model: Correct",
        )
        self.assertEqual(
            round_result_message(player_choice=1, model_choice=0, correct_choice=1),
            "You: Correct    |    Model: Wrong",
        )

    def test_unreadable_image_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unreadable image"):
                open_rgb_image(path)

    def test_online_training_uses_record_labels_and_changes_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for label, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                path = root / f"{label}.jpg"
                make_image(path, color)
                records.append(ImageRecord(path, path.name, label, str(label)))
            model = TinyImageModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss = train_online_round(model, optimizer, records, torch.device("cpu"))
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_checkpoint_round_trip_includes_optimizer_and_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = model(torch.ones(2, 4)).sum()
            loss.backward()
            optimizer.step()
            state = SessionState(
                class_names=["a", "b", "c"],
                round_count=4,
                used_images=["test/a/1.jpg"],
                warmup_complete=True,
            )
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            store.save(model, optimizer, state)

            restored_model = torch.nn.Linear(4, 3)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored = store.load(
                restored_model,
                restored_optimizer,
                ["a", "b", "c"],
                torch.device("cpu"),
            )
            self.assertEqual(restored.round_count, 4)
            self.assertEqual(restored.used_images, ["test/a/1.jpg"])
            for name, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]))
            self.assertTrue(restored_optimizer.state)

    def test_reset_and_corrupt_state_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            state = SessionState(class_names=["a", "b"], warmup_complete=True)
            store.save(model, optimizer, state)
            store.state_path.write_text("{bad json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                store.load(model, optimizer, ["a", "b"], torch.device("cpu"))
            store.reset()
            self.assertFalse(store.exists())

    def test_replay_uses_only_known_true_labeled_records(self):
        records = [
            ImageRecord(Path(f"{index}.jpg"), str(index), index % 2, str(index % 2))
            for index in range(10)
        ]
        index = {record.key: record for record in records}
        selected = choose_replay_records(records[:4], [str(i) for i in range(4, 10)], index, 3, limit=3)
        self.assertEqual(selected[:4], records[:4])
        self.assertEqual(len(selected), 7)
        self.assertTrue(all(record.key in index for record in selected))

    def test_end_to_end_model_smoke_without_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for label in range(3):
                for image_index in range(3):
                    path = root / f"{label}_{image_index}.jpg"
                    color = [30, 30, 30]
                    color[label] = 180 + image_index * 10
                    make_image(path, tuple(color))
                    records.append(ImageRecord(path, path.name, label, str(label)))

            model = SportsBallModel(num_classes=3, pretrained=False)
            warmup_model(
                model,
                balanced_warmup_records(records, 1, 5),
                torch.device("cpu"),
                seed=5,
                epochs=1,
                limit_per_class=1,
            )
            game_round = build_round(records, set(), seed=5, round_number=0)
            prediction, probabilities = predict_round(model, game_round.records, torch.device("cpu"))
            self.assertIn(prediction, range(4))
            self.assertEqual(tuple(probabilities.shape), (4, 3))

            model.enable_online_learning()
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
            )
            train_online_round(model, optimizer, game_round.records, torch.device("cpu"))
            store = SessionStore(root / "artifacts")
            state = SessionState(class_names=["0", "1", "2"], warmup_complete=True, round_count=1)
            store.save(model, optimizer, state)

            restored_model = SportsBallModel(num_classes=3, pretrained=False)
            restored_model.enable_online_learning()
            restored_optimizer = torch.optim.AdamW(
                (parameter for parameter in restored_model.parameters() if parameter.requires_grad), lr=1e-4
            )
            restored = store.load(
                restored_model,
                restored_optimizer,
                ["0", "1", "2"],
                torch.device("cpu"),
            )
            self.assertEqual(restored.round_count, 1)


if __name__ == "__main__":
    unittest.main()
