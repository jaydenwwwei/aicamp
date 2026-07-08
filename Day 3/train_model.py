"""Fine-tune MobileNetV3 for Lego direction and head-color prediction."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from lego_model import (
    COLOR_CLASSES,
    DIRECTION_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LegoMultiTaskModel,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path.home() / ".cache/kagglehub/datasets/ihelon/lego-minifigures-classification/versions/30"
DEFAULT_LABELS = SCRIPT_DIR / "labels.csv"
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts"


class LegoDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, transform):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / Path(row["path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            DIRECTION_CLASSES.index(row["direction"]),
            COLOR_CLASSES.index(row["head_color"]),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels not found: {path}. Run label_dataset.py first.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "class_id", "split", "direction", "head_color", "review_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Labels CSV is empty or missing columns: {sorted(required)}")

    invalid_direction = sorted({row["direction"] for row in rows} - set(DIRECTION_CLASSES))
    invalid_color = sorted({row["head_color"] for row in rows} - set(COLOR_CLASSES))
    if invalid_direction or invalid_color:
        raise ValueError(f"Unknown labels: directions={invalid_direction}, colors={invalid_color}")

    unresolved = [row for row in rows if row["review_status"] == "needs_review"]
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} labels have not been finalized. Run label_dataset.py once more to "
            "convert them to unverified pseudo-labels."
        )
    usable = [
        row
        for row in rows
        if row["review_status"] in {"auto_accepted", "pseudo_labeled", "reviewed"}
    ]
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in usable):
            raise ValueError(f"No usable rows found for the {split} split")

    group_splits: dict[str, set[str]] = {}
    for row in usable:
        group_splits.setdefault(row["class_id"], set()).add(row["split"])
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Character IDs appear in multiple splits: {leaked[:5]}")
    return usable


def validate_training_coverage(train_rows: list[dict[str, str]]) -> None:
    missing_directions = sorted(set(DIRECTION_CLASSES) - {row["direction"] for row in train_rows})
    missing_colors = sorted(set(COLOR_CLASSES) - {row["head_color"] for row in train_rows})
    if missing_directions or missing_colors:
        raise ValueError(
            "Training data does not cover every output class. "
            f"Missing directions={missing_directions}; missing colors={missing_colors}. "
            "Correct labels or collect more images before training."
        )


def class_weights(rows: list[dict[str, str]], key: str, classes: list[str]) -> torch.Tensor:
    counts = Counter(row[key] for row in rows)
    total = len(rows)
    return torch.tensor(
        [(total / (len(classes) * counts[value])) ** 0.5 for value in classes],
        dtype=torch.float32,
    )


def sample_weights(rows: list[dict[str, str]]) -> list[float]:
    direction_counts = Counter(row["direction"] for row in rows)
    color_counts = Counter(row["head_color"] for row in rows)
    return [
        (1.0 / direction_counts[row["direction"]] * 1.0 / color_counts[row["head_color"]]) ** 0.5
        for row in rows
    ]


def confusion_matrix(targets: list[int], predictions: list[int], class_count: int) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def metrics_from_predictions(targets: list[int], predictions: list[int], class_count: int) -> dict:
    matrix = confusion_matrix(targets, predictions, class_count)
    correct = sum(matrix[index][index] for index in range(class_count))
    f1_scores = []
    for index in range(class_count):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(class_count) if row != index)
        false_negative = sum(matrix[index][column] for column in range(class_count) if column != index)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": correct / max(1, len(targets)),
        "macro_f1": sum(f1_scores) / class_count,
        "per_class_f1": f1_scores,
        "confusion_matrix": matrix,
    }


def run_epoch(model, loader, device, direction_loss, color_loss, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    direction_targets: list[int] = []
    direction_predictions: list[int] = []
    color_targets: list[int] = []
    color_predictions: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, directions, colors in loader:
            images = images.to(device)
            directions = directions.to(device)
            colors = colors.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            direction_logits, color_logits = model(images)
            loss = direction_loss(direction_logits, directions) + color_loss(color_logits, colors)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            direction_targets.extend(directions.cpu().tolist())
            direction_predictions.extend(direction_logits.argmax(dim=1).cpu().tolist())
            color_targets.extend(colors.cpu().tolist())
            color_predictions.extend(color_logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "direction": metrics_from_predictions(
            direction_targets, direction_predictions, len(DIRECTION_CLASSES)
        ),
        "color": metrics_from_predictions(color_targets, color_predictions, len(COLOR_CLASSES)),
    }


def loaders(rows: list[dict[str, str]], dataset_root: Path, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    split_rows = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    validate_training_coverage(split_rows["train"])
    datasets = {
        name: LegoDataset(values, dataset_root, train_transform if name == "train" else eval_transform)
        for name, values in split_rows.items()
    }
    sampler = WeightedRandomSampler(
        sample_weights(split_rows["train"]),
        num_samples=len(split_rows["train"]),
        replacement=True,
    )
    return (
        split_rows,
        {
            "train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0),
            "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
            "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
        },
    )


def print_confusion(title: str, classes: list[str], matrix: list[list[int]]) -> None:
    print(f"\n{title} confusion matrix (rows=true, columns=predicted)")
    print("classes: " + ", ".join(f"{index}={value}" for index, value in enumerate(classes)))
    for index, row in enumerate(matrix):
        print(f"{index:>2}: " + " ".join(f"{value:>4}" for value in row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue fine-tuning from artifacts/lego_mobilenet.safetensors.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not args.resume and not 0 <= args.freeze_epochs <= args.epochs:
        raise ValueError("epochs must be positive and freeze-epochs must be between 0 and epochs")
    seed_everything(args.seed)
    rows = load_rows(args.labels)
    split_rows, data_loaders = loaders(rows, args.dataset_root, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}: " + ", ".join(f"{key}={len(value)}" for key, value in split_rows.items()))

    args.artifacts.mkdir(parents=True, exist_ok=True)
    best_path = args.artifacts / "lego_mobilenet.safetensors"
    metrics_path = args.artifacts / "metrics.json"
    previous_report = {}
    if args.resume:
        if not best_path.is_file():
            raise FileNotFoundError(f"Cannot resume because no checkpoint exists: {best_path}")
        if metrics_path.is_file():
            previous_report = json.loads(metrics_path.read_text(encoding="utf-8"))

    model = LegoMultiTaskModel(
        len(DIRECTION_CLASSES),
        len(COLOR_CLASSES),
        pretrained=not args.resume,
    ).to(device)
    if args.resume:
        model.load_state_dict(load_file(str(best_path), device=str(device)))
        model.unfreeze_backbone()
        print(f"Continuing from {best_path} with the full network unfrozen.")
    elif args.freeze_epochs:
        model.freeze_backbone()
    effective_learning_rate = args.learning_rate / 10 if args.resume else args.learning_rate
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=effective_learning_rate,
        weight_decay=1e-4,
    )
    direction_loss = nn.CrossEntropyLoss(
        weight=class_weights(split_rows["train"], "direction", DIRECTION_CLASSES).to(device)
    )
    color_loss = nn.CrossEntropyLoss(
        weight=class_weights(split_rows["train"], "head_color", COLOR_CLASSES).to(device)
    )

    best_score = float(previous_report.get("best_validation_score", -1.0))
    stale_epochs = 0
    history = list(previous_report.get("history", []))
    starting_epoch = max((int(item.get("epoch", 0)) for item in history), default=0)
    previous_runs = previous_report.get("training_runs")
    if previous_runs is None:
        previous_runs = 1 if history else 0
    training_run = int(previous_runs) + 1
    for epoch in range(args.epochs):
        if not args.resume and epoch == args.freeze_epochs and args.freeze_epochs < args.epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate / 10, weight_decay=1e-4)
        train_metrics = run_epoch(model, data_loaders["train"], device, direction_loss, color_loss, optimizer)
        val_metrics = run_epoch(model, data_loaders["val"], device, direction_loss, color_loss)
        score = (val_metrics["direction"]["macro_f1"] + val_metrics["color"]["macro_f1"]) / 2
        epoch_number = starting_epoch + epoch + 1
        history.append(
            {
                "epoch": epoch_number,
                "training_run": training_run,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"epoch {epoch_number:02d}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_score={score:.3f}"
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            safe_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            save_file(safe_state, str(best_path))
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation score did not improve.")
                break

    model.load_state_dict(load_file(str(best_path), device=str(device)))
    test_metrics = run_epoch(model, data_loaders["test"], device, direction_loss, color_loss)
    report = {
        "dataset_counts": {key: len(value) for key, value in split_rows.items()},
        "best_validation_score": best_score,
        "training_runs": training_run,
        "last_run_resumed": args.resume,
        "optimizer_state_restored": False,
        "test": test_metrics,
        "evaluation": {
            "label_status": "clip_pseudo_labels_unverified",
            "ground_truth_accuracy_available": False,
            "warning": "Test scores measure agreement with CLIP pseudo-labels, not real-world accuracy.",
        },
        "history": history,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    config = {
        "direction_classes": DIRECTION_CLASSES,
        "color_classes": COLOR_CLASSES,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "uncertain_threshold": 0.60,
        "architecture": "mobilenet_v3_small_multitask",
    }
    (args.artifacts / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print_confusion("Direction", DIRECTION_CLASSES, test_metrics["direction"]["confusion_matrix"])
    print_confusion("Head color", COLOR_CLASSES, test_metrics["color"]["confusion_matrix"])
    print(
        f"\nPseudo-label agreement: direction={test_metrics['direction']['accuracy']:.1%}; "
        f"color={test_metrics['color']['accuracy']:.1%}. These are not ground-truth accuracy scores."
    )
    print(f"Artifacts saved to {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
