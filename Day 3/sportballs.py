"""Play a four-picture sports-ball odd-one-out game that learns each round."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sportsball_model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SportsBallModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts" / "sportsball_game"
DATASET_HANDLE = "samuelcortinhas/sports-balls-multiclass-image-classification"
EXPECTED_CLASS_COUNT = 15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HARD_CLASS_GROUPS = (
    frozenset({"american_football", "rugby_ball"}),
    frozenset({"baseball", "cricket_ball"}),
    frozenset({"tennis_ball", "table_tennis_ball"}),
    frozenset({"hockey_ball", "hockey_puck"}),
    frozenset({"football", "volleyball", "basketball"}),
    frozenset({"golf_ball", "billiard_ball", "bowling_ball"}),
)


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def round_result_message(player_choice: int, model_choice: int, correct_choice: int) -> str:
    """Describe both players' results without exposing training details."""
    player_result = "Correct" if player_choice == correct_choice else "Wrong"
    model_result = "Correct" if model_choice == correct_choice else "Wrong"
    return f"You: {player_result}    |    Model: {model_result}"


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    key: str
    label: int
    class_name: str


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    class_names: tuple[str, ...]
    train_records: tuple[ImageRecord, ...]
    game_records: tuple[ImageRecord, ...]


@dataclass(frozen=True)
class GameRound:
    records: tuple[ImageRecord, ...]
    odd_index: int
    majority_label: int
    odd_label: int
    recycled_pool: bool = False
    hard_mode: bool = False


@dataclass
class SessionState:
    schema_version: int = 1
    class_names: list[str] = field(default_factory=list)
    seed: int = 42
    round_count: int = 0
    pool_cycle: int = 0
    used_images: list[str] = field(default_factory=list)
    replay_images: list[str] = field(default_factory=list)
    human_correct: int = 0
    model_correct: int = 0
    warmup_complete: bool = False

    @classmethod
    def from_json(cls, payload: dict) -> "SessionState":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_dataset(root: Path) -> DatasetLayout:
    """Validate train/test class folders and return stable image records."""
    root = Path(root).expanduser().resolve()
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise ValueError(f"Dataset must contain train and test folders: {root}")

    train_classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if train_classes != test_classes:
        raise ValueError("The train and test folders must contain the same class names.")
    if len(train_classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} sports-ball classes, found {len(train_classes)}."
        )

    def records_for(split: str) -> tuple[ImageRecord, ...]:
        records: list[ImageRecord] = []
        for label, class_name in enumerate(train_classes):
            files = _image_files(root / split / class_name)
            minimum = 3 if split == "test" else 1
            if len(files) < minimum:
                raise ValueError(
                    f"Class '{class_name}' needs at least {minimum} readable image files in {split}."
                )
            records.extend(
                ImageRecord(
                    path=path,
                    key=path.relative_to(root).as_posix(),
                    label=label,
                    class_name=class_name,
                )
                for path in files
            )
        return tuple(records)

    return DatasetLayout(root, tuple(train_classes), records_for("train"), records_for("test"))


def resolve_dataset_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install Day 3/requirements.txt or pass --dataset-root."
        ) from exc
    try:
        return Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
    except Exception as exc:
        raise RuntimeError(f"Could not download or locate the sports-ball dataset: {exc}") from exc


def open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def build_round(
    records: Sequence[ImageRecord],
    used_keys: set[str],
    seed: int,
    round_number: int,
    pool_cycle: int = 0,
    hard_mode: bool = False,
) -> GameRound:
    """Create a deterministic 3+1 round, clearing an exhausted pool when needed."""
    rng = random.Random(seed + round_number * 1_000_003 + pool_cycle * 97_409)

    def available_by_label() -> dict[int, list[ImageRecord]]:
        grouped: dict[int, list[ImageRecord]] = {}
        for record in records:
            if record.key not in used_keys:
                grouped.setdefault(record.label, []).append(record)
        return grouped

    class_by_label = {record.label: record.class_name for record in records}

    def possible_odd_labels(majority_label: int, grouped: dict[int, list[ImageRecord]]) -> list[int]:
        candidates = [
            label for label, items in grouped.items() if label != majority_label and items
        ]
        if not hard_mode:
            return sorted(candidates)
        majority_class = class_by_label[majority_label]
        hard_neighbors = next(
            (group - {majority_class} for group in HARD_CLASS_GROUPS if majority_class in group),
            frozenset(),
        )
        return sorted(label for label in candidates if class_by_label[label] in hard_neighbors)

    def possible_majorities(grouped: dict[int, list[ImageRecord]]) -> list[int]:
        return sorted(
            label
            for label, items in grouped.items()
            if len(items) >= 3 and possible_odd_labels(label, grouped)
        )

    grouped = available_by_label()
    valid_majorities = possible_majorities(grouped)
    recycled = False
    if not valid_majorities:
        used_keys.clear()
        grouped = available_by_label()
        valid_majorities = possible_majorities(grouped)
        recycled = True
    if not valid_majorities:
        raise ValueError("The game pool cannot make a round with three matching and one odd image.")

    majority_label = rng.choice(sorted(valid_majorities))
    odd_labels = possible_odd_labels(majority_label, grouped)
    odd_label = rng.choice(odd_labels)
    chosen = rng.sample(grouped[majority_label], 3) + [rng.choice(grouped[odd_label])]
    rng.shuffle(chosen)
    odd_index = next(index for index, record in enumerate(chosen) if record.label == odd_label)
    used_keys.update(record.key for record in chosen)
    return GameRound(tuple(chosen), odd_index, majority_label, odd_label, recycled, hard_mode)


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform: Callable):
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        return self.transform(open_rgb_image(record.path)), record.label


def balanced_warmup_records(
    records: Sequence[ImageRecord], limit_per_class: int, seed: int
) -> list[ImageRecord]:
    grouped: dict[int, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    rng = random.Random(seed)
    selected: list[ImageRecord] = []
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        selected.extend(items[:limit_per_class])
    rng.shuffle(selected)
    return selected


def warmup_model(
    model: SportsBallModel,
    records: Sequence[ImageRecord],
    device: torch.device,
    seed: int,
    epochs: int = 3,
    limit_per_class: int = 100,
    progress: Callable[[str], None] | None = None,
) -> None:
    torch.manual_seed(seed)
    model.freeze_backbone()
    model.to(device).train()
    selected = balanced_warmup_records(records, limit_per_class, seed)
    loader = DataLoader(
        RecordDataset(selected, training_transform()), batch_size=32, shuffle=True, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
        if progress:
            progress(
                f"Initial training: epoch {epoch + 1}/{epochs} "
                f"(loss {running_loss / max(1, len(selected)):.3f})"
            )


def jensen_shannon_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    left = left.clamp_min(epsilon)
    right = right.clamp_min(epsilon)
    middle = (left + right) / 2
    return 0.5 * (
        torch.sum(left * (left.log() - middle.log()))
        + torch.sum(right * (right.log() - middle.log()))
    )


def odd_one_index(probabilities: torch.Tensor) -> tuple[int, list[float]]:
    """Return the item whose class distribution differs most from the other three."""
    if probabilities.ndim != 2 or probabilities.shape[0] < 3:
        raise ValueError("Odd-one scoring needs at least three class-probability rows.")
    row_sums = probabilities.sum(dim=1, keepdim=True)
    probabilities = probabilities / row_sums.clamp_min(1e-8)
    scores = []
    for index in range(len(probabilities)):
        score = sum(
            jensen_shannon_divergence(probabilities[index], probabilities[other])
            for other in range(len(probabilities))
            if other != index
        )
        scores.append(float(score.item()))
    return max(range(len(scores)), key=scores.__getitem__), scores


def predict_round(
    model: SportsBallModel, records: Sequence[ImageRecord], device: torch.device
) -> tuple[int, torch.Tensor]:
    transform = evaluation_transform()
    batch = torch.stack([transform(open_rgb_image(record.path)) for record in records]).to(device)
    model.to(device).eval()
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1).cpu()
    prediction, _ = odd_one_index(probabilities)
    return prediction, probabilities


def make_online_optimizer(model: SportsBallModel) -> torch.optim.Optimizer:
    model.enable_online_learning()
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )


def train_online_round(
    model: SportsBallModel,
    optimizer: torch.optim.Optimizer,
    records: Sequence[ImageRecord],
    device: torch.device,
) -> float:
    """Learn from true folder labels; no player selection is accepted here."""
    if not records:
        raise ValueError("Online training needs at least one labeled image.")
    model.to(device).train()
    loader = DataLoader(
        RecordDataset(records, training_transform()), batch_size=min(8, len(records)), shuffle=True
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_items += len(labels)
    return total_loss / total_items


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


class SessionStore:
    def __init__(self, directory: Path = DEFAULT_ARTIFACT_DIR):
        self.directory = Path(directory)
        self.model_path = self.directory / "model.safetensors"
        self.optimizer_path = self.directory / "optimizer.safetensors"
        self.state_path = self.directory / "session.json"

    def reset(self) -> None:
        for path in (self.model_path, self.optimizer_path, self.state_path):
            path.unlink(missing_ok=True)

    def exists(self) -> bool:
        return all(path.is_file() for path in (self.model_path, self.optimizer_path, self.state_path))

    def save(
        self,
        model: SportsBallModel,
        optimizer: torch.optim.Optimizer,
        state: SessionState,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_safetensors(self.model_path, model.state_dict())
        named_parameters = dict(model.named_parameters())
        parameter_names = {parameter: name for name, parameter in named_parameters.items()}
        optimizer_tensors: dict[str, torch.Tensor] = {}
        for parameter, values in optimizer.state.items():
            name = parameter_names.get(parameter)
            if name is None:
                continue
            for state_name, value in values.items():
                if torch.is_tensor(value):
                    optimizer_tensors[f"{name}::{state_name}"] = value
        if not optimizer_tensors:
            optimizer_tensors["__empty__"] = torch.zeros(1)
        _atomic_safetensors(self.optimizer_path, optimizer_tensors)
        _atomic_json(self.state_path, asdict(state))

    def load(
        self,
        model: SportsBallModel,
        optimizer: torch.optim.Optimizer,
        expected_classes: Sequence[str],
        device: torch.device,
    ) -> SessionState:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = SessionState.from_json(payload)
        if state.schema_version != 1:
            raise ValueError("Unsupported sports-ball checkpoint version.")
        if state.class_names != list(expected_classes):
            raise ValueError("Saved model classes do not match this dataset.")
        model.load_state_dict(load_file(str(self.model_path), device=str(device)))

        saved_optimizer = load_file(str(self.optimizer_path), device=str(device))
        for name, parameter in model.named_parameters():
            prefix = f"{name}::"
            values = {
                key.removeprefix(prefix): tensor.to(device)
                for key, tensor in saved_optimizer.items()
                if key.startswith(prefix)
            }
            if values:
                optimizer.state[parameter] = values
        return state


def choose_replay_records(
    current: Sequence[ImageRecord],
    replay_keys: Sequence[str],
    record_index: dict[str, ImageRecord],
    seed: int,
    limit: int = 28,
) -> list[ImageRecord]:
    available = [record_index[key] for key in replay_keys if key in record_index]
    rng = random.Random(seed)
    rng.shuffle(available)
    return list(current) + available[:limit]


class SportsBallGame:
    IMAGE_BOX = (300, 220)

    def __init__(self, root, dataset_root: Path | None, seed: int, reset: bool):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.dataset_root = dataset_root
        self.seed = seed
        self.store = SessionStore()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layout: DatasetLayout | None = None
        self.model: SportsBallModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.state: SessionState | None = None
        self.current_round: GameRound | None = None
        self.current_prediction: int | None = None
        self.result_message = ""
        self.hard_mode = False
        self.photos: list[ImageTk.PhotoImage] = []
        self.buttons: list = []
        self.closed = False

        if reset:
            self.store.reset()
        self.root.title("Sports-Ball Odd One Out")
        self.root.geometry("800x720")
        self.root.minsize(740, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.container = tk.Frame(root, padx=18, pady=14)
        self.container.pack(fill="both", expand=True)
        self.title_label = tk.Label(
            self.container, text="Odd One Out", font=("Segoe UI", 24, "bold")
        )
        self.title_label.pack(pady=(0, 4))
        self.status = tk.Label(
            self.container,
            text="Preparing the dataset...",
            font=("Segoe UI", 12),
            wraplength=680,
        )
        self.status.pack(pady=(0, 10))
        self.mode_button = tk.Button(
            self.container,
            text="Mode: Normal  •  Click for Hard",
            font=("Segoe UI", 11, "bold"),
            foreground="#245c2a",
            command=self.toggle_hard_mode,
        )
        self.mode_button.pack(pady=(0, 7))
        self.score_label = tk.Label(
            self.container,
            text="You: 0/0    |    Model: 0/0",
            font=("Segoe UI", 11, "bold"),
            foreground="#3949ab",
        )
        self.score_label.pack(pady=(0, 8))
        self.grid = tk.Frame(self.container)
        self.grid.pack(expand=True)
        self.next_button = tk.Button(
            self.container,
            text="Next Round",
            font=("Segoe UI", 12, "bold"),
            command=self.prepare_next_round,
        )
        self.retry_button = tk.Button(
            self.container, text="Retry", font=("Segoe UI", 12), command=self.start_initialization
        )
        self.start_initialization()

    def close(self) -> None:
        self.closed = True
        self.root.destroy()

    def set_status_from_worker(self, text: str) -> None:
        if not self.closed:
            self.root.after(0, lambda: self.status.config(text=text) if not self.closed else None)

    def update_scoreboard(self) -> None:
        if self.state is None:
            self.score_label.config(text="You: 0/0    |    Model: 0/0")
            return
        rounds = self.state.round_count
        self.score_label.config(
            text=(
                f"You: {self.state.human_correct}/{rounds}    |    "
                f"Model: {self.state.model_correct}/{rounds}"
            )
        )

    def toggle_hard_mode(self) -> None:
        self.hard_mode = not self.hard_mode
        if self.hard_mode:
            self.mode_button.config(
                text="Mode: HARD  •  Click for Normal",
                foreground="#a52a2a",
            )
            message = "Hard mode enabled: the next round uses similar-looking sports objects."
        else:
            self.mode_button.config(
                text="Mode: Normal  •  Click for Hard",
                foreground="#245c2a",
            )
            message = "Normal mode enabled for the next round."
        if self.current_round is None:
            self.status.config(text=message)

    def run_worker(self, operation: Callable, success: Callable) -> None:
        def target():
            try:
                result = operation()
            except Exception as exc:
                if not self.closed:
                    self.root.after(0, lambda error=exc: self.show_error(error))
            else:
                if not self.closed:
                    self.root.after(0, lambda: success(result))

        threading.Thread(target=target, daemon=True).start()

    def start_initialization(self) -> None:
        self.retry_button.pack_forget()
        self.next_button.pack_forget()
        self.status.config(text="Preparing the dataset...")
        self.run_worker(self.initialize, lambda _: self.prepare_next_round())

    def initialize(self) -> None:
        dataset_path = resolve_dataset_root(self.dataset_root)
        self.layout = discover_dataset(dataset_path)
        self.model = SportsBallModel(len(self.layout.class_names), pretrained=True)
        self.model.to(self.device)
        self.optimizer = make_online_optimizer(self.model)
        warning = None
        if self.store.exists():
            try:
                self.state = self.store.load(
                    self.model, self.optimizer, self.layout.class_names, self.device
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                warning = f"Saved progress could not be loaded; starting fresh. ({exc})"
                self.state = None
        if self.state is None:
            self.state = SessionState(class_names=list(self.layout.class_names), seed=self.seed)
            self.set_status_from_worker(warning or "Starting the model's short first training session...")
            warmup_model(
                self.model,
                self.layout.train_records,
                self.device,
                self.seed,
                progress=self.set_status_from_worker,
            )
            self.state.warmup_complete = True
            self.optimizer = make_online_optimizer(self.model)
            self.store.save(self.model, self.optimizer, self.state)

    def prepare_next_round(self) -> None:
        self.next_button.pack_forget()
        self.clear_grid()
        self.current_round = None
        selected_hard_mode = self.hard_mode
        mode_name = "hard" if selected_hard_mode else "normal"
        self.status.config(text=f"Choosing a {mode_name} round and asking the model...")

        def prepare():
            assert self.layout and self.model and self.state
            used = set(self.state.used_images)
            game_round = build_round(
                self.layout.game_records,
                used,
                self.state.seed,
                self.state.round_count,
                self.state.pool_cycle,
                selected_hard_mode,
            )
            if game_round.recycled_pool:
                self.state.pool_cycle += 1
            self.state.used_images = sorted(used)
            prediction, _ = predict_round(self.model, game_round.records, self.device)
            pictures = [open_rgb_image(record.path) for record in game_round.records]
            return game_round, prediction, pictures

        self.run_worker(prepare, self.show_round)

    def clear_grid(self) -> None:
        for widget in self.grid.winfo_children():
            widget.destroy()
        self.buttons.clear()
        self.photos.clear()

    def show_round(self, result) -> None:
        game_round, prediction, pictures = result
        self.current_round = game_round
        self.current_prediction = prediction
        self.update_scoreboard()
        prompt = "Which picture is the odd one out? Click it!"
        if game_round.hard_mode:
            prompt += "  (HARD: these sports objects are deliberately similar.)"
        self.status.config(text=prompt)
        for index, picture in enumerate(pictures):
            fitted = ImageOps.contain(picture, self.IMAGE_BOX)
            background = Image.new("RGB", self.IMAGE_BOX, "white")
            position = (
                (self.IMAGE_BOX[0] - fitted.width) // 2,
                (self.IMAGE_BOX[1] - fitted.height) // 2,
            )
            background.paste(fitted, position)
            photo = ImageTk.PhotoImage(background)
            self.photos.append(photo)
            button = self.tk.Button(
                self.grid,
                image=photo,
                command=lambda selected=index: self.select_image(selected),
                relief="solid",
                borderwidth=2,
                highlightthickness=4,
                highlightbackground="#d0d0d0",
                highlightcolor="#d0d0d0",
                cursor="hand2",
                compound="top",
                font=("Segoe UI", 10, "bold"),
                text="",
            )
            button.grid(row=index // 2, column=index % 2, padx=8, pady=8)
            self.buttons.append(button)

    def select_image(self, selected: int) -> None:
        if self.current_round is None or self.current_prediction is None or self.state is None:
            return
        correct = self.current_round.odd_index
        model_choice = self.current_prediction
        for index, button in enumerate(self.buttons):
            button.config(state="disabled", cursor="")
            labels: list[str] = []
            if index == correct:
                labels.append("Correct")
            if index == selected:
                labels.append("You")
            if index == model_choice:
                labels.append("Model")
            button.config(text=" • ".join(labels))

            if index == correct:
                color = "#2e9d48"  # green: true odd image
            elif index == selected and index == model_choice:
                color = "#ef6c00"  # orange: both chose the same wrong image
            elif index == selected:
                color = "#c63c3c"  # red: player's wrong choice
            elif index == model_choice:
                color = "#7e57c2"  # purple: model's wrong choice
            else:
                color = "#d0d0d0"
            button.config(highlightbackground=color, highlightcolor=color)

        self.result_message = round_result_message(selected, model_choice, correct)
        self.status.config(text=self.result_message + "\nThe model is learning from this round...")

        self.state.round_count += 1
        self.state.human_correct += int(selected == correct)
        self.state.model_correct += int(self.current_prediction == correct)
        self.update_scoreboard()
        for record in self.current_round.records:
            if record.key not in self.state.replay_images:
                self.state.replay_images.append(record.key)

        self.run_worker(self.learn_and_save, self.training_finished)

    def learn_and_save(self) -> float:
        assert self.layout and self.model and self.optimizer and self.state and self.current_round
        record_index = {record.key: record for record in self.layout.game_records}
        current_keys = {record.key for record in self.current_round.records}
        earlier_replay = [key for key in self.state.replay_images if key not in current_keys]
        training_records = choose_replay_records(
            self.current_round.records,
            earlier_replay,
            record_index,
            self.state.seed + self.state.round_count,
        )
        loss = train_online_round(self.model, self.optimizer, training_records, self.device)
        self.store.save(self.model, self.optimizer, self.state)
        return loss

    def training_finished(self, _loss: float) -> None:
        self.status.config(text=self.result_message + "\nThe model learned from these pictures.")
        self.next_button.pack(pady=(8, 0))

    def show_error(self, error: Exception) -> None:
        self.clear_grid()
        self.next_button.pack_forget()
        self.status.config(text=f"Could not continue: {error}")
        self.retry_button.pack(pady=12)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Existing dataset folder containing train/test.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for training and game rounds.")
    parser.add_argument("--reset", action="store_true", help="Delete saved sports-ball progress first.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import tkinter as tk
    except ImportError:
        print("Error: Tkinter is required to run the desktop game.", file=sys.stderr)
        return 1
    root = tk.Tk()
    SportsBallGame(root, args.dataset_root, args.seed, args.reset)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
