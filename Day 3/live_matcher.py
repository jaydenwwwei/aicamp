"""Live webcam app that finds the most visually similar Lego minifigures."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import CLIPModel, CLIPProcessor

from label_dataset import DEFAULT_DATASET, load_index


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
INDEX_TENSOR_NAME = "lego_match_index.safetensors"
INDEX_METADATA_NAME = "lego_match_index.json"
INDEX_VERSION = 1
WINDOW_NAME = "Which Lego minifigure are you?"


class CameraError(RuntimeError):
    """Raised when the webcam cannot provide frames."""


@dataclass(frozen=True)
class Match:
    index: int
    score: float


def feature_tensor(output) -> torch.Tensor:
    """Support tensor results and the Transformers 5 pooled-output API."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def parse_set_name(raw_value: str) -> str:
    try:
        values = ast.literal_eval(raw_value)
        if isinstance(values, list) and values:
            return str(values[0])
    except (ValueError, SyntaxError):
        pass
    return raw_value.strip("[]'\"") or "Unknown set"


def load_identity_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Lego metadata not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        result[row["class_id"]] = {
            "minifigure_name": row["minifigure_name"],
            "set_name": parse_set_name(row["lego_names"]),
        }
    return result


def disambiguated_names(metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for value in metadata.values():
        name = value["minifigure_name"]
        counts[name] = counts.get(name, 0) + 1
    return {
        class_id: (
            f"{value['minifigure_name']} - {value['set_name']}"
            if counts[value["minifigure_name"]] > 1
            else value["minifigure_name"]
        )
        for class_id, value in metadata.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deduplicate_records(
    records: list[dict[str, str]],
    dataset_root: Path,
    digest_fn: Callable[[Path], str] = sha256,
) -> tuple[list[dict[str, str]], int]:
    unique = []
    seen = set()
    duplicates = 0
    for record in records:
        image_path = dataset_root / Path(record["path"])
        digest = digest_fn(image_path)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(record)
    return unique, duplicates


def aggregate_prototypes(
    embeddings: torch.Tensor,
    class_ids: list[str],
    paths: list[str],
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Average normalized image embeddings and select a representative image."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(class_ids) or len(paths) != len(class_ids):
        raise ValueError("Embeddings, class IDs, and paths must have matching rows")
    embeddings = normalize_embeddings(embeddings.float())
    ordered_ids = sorted(set(class_ids), key=lambda value: int(value))
    prototypes = []
    representatives = []
    for class_id in ordered_ids:
        indices = [index for index, value in enumerate(class_ids) if value == class_id]
        group = embeddings[indices]
        prototype = normalize_embeddings(group.mean(dim=0, keepdim=True))[0]
        representative_local_index = int((group @ prototype).argmax().item())
        prototypes.append(prototype)
        representatives.append(paths[indices[representative_local_index]])
    return torch.stack(prototypes), ordered_ids, representatives


def encode_images(
    records: list[dict[str, str]],
    dataset_root: Path,
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        for record in batch:
            with Image.open(dataset_root / Path(record["path"])) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            output = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            encoded.append(normalize_embeddings(feature_tensor(output)).cpu())
        print(f"Indexed {min(start + batch_size, len(records))}/{len(records)} Lego images")
    return torch.cat(encoded, dim=0)


def build_match_index(
    dataset_root: Path,
    index_dir: Path,
    model,
    processor,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict]:
    records = load_index(dataset_root)
    records, duplicate_count = deduplicate_records(records, dataset_root)
    print(f"Using {len(records)} unique images; removed {duplicate_count} exact duplicates.")
    embeddings = encode_images(records, dataset_root, model, processor, device, batch_size)
    prototypes, class_ids, representative_paths = aggregate_prototypes(
        embeddings,
        [record["class_id"] for record in records],
        [record["path"] for record in records],
    )

    identity_metadata = load_identity_metadata(dataset_root)
    display_names = disambiguated_names(identity_metadata)
    entries = []
    for class_id, representative_path in zip(class_ids, representative_paths):
        identity = identity_metadata[class_id]
        entries.append(
            {
                "class_id": class_id,
                "display_name": display_names[class_id],
                "minifigure_name": identity["minifigure_name"],
                "set_name": identity["set_name"],
                "reference_path": representative_path,
            }
        )

    metadata = {
        "version": INDEX_VERSION,
        "model_id": CLIP_MODEL_ID,
        "dataset_root": str(dataset_root.resolve()),
        "source_image_count": len(records) + duplicate_count,
        "unique_image_count": len(records),
        "duplicate_count": duplicate_count,
        "identity_count": len(entries),
        "entries": entries,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    save_file({"prototypes": prototypes.contiguous()}, str(index_dir / INDEX_TENSOR_NAME))
    (index_dir / INDEX_METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} Lego prototypes to {index_dir}")
    return prototypes, metadata


def load_match_index(index_dir: Path, device: torch.device) -> tuple[torch.Tensor, dict]:
    tensor_path = index_dir / INDEX_TENSOR_NAME
    metadata_path = index_dir / INDEX_METADATA_NAME
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Lego match index is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prototypes = load_file(str(tensor_path), device=str(device))["prototypes"]
    if prototypes.ndim != 2 or prototypes.shape[0] != len(metadata.get("entries", [])):
        raise ValueError("Lego match index metadata does not match its tensors")
    return normalize_embeddings(prototypes), metadata


def index_is_compatible(index_dir: Path, dataset_root: Path) -> bool:
    metadata_path = index_dir / INDEX_METADATA_NAME
    tensor_path = index_dir / INDEX_TENSOR_NAME
    if not metadata_path.is_file() or not tensor_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("version") == INDEX_VERSION
        and metadata.get("model_id") == CLIP_MODEL_ID
        and Path(metadata.get("dataset_root", "")) == dataset_root.resolve()
        and metadata.get("identity_count") == 38
    )


def rank_matches(query: torch.Tensor, prototypes: torch.Tensor, top_k: int = 3) -> list[Match]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query = normalize_embeddings(query.reshape(1, -1).float())[0]
    prototypes = normalize_embeddings(prototypes.float())
    if query.shape[0] != prototypes.shape[1]:
        raise ValueError("Query and prototype embedding dimensions do not match")
    scores = prototypes @ query
    count = min(top_k, prototypes.shape[0])
    values, indices = scores.topk(count)
    return [Match(int(index), float(score)) for index, score in zip(indices.cpu(), values.cpu())]


class LatestFrameSlot:
    """A one-item slot where newer frames replace frames not yet consumed."""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._closed = False

    def put(self, frame: np.ndarray) -> None:
        with self._condition:
            self._frame = frame.copy()
            self._condition.notify()

    def take(self):
        with self._condition:
            while self._frame is None and not self._closed:
                self._condition.wait()
            if self._closed:
                return None
            frame = self._frame
            self._frame = None
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class MatchWorker:
    def __init__(self, match_fn: Callable[[np.ndarray], list[Match]]):
        self.slot = LatestFrameSlot()
        self.match_fn = match_fn
        self._result_lock = threading.Lock()
        self._result: list[Match] = []
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="lego-match-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        self.slot.put(frame)

    def latest(self) -> tuple[list[Match], Exception | None]:
        with self._result_lock:
            return list(self._result), self._error

    def stop(self) -> None:
        self.slot.close()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        while True:
            frame = self.slot.take()
            if frame is None:
                return
            try:
                result = self.match_fn(frame)
                error = None
            except Exception as exc:  # keep the camera UI alive and surface the error there
                result = []
                error = exc
            with self._result_lock:
                self._result = result
                self._error = error


def open_camera(index: int, capture_factory=cv2.VideoCapture):
    capture = capture_factory(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(f"Could not open camera index {index}")
    return capture


def read_camera_frame(capture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise CameraError("The camera opened but did not return a frame")
    return frame


def load_reference_images(dataset_root: Path, metadata: dict, size: int = 112) -> list[np.ndarray]:
    references = []
    for entry in metadata["entries"]:
        image = cv2.imread(str(dataset_root / Path(entry["reference_path"])))
        if image is None:
            image = np.full((size, size, 3), 80, dtype=np.uint8)
        references.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
    return references


def fit_camera_frame(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - width) // 2
    y = (target_height - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas


def render_panel(
    frame: np.ndarray,
    matches: list[Match],
    metadata: dict,
    reference_images: list[np.ndarray],
    error: Exception | None = None,
) -> np.ndarray:
    height = max(480, frame.shape[0])
    camera_width = max(640, int(height * 4 / 3))
    panel_width = 430
    camera = fit_camera_frame(frame, height, camera_width)
    panel = np.full((height, panel_width, 3), (28, 28, 28), dtype=np.uint8)
    cv2.putText(panel, "Closest Lego matches", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if error is not None:
        cv2.putText(panel, "Matcher error", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        cv2.putText(panel, str(error)[:48], (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    elif not matches:
        cv2.putText(panel, "Analyzing...", (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    else:
        row_height = max(125, (height - 85) // max(1, len(matches)))
        for rank, match in enumerate(matches, start=1):
            entry = metadata["entries"][match.index]
            y = 52 + (rank - 1) * row_height
            thumbnail = reference_images[match.index]
            thumb_size = min(112, row_height - 12)
            thumbnail = cv2.resize(thumbnail, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            panel[y : y + thumb_size, 12 : 12 + thumb_size] = thumbnail
            text_x = 136
            display_name = entry["display_name"]
            if len(display_name) > 31:
                display_name = display_name[:28] + "..."
            cv2.putText(panel, f"#{rank} {display_name}", (text_x, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(
                panel,
                f"CLIP similarity: {match.score * 100:.1f}%",
                (text_x, y + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (110, 220, 255),
                1,
            )
    cv2.putText(panel, "Q quit   S save   R rebuild", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return np.hstack([camera, panel])


def save_composite(composite: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"lego_match_{timestamp}.jpg"
    if not cv2.imwrite(str(path), composite):
        raise OSError(f"Could not save match image: {path}")
    return path


def key_action(key: int) -> str | None:
    key = key & 0xFF
    return {ord("q"): "quit", ord("s"): "save", ord("r"): "rebuild"}.get(key)


def load_clip(device: torch.device):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).to(device).eval()
    return model, processor


def make_match_function(model, processor, prototypes: torch.Tensor, device: torch.device, top_k: int):
    def match_frame(frame: np.ndarray) -> list[Match]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            output = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
        query = feature_tensor(output)[0]
        return rank_matches(query, prototypes, top_k)

    return match_frame


def ensure_index(args, model, processor, device: torch.device, force: bool = False):
    if force or not index_is_compatible(args.index_dir, args.dataset_root):
        return build_match_index(args.dataset_root, args.index_dir, model, processor, device, args.batch_size)
    return load_match_index(args.index_dir, device)


def run_camera(args, model, processor, prototypes: torch.Tensor, metadata: dict, device: torch.device) -> int:
    capture = open_camera(args.camera_index)
    references = load_reference_images(args.dataset_root, metadata)
    worker = MatchWorker(make_match_function(model, processor, prototypes, device, args.top_k))
    worker.start()
    last_composite = None
    print("Camera controls: Q=quit, S=save result, R=rebuild Lego index")
    try:
        while True:
            frame = read_camera_frame(capture)
            worker.submit(frame)
            matches, error = worker.latest()
            composite = render_panel(frame, matches, metadata, references, error)
            last_composite = composite
            cv2.imshow(WINDOW_NAME, composite)
            action = key_action(cv2.waitKey(1))
            if action == "quit":
                return 0
            if action == "save" and last_composite is not None:
                print(f"Saved {save_composite(last_composite, args.output_dir)}")
            if action == "rebuild":
                worker.stop()
                prototypes, metadata = ensure_index(args, model, processor, device, force=True)
                references = load_reference_images(args.dataset_root, metadata)
                worker = MatchWorker(make_match_function(model, processor, prototypes, device, args.top_k))
                worker.start()
    finally:
        worker.stop()
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.top_k <= 3:
        raise ValueError("top-k must be between 1 and 3")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Lego dataset not found: {args.dataset_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP on {device}. Camera frames remain on this computer.")
    model, processor = load_clip(device)
    prototypes, metadata = ensure_index(args, model, processor, device, args.rebuild_index)
    print(f"Loaded {len(metadata['entries'])} Lego identities.")
    if args.build_index_only:
        return 0
    return run_camera(args, model, processor, prototypes, metadata, device)


if __name__ == "__main__":
    raise SystemExit(main())
