"""Create CLIP-assisted Lego direction/color labels and review uncertain rows."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from lego_model import COLOR_CLASSES, DIRECTION_CLASSES


DEFAULT_DATASET = Path.home() / ".cache/kagglehub/datasets/ihelon/lego-minifigures-classification/versions/30"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "labels.csv"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
MARGIN_THRESHOLD = 0.15

FIELDNAMES = [
    "path",
    "class_id",
    "split",
    "direction",
    "direction_confidence",
    "direction_margin",
    "head_color",
    "color_confidence",
    "color_margin",
    "review_status",
    "review_reason",
    "label_source",
]

DIRECTION_PROMPTS = {
    value: f"a photo of a Lego minifigure facing {value.replace('-', ' ')}"
    for value in DIRECTION_CLASSES
}
COLOR_PROMPTS = {
    value: (
        "a photo of a Lego minifigure whose head is another color"
        if value == "other"
        else f"a photo of a Lego minifigure with a {value} colored head"
    )
    for value in COLOR_CLASSES
}


def grouped_splits(class_ids: list[str], seed: int = 42) -> dict[str, str]:
    groups = sorted(set(class_ids), key=lambda value: int(value))
    random.Random(seed).shuffle(groups)
    count = len(groups)
    train_end = max(1, round(count * 0.70))
    val_count = max(1, round(count * 0.15))
    val_end = min(count - 1, train_end + val_count)
    split = {}
    for index, group in enumerate(groups):
        split[group] = "train" if index < train_end else "val" if index < val_end else "test"
    return split


def load_index(dataset_root: Path) -> list[dict[str, str]]:
    index_path = dataset_root / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    manifest_paths = [index_path]
    test_path = dataset_root / "test.csv"
    if test_path.is_file():
        manifest_paths.append(test_path)
    rows = []
    seen_paths = set()
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["path"] not in seen_paths:
                    rows.append(row)
                    seen_paths.add(row["path"])
    if not rows or not {"path", "class_id"}.issubset(rows[0]):
        raise ValueError("Dataset manifests must contain path and class_id columns")
    class_by_folder = {
        Path(row["path"]).parent.as_posix(): row["class_id"]
        for row in rows
        if Path(row["path"]).parent.as_posix() != "test"
    }
    for image_path in sorted(dataset_root.rglob("*.jpg")):
        relative = image_path.relative_to(dataset_root).as_posix()
        if relative in seen_paths:
            continue
        class_id = class_by_folder.get(Path(relative).parent.as_posix())
        if class_id is not None:
            rows.append({"path": relative, "class_id": class_id})
            seen_paths.add(relative)
    split_map = grouped_splits([row["class_id"] for row in rows])
    for row in rows:
        row["split"] = split_map[row["class_id"]]
    return rows


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prompt_embeddings(model, processor, prompts: dict[str, str], device: torch.device):
    values = list(prompts)
    inputs = processor(text=[prompts[value] for value in values], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embeddings = model.get_text_features(**inputs)
        if hasattr(embeddings, "pooler_output"):
            embeddings = embeddings.pooler_output
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return values, embeddings


def classify_embeddings(image_embeddings, text_embeddings, class_names: list[str]):
    probabilities = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    top_values, top_indices = probabilities.topk(k=2, dim=-1)
    results = []
    for values, indices in zip(top_values.cpu(), top_indices.cpu()):
        results.append(
            (
                class_names[int(indices[0])],
                float(values[0]),
                float(values[0] - values[1]),
            )
        )
    return results


def generate_labels(dataset_root: Path, output: Path, batch_size: int) -> list[dict[str, str]]:
    indexed = load_index(dataset_root)
    existing = load_existing(output)
    pending = [row for row in indexed if row["path"] not in existing]
    if not pending:
        return [existing[row["path"]] for row in indexed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).to(device).eval()
    direction_names, direction_text = prompt_embeddings(model, processor, DIRECTION_PROMPTS, device)
    color_names, color_text = prompt_embeddings(model, processor, COLOR_PROMPTS, device)

    all_rows = dict(existing)
    for start in range(0, len(pending), batch_size):
        batch_rows = pending[start : start + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(dataset_root / Path(row["path"])) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            image_embeddings = model.get_image_features(pixel_values=pixel_values)
            if hasattr(image_embeddings, "pooler_output"):
                image_embeddings = image_embeddings.pooler_output
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        directions = classify_embeddings(image_embeddings, direction_text, direction_names)
        colors = classify_embeddings(image_embeddings, color_text, color_names)
        for source, direction, color in zip(batch_rows, directions, colors):
            uncertain_reasons = []
            if direction[2] < MARGIN_THRESHOLD:
                uncertain_reasons.append("direction")
            if color[2] < MARGIN_THRESHOLD:
                uncertain_reasons.append("color")
            if source["split"] == "test":
                uncertain_reasons.append("test_requires_review")
            all_rows[source["path"]] = {
                "path": source["path"],
                "class_id": source["class_id"],
                "split": source["split"],
                "direction": direction[0],
                "direction_confidence": f"{direction[1]:.6f}",
                "direction_margin": f"{direction[2]:.6f}",
                "head_color": color[0],
                "color_confidence": f"{color[1]:.6f}",
                "color_margin": f"{color[2]:.6f}",
                "review_status": "needs_review" if uncertain_reasons else "auto_accepted",
                "review_reason": ";".join(uncertain_reasons),
                "label_source": "clip" if uncertain_reasons else "clip_auto_accepted",
            }
        ordered = [all_rows[row["path"]] for row in indexed if row["path"] in all_rows]
        save_rows(output, ordered)
        print(f"Labeled {min(start + batch_size, len(pending))}/{len(pending)} new images")

    return [all_rows[row["path"]] for row in indexed]


def choose_class(label: str, values: list[str]) -> str:
    print(f"Current: {label}")
    print("  ".join(f"{index + 1}:{value}" for index, value in enumerate(values)))
    while True:
        answer = input("Choose number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(values):
            return values[int(answer) - 1]
        print("Invalid choice.")


def review_required(dataset_root: Path, output: Path, rows: list[dict[str, str]]) -> None:
    pending = [row for row in rows if row["review_status"] == "needs_review"]
    if not pending:
        print("No labels require review.")
        return

    print(
        "Review options:\n"
        "  Enter or 1 = direction and color are correct\n"
        "  2 or d     = incorrect direction\n"
        "  3 or c     = incorrect color\n"
        "  4 or b     = incorrect direction and color\n"
        "  5 or s     = skip for now\n"
        "  q          = save and quit"
    )
    for index, row in enumerate(pending, start=1):
        image_path = dataset_root / Path(row["path"])
        image = cv2.imread(str(image_path))
        if image is not None:
            cv2.imshow("Lego label review", image)
            cv2.waitKey(1)
        print(
            f"\n[{index}/{len(pending)}] {row['path']} ({row['split']})\n"
            f"direction={row['direction']}  head_color={row['head_color']}  reason={row['review_reason']}"
        )
        command = input("Choose review option: ").strip().lower()
        if command == "q":
            save_rows(output, rows)
            break
        if command in {"5", "s"}:
            row["review_status"] = "skipped"
            save_rows(output, rows)
            continue
        if command in {"2", "d", "4", "b"}:
            row["direction"] = choose_class(row["direction"], DIRECTION_CLASSES)
        if command in {"3", "c", "4", "b"}:
            row["head_color"] = choose_class(row["head_color"], COLOR_CLASSES)
        if command not in {"", "1", "2", "d", "3", "c", "4", "b"}:
            print("Invalid option; this image was not changed.")
            continue
        row["review_status"] = "reviewed"
        row["review_reason"] = ""
        row["label_source"] = "human_reviewed"
        save_rows(output, rows)
    cv2.destroyAllWindows()


def accept_as_pseudo_labels(output: Path, rows: list[dict[str, str]]) -> None:
    """Keep every CLIP result without claiming that it is human ground truth."""
    changed = 0
    for row in rows:
        if row["review_status"] == "needs_review":
            row["review_status"] = "pseudo_labeled"
            row["label_source"] = "clip_unverified"
            changed += 1
    if changed:
        save_rows(output, rows)
    print(f"Accepted {changed} uncertain CLIP results as unverified pseudo-labels.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--review",
        choices=["required", "none"],
        default="none",
        help="By default, accept all CLIP output as unverified pseudo-labels.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = generate_labels(args.dataset_root, args.output, args.batch_size)
    if args.review == "required":
        review_required(args.dataset_root, args.output, rows)
    else:
        accept_as_pseudo_labels(args.output, rows)
    print(f"Labels saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
