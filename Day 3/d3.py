"""Predict a Lego minifigure's direction and head color from one image."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from safetensors.torch import load_file
from torchvision import transforms

from lego_model import LegoMultiTaskModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "artifacts" / "lego_mobilenet.safetensors"
DEFAULT_CONFIG = SCRIPT_DIR / "artifacts" / "model_config.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def readable_image(path_text: str) -> tuple[Path, Image.Image]:
    """Validate and open an image, returning a normalized RGB copy."""
    path = Path(path_text.strip().strip('"')).expanduser()
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            return path, image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"File is not a readable image: {path}") from exc


def load_predictor(model_path: Path, config_path: Path, device: torch.device):
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Trained model artifacts were not found. Run train_model.py after "
            "creating and reviewing labels."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = LegoMultiTaskModel(
        num_directions=len(config["direction_classes"]),
        num_colors=len(config["color_classes"]),
        pretrained=False,
    )
    state = load_file(str(model_path), device=str(device))
    model.load_state_dict(state)
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    return model, transform, config


def predict(image: Image.Image, model, transform, config, device: torch.device) -> dict:
    batch = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        direction_logits, color_logits = model(batch)
        direction_probs = direction_logits.softmax(dim=1)[0]
        color_probs = color_logits.softmax(dim=1)[0]

    direction_index = int(direction_probs.argmax().item())
    color_index = int(color_probs.argmax().item())
    direction_confidence = float(direction_probs[direction_index].item())
    color_confidence = float(color_probs[color_index].item())
    threshold = float(config.get("uncertain_threshold", 0.60))

    return {
        "direction": config["direction_classes"][direction_index],
        "direction_confidence": direction_confidence,
        "direction_uncertain": direction_confidence < threshold,
        "head_color": config["color_classes"][color_index],
        "color_confidence": color_confidence,
        "color_uncertain": color_confidence < threshold,
    }


def annotate_and_save(image: Image.Image, prediction: dict, source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    direction_flag = " (uncertain)" if prediction["direction_uncertain"] else ""
    color_flag = " (uncertain)" if prediction["color_uncertain"] else ""
    lines = [
        f"Direction: {prediction['direction']} ({prediction['direction_confidence']:.1%}){direction_flag}",
        f"Head color: {prediction['head_color']} ({prediction['color_confidence']:.1%}){color_flag}",
    ]
    padding = 8
    line_height = 22
    panel_height = padding * 2 + line_height * len(lines)
    draw.rectangle((0, 0, annotated.width, panel_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((padding, padding + index * line_height), line, fill=(255, 255, 255))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{source.stem}_prediction_{timestamp}.jpg"
    annotated.save(output_path, quality=95)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Path to one image. Prompts when omitted.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_text = args.image or input("Enter the path to a Lego image: ")
    try:
        source, image = readable_image(image_text)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform, config = load_predictor(args.model, args.config, device)
        result = predict(image, model, transform, config, device)
        output_path = annotate_and_save(image, result, source, args.output_dir)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Direction: {result['direction']} ({result['direction_confidence']:.1%})")
    print(f"Head color: {result['head_color']} ({result['color_confidence']:.1%})")
    if result["direction_uncertain"] or result["color_uncertain"]:
        print("Warning: at least one prediction is uncertain.")
    print(f"Annotated image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
