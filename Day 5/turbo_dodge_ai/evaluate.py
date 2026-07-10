"""Evaluate or watch a saved Turbo Dodge AI SAC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

try:
    from .qualification import PACKAGE_ROOT, load_manifest, rollout_policy
    from .training import PHASE_BY_NAME, create_environment
except ImportError:  # Allows `python turbo_dodge_ai/evaluate.py` from Day 5.
    from qualification import PACKAGE_ROOT, load_manifest, rollout_policy  # type: ignore[no-redef]
    from training import PHASE_BY_NAME, create_environment  # type: ignore[no-redef]


def _resolve_model_path(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Model checkpoint was not found: {candidate}")
    manifest = load_manifest()
    if not manifest or not bool(manifest.get("qualified")) or not isinstance(manifest.get("model_path"), str):
        raise FileNotFoundError("No qualified model manifest exists; train the model first or pass --model.")
    candidate = PACKAGE_ROOT / str(manifest["model_path"])
    if not candidate.exists():
        raise FileNotFoundError(f"Manifest points to a missing model: {candidate}")
    return candidate.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Turbo Dodge SAC checkpoint.")
    parser.add_argument("--model", help="Checkpoint .zip; defaults to the qualified model.")
    parser.add_argument("--phase", choices=tuple(PHASE_BY_NAME), default="progressive")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19_000_000)
    parser.add_argument("--render", action="store_true", help="Show Pygame while evaluating.")
    parser.add_argument("--max-episode-seconds", type=float, default=120.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.episodes < 1 or args.max_episode_seconds <= 0.0:
        print("--episodes and --max-episode-seconds must be positive.", file=sys.stderr)
        return 2
    try:
        from stable_baselines3 import SAC
    except ImportError:
        print("Evaluation requires Stable-Baselines3. Install turbo_dodge_ai/requirements.txt first.", file=sys.stderr)
        return 2
    try:
        model_path = _resolve_model_path(args.model)
        model = SAC.load(str(model_path), device="auto")

        def factory(*, phase: str, seed: int, render_mode: str | None = None):
            return create_environment(
                phase=phase,
                seed=seed,
                render_mode=render_mode,
                max_episode_seconds=args.max_episode_seconds,
            )

        metrics, records = rollout_policy(
            model,
            factory,
            phase=args.phase,
            episodes=args.episodes,
            seed_start=args.seed,
            render=args.render,
        )
        print(json.dumps({"model": str(model_path), "phase": args.phase, "metrics": metrics}, indent=2, sort_keys=True))
        if args.render:
            print(f"Rendered {len(records)} evaluation episode(s).")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
