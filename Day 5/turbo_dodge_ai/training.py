"""Train Turbo Dodge AI with a staged Stable-Baselines3 SAC curriculum.

The module is safe to import from the Pygame menu: heavyweight ML packages are
loaded only after the user explicitly starts training.  Run it as either
``python -m turbo_dodge_ai.training`` from the ``Day 5`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Supports both package execution and direct script execution.
    from .qualification import (
        ARTIFACTS_DIR,
        PACKAGE_ROOT,
        assess_qualification,
        format_qualification_status,
        publish_manifest,
        relative_to_package,
        rollout_policy,
        utc_now,
    )
except ImportError:  # pragma: no cover - direct ``python training.py`` route
    from qualification import (  # type: ignore[no-redef]
        ARTIFACTS_DIR,
        PACKAGE_ROOT,
        assess_qualification,
        format_qualification_status,
        publish_manifest,
        relative_to_package,
        rollout_policy,
        utc_now,
    )


class DependencyUnavailable(RuntimeError):
    """Raised when the optional AI-training dependencies are not installed."""


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    default_steps: int
    driver_count: int
    description: str


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec("easy", 50_000, 1, "single hazards at low speed"),
    PhaseSpec("progressive", 200_000, 1, "mixed single and paired hazards"),
    PhaseSpec("advanced", 300_000, 1, "fast, dense hazard groups"),
    PhaseSpec("multiplayer", 250_000, 2, "opponent-aware shared-road racing"),
)
PHASE_BY_NAME = {spec.name: spec for spec in PHASE_SPECS}

EPISODE_COLUMNS = (
    "timestamp",
    "phase",
    "total_timesteps",
    "phase_timesteps",
    "episode_return",
    "episode_steps",
    "survival_seconds",
    "passed_groups",
    "collision_reason",
    "match_result",
)


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file so the UI never reads a half-written one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_environment_class() -> type[Any]:
    """Load the core environment only when training/evaluation is requested."""

    try:
        from .environment import TurboDodgeEnv
    except ImportError as package_error:
        try:
            from environment import TurboDodgeEnv  # type: ignore[import-not-found]
        except ImportError as script_error:
            raise DependencyUnavailable(
                "TurboDodgeEnv could not be imported. Run this from the "
                "'Day 5' directory with the complete turbo_dodge_ai package present."
            ) from script_error
        # Preserve useful package import errors in normal module execution.
        if __package__:
            _ = package_error
    return TurboDodgeEnv


def create_environment(
    *,
    phase: str,
    seed: int | None = None,
    render_mode: str | None = None,
    max_episode_seconds: float = 120.0,
) -> Any:
    """Create an unwrapped environment suitable for evaluation or rendering."""

    if phase not in PHASE_BY_NAME:
        raise ValueError(f"Unknown phase {phase!r}; choose from {', '.join(PHASE_BY_NAME)}")
    environment_class = _load_environment_class()
    return environment_class(
        phase=phase,
        render_mode=render_mode,
        driver_count=PHASE_BY_NAME[phase].driver_count,
        max_episode_seconds=max_episode_seconds,
        seed=seed,
    )


def _require_training_dependencies() -> dict[str, Any]:
    """Import ML dependencies late and give a useful install command if absent."""

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        requirements = PACKAGE_ROOT / "requirements.txt"
        raise DependencyUnavailable(
            "AI training needs Stable-Baselines3, PyTorch, Gymnasium, and NumPy. "
            f"Install them with:\n  {sys.executable or 'python'} -m pip install -r \"{requirements}\"\n"
            "The Play mode remains available without these optional training dependencies."
        ) from error
    return {"SAC": SAC, "BaseCallback": BaseCallback, "Monitor": Monitor}


def _next_multiple_after(value: int, interval: int) -> int:
    return ((value // interval) + 1) * interval


class ArtifactManager:
    """Own run artifacts and the lightweight status files consumed by the menu."""

    def __init__(self, run_dir: Path, *, seed: int) -> None:
        self.run_dir = run_dir.resolve()
        self.seed = seed
        self.checkpoints = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.evaluations = self.run_dir / "evaluations"
        self.monitor_dir = self.run_dir / "monitor"
        self.state_path = self.run_dir / "training_state.json"
        self.status: dict[str, Any] = {
            "state": "initializing",
            "run_dir": relative_to_package(self.run_dir),
            "seed": seed,
            "phase": None,
            "phase_timesteps": 0,
            "total_timesteps": 0,
            "completed_phases": [],
            "last_checkpoint": None,
            "last_evaluation": None,
            "updated_at": utc_now(),
        }
        for directory in (self.checkpoints, self.metrics_dir, self.evaluations, self.monitor_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing_or_new(cls, run_dir: Path, *, seed: int) -> "ArtifactManager":
        manager = cls(run_dir, seed=seed)
        existing = manager.load_state()
        if existing:
            manager.status.update(existing)
            manager.status["state"] = "resuming"
            manager.status["updated_at"] = utc_now()
        return manager

    def load_state(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def write_status(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = utc_now()
        _write_json(self.state_path, self.status)
        # The stable path is intentionally separate from the run directory so
        # the Pygame menu can show status without scanning every run.
        _write_json(ARTIFACTS_DIR / "training_status.json", self.status)

    def record_episode(self, *, phase: str, phase_steps: int, total_steps: int, info: Mapping[str, Any]) -> None:
        episode = info.get("episode", {})
        if not isinstance(episode, Mapping):
            episode = {}
        row = {
            "timestamp": utc_now(),
            "phase": phase,
            "total_timesteps": total_steps,
            "phase_timesteps": phase_steps,
            "episode_return": _safe_float(episode.get("r")),
            "episode_steps": int(_safe_float(episode.get("l"))),
            "survival_seconds": _safe_float(info.get("elapsed_time")),
            "passed_groups": int(_safe_float(info.get("passed_groups"))),
            "collision_reason": str(info.get("collision_reason", "")),
            "match_result": str(info.get("match_result", info.get("outcome", ""))),
        }
        csv_path = self.metrics_dir / "training_metrics.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, model: Any, *, phase: str, phase_steps: int, label: str = "checkpoint") -> Path:
        total_steps = int(getattr(model, "num_timesteps", 0))
        phase_directory = self.checkpoints / phase
        phase_directory.mkdir(parents=True, exist_ok=True)
        base = phase_directory / f"{label}_{total_steps:09d}"
        model.save(str(base))
        model_path = base.with_suffix(".zip")
        replay_path = phase_directory / f"{base.name}_replay_buffer.pkl"
        save_replay_buffer = getattr(model, "save_replay_buffer", None)
        if callable(save_replay_buffer):
            save_replay_buffer(str(replay_path))
        self.write_status(
            state="training",
            phase=phase,
            phase_timesteps=phase_steps,
            total_timesteps=total_steps,
            last_checkpoint=relative_to_package(model_path),
            last_replay_buffer=relative_to_package(replay_path) if replay_path.exists() else None,
        )
        print(f"Saved {phase} checkpoint at {total_steps:,} steps: {model_path.name}", flush=True)
        return model_path

    def record_evaluation(
        self,
        *,
        phase: str,
        total_steps: int,
        metrics: Mapping[str, Any],
        records: list[Mapping[str, Any]],
        seed_start: int,
    ) -> Path:
        payload = {
            "timestamp": utc_now(),
            "phase": phase,
            "total_timesteps": total_steps,
            "seed_start": seed_start,
            "metrics": dict(metrics),
            "records": [dict(record) for record in records],
        }
        filename = f"{phase}_{total_steps:09d}.json"
        result_path = self.evaluations / filename
        _write_json(result_path, payload)
        _write_json(self.evaluations / f"latest_{phase}.json", payload)
        _append_jsonl(self.evaluations / "history.jsonl", {key: value for key, value in payload.items() if key != "records"})
        self.write_status(last_evaluation=relative_to_package(result_path), total_timesteps=total_steps)
        print(
            f"Evaluation ({phase}, {len(records)} episodes): median survival "
            f"{_safe_float(metrics.get('median_survival_seconds')):.1f}s, "
            f"30s survival {_safe_float(metrics.get('survival_rate_30_seconds')):.0%}",
            flush=True,
        )
        return result_path


def _default_run_dir(run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / "runs" / (run_name or f"sac-{timestamp}")


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob("**/*.zip"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _guess_replay_buffer(model_path: Path) -> Path | None:
    expected = model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")
    return expected if expected.exists() else None


def _evaluation_factory(max_episode_seconds: float):
    def factory(*, phase: str, seed: int, render_mode: str | None = None) -> Any:
        return create_environment(
            phase=phase,
            seed=seed,
            render_mode=render_mode,
            max_episode_seconds=max_episode_seconds,
        )

    return factory


def _make_callback(
    base_callback: type[Any],
    *,
    artifacts: ArtifactManager,
    phase: str,
    phase_start_steps: int,
    checkpoint_every: int,
    evaluation_every: int,
    evaluation_episodes: int,
    evaluation_seed: int,
    max_episode_seconds: float,
) -> Any:
    """Create an SB3 callback after SB3 has been imported lazily."""

    class TrainingCallback(base_callback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_checkpoint = 0
            self.next_evaluation = 0

        def _on_training_start(self) -> None:
            current = int(self.model.num_timesteps)
            self.next_checkpoint = _next_multiple_after(current, checkpoint_every)
            self.next_evaluation = _next_multiple_after(current, evaluation_every)

        def _on_step(self) -> bool:
            current = int(self.num_timesteps)
            phase_steps = current - phase_start_steps
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if done and isinstance(info, Mapping):
                    artifacts.record_episode(
                        phase=phase,
                        phase_steps=phase_steps,
                        total_steps=current,
                        info=info,
                    )

            if current >= self.next_checkpoint:
                artifacts.save_checkpoint(self.model, phase=phase, phase_steps=phase_steps)
                self.next_checkpoint = _next_multiple_after(current, checkpoint_every)

            if current >= self.next_evaluation:
                try:
                    metrics, records = rollout_policy(
                        self.model,
                        _evaluation_factory(max_episode_seconds),
                        phase=phase,
                        episodes=evaluation_episodes,
                        seed_start=evaluation_seed,
                    )
                    artifacts.record_evaluation(
                        phase=phase,
                        total_steps=current,
                        metrics=metrics,
                        records=records,
                        seed_start=evaluation_seed,
                    )
                except Exception as error:  # Training should not lose all progress over a report failure.
                    print(f"Warning: periodic evaluation failed: {error}", file=sys.stderr, flush=True)
                self.next_evaluation = _next_multiple_after(current, evaluation_every)

            if current % max(1_000, checkpoint_every // 10) == 0:
                artifacts.write_status(
                    state="training",
                    phase=phase,
                    phase_timesteps=phase_steps,
                    total_timesteps=current,
                )
            return True

    return TrainingCallback()


def _wrap_training_environment(monitor_class: type[Any], *, phase: str, seed: int, run_dir: Path, max_episode_seconds: float) -> Any:
    raw = create_environment(
        phase=phase,
        seed=seed,
        render_mode=None,
        max_episode_seconds=max_episode_seconds,
    )
    return monitor_class(raw, filename=str(run_dir / "monitor" / phase))


def _phase_steps(args: argparse.Namespace, phase: str) -> int:
    value = getattr(args, f"steps_{phase}")
    if value < 0:
        raise ValueError(f"--steps-{phase} must not be negative")
    return int(value)


def _read_completed_phases(artifacts: ArtifactManager) -> set[str]:
    return {str(phase) for phase in artifacts.status.get("completed_phases", [])}


def _load_model(
    sac_class: type[Any],
    *,
    model_path: Path | None,
    environment: Any,
    device: str,
    seed: int,
) -> Any:
    if model_path is not None:
        print(f"Resuming SAC model from {model_path}", flush=True)
        return sac_class.load(str(model_path), env=environment, device=device)
    return sac_class(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=10_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        device=device,
        seed=seed,
        verbose=1,
    )


def _load_replay_buffer(model: Any, replay_path: Path | None) -> None:
    if replay_path is None or not replay_path.exists():
        return
    load_replay_buffer = getattr(model, "load_replay_buffer", None)
    if callable(load_replay_buffer):
        load_replay_buffer(str(replay_path))
        print(f"Restored replay buffer from {replay_path.name}", flush=True)


def _make_plots(run_dir: Path) -> None:
    """Create useful plots when matplotlib is installed; never fail training over it."""

    csv_path = run_dir / "metrics" / "training_metrics.csv"
    if not csv_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return
        steps = [_safe_float(row.get("total_timesteps")) for row in rows]
        returns = [_safe_float(row.get("episode_return")) for row in rows]
        survival = [_safe_float(row.get("survival_seconds")) for row in rows]
        passed = [_safe_float(row.get("passed_groups")) for row in rows]
        figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(steps, returns, alpha=0.75)
        axes[0].set_ylabel("Episode return")
        axes[1].plot(steps, survival, alpha=0.75)
        axes[1].set_ylabel("Survival (s)")
        axes[2].plot(steps, passed, alpha=0.75)
        axes[2].set_ylabel("Passed groups")
        axes[2].set_xlabel("Training steps")
        figure.tight_layout()
        figure.savefig(run_dir / "metrics" / "training_progress.png", dpi=150)
        plt.close(figure)

        # Evaluation uses fixed seeds, so these curves remain comparable across
        # checkpoints and show the two metrics that gate versus unlocking.
        history_path = run_dir / "evaluations" / "history.jsonl"
        if history_path.exists():
            evaluations: list[dict[str, Any]] = []
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("metrics"), Mapping):
                    evaluations.append(payload)
            if evaluations:
                evaluation_steps = [_safe_float(item.get("total_timesteps")) for item in evaluations]
                success = [_safe_float(dict(item["metrics"]).get("survival_rate_30_seconds")) for item in evaluations]
                steering = [_safe_float(dict(item["metrics"]).get("mean_abs_steering_change")) for item in evaluations]
                figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                axes[0].plot(evaluation_steps, success, marker="o")
                axes[0].axhline(0.70, color="tab:green", linestyle="--", label="qualification target")
                axes[0].set_ylabel("30s survival rate")
                axes[0].set_ylim(0.0, 1.0)
                axes[0].legend(loc="best")
                axes[1].plot(evaluation_steps, steering, marker="o", color="tab:orange")
                axes[1].axhline(0.15, color="tab:green", linestyle="--", label="smoothness target")
                axes[1].set_ylabel("Mean |Δ steering|")
                axes[1].set_xlabel("Training steps")
                axes[1].legend(loc="best")
                figure.tight_layout()
                figure.savefig(run_dir / "metrics" / "evaluation_progress.png", dpi=150)
                plt.close(figure)
    except Exception as error:
        print(f"Warning: could not create metrics plot: {error}", file=sys.stderr)


def _run_qualification(
    *,
    model: Any,
    artifacts: ArtifactManager,
    model_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run held-out benchmarks and publish the menu-facing manifest."""

    factory = _evaluation_factory(args.max_episode_seconds)
    benchmark_seed = args.evaluation_seed + 10_000_000
    solo_metrics, solo_records = rollout_policy(
        model,
        factory,
        phase="progressive",
        episodes=args.qualification_episodes,
        seed_start=benchmark_seed,
    )
    multiplayer_metrics, multiplayer_records = rollout_policy(
        model,
        factory,
        phase="multiplayer",
        episodes=args.qualification_episodes,
        seed_start=benchmark_seed + 100_000,
    )
    assessment = assess_qualification(solo_metrics, multiplayer_metrics, require_multiplayer=True)
    payload = {
        "timestamp": utc_now(),
        "model_path": relative_to_package(model_path),
        "solo": {"metrics": solo_metrics, "records": solo_records},
        "multiplayer": {"metrics": multiplayer_metrics, "records": multiplayer_records},
        "qualification": assessment,
    }
    _write_json(artifacts.evaluations / "qualification_benchmark.json", payload)
    manifest = publish_manifest(
        model_path=model_path,
        phase="multiplayer",
        solo_metrics=solo_metrics,
        multiplayer_metrics=multiplayer_metrics,
        qualification=assessment,
        require_multiplayer=True,
    )
    print(format_qualification_status(assessment), flush=True)
    return manifest


def train(args: argparse.Namespace) -> int:
    """Execute the requested curriculum.  Returns a CLI-compatible exit code."""

    dependencies = _require_training_dependencies()
    selected_phases = list(args.phases)
    if args.no_multiplayer:
        selected_phases = [phase for phase in selected_phases if phase != "multiplayer"]
    if not selected_phases:
        raise ValueError("At least one training phase must be selected")

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else _default_run_dir(args.run_name)
    run_dir = run_dir.resolve()
    artifacts = ArtifactManager.from_existing_or_new(run_dir, seed=args.seed)
    artifacts.write_status(state="starting", selected_phases=selected_phases)

    resume_path = Path(args.resume).expanduser().resolve() if args.resume else _find_latest_checkpoint(run_dir)
    initial_phase = selected_phases[0]
    environment = _wrap_training_environment(
        dependencies["Monitor"],
        phase=initial_phase,
        seed=args.seed,
        run_dir=run_dir,
        max_episode_seconds=args.max_episode_seconds,
    )
    model = _load_model(
        dependencies["SAC"],
        model_path=resume_path,
        environment=environment,
        device=args.device,
        seed=args.seed,
    )
    replay_path: Path | None = Path(args.replay_buffer).expanduser().resolve() if args.replay_buffer else None
    if replay_path is None and resume_path is not None:
        replay_path = _guess_replay_buffer(resume_path)
    _load_replay_buffer(model, replay_path)

    completed = _read_completed_phases(artifacts)
    latest_model_path: Path | None = resume_path
    active_phase = initial_phase
    phase_progress_origin = int(getattr(model, "num_timesteps", 0))
    try:
        for phase_index, phase in enumerate(selected_phases):
            active_phase = phase
            planned_steps = _phase_steps(args, phase)
            if phase in completed:
                print(f"Skipping completed phase: {phase}", flush=True)
                continue
            if phase_index > 0:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
                environment = _wrap_training_environment(
                    dependencies["Monitor"],
                    phase=phase,
                    seed=args.seed + phase_index,
                    run_dir=run_dir,
                    max_episode_seconds=args.max_episode_seconds,
                )
                model.set_env(environment)

            phase_start_steps = int(getattr(model, "num_timesteps", 0))
            previous_phase = artifacts.status.get("phase")
            resumed_phase_steps = int(_safe_float(artifacts.status.get("phase_timesteps"))) if previous_phase == phase else 0
            phase_progress_origin = phase_start_steps - resumed_phase_steps
            remaining_steps = max(0, planned_steps - resumed_phase_steps)
            artifacts.write_status(
                state="training",
                phase=phase,
                phase_timesteps=resumed_phase_steps,
                total_timesteps=phase_start_steps,
                selected_phases=selected_phases,
            )
            print(
                f"\n=== {phase.title()} phase: {remaining_steps:,} remaining steps "
                f"({PHASE_BY_NAME[phase].description}) ===",
                flush=True,
            )

            if remaining_steps:
                callback = _make_callback(
                    dependencies["BaseCallback"],
                    artifacts=artifacts,
                    phase=phase,
                    phase_start_steps=phase_progress_origin,
                    checkpoint_every=args.checkpoint_every,
                    evaluation_every=args.evaluation_every,
                    evaluation_episodes=args.evaluation_episodes,
                    evaluation_seed=args.evaluation_seed + phase_index * 1_000_000,
                    max_episode_seconds=args.max_episode_seconds,
                )
                model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=False)

            current_phase_steps = planned_steps
            latest_model_path = artifacts.save_checkpoint(
                model,
                phase=phase,
                phase_steps=current_phase_steps,
                label="phase_complete",
            )
            # Always leave a fresh phase report, even when the interval does not align exactly.
            metrics, records = rollout_policy(
                model,
                _evaluation_factory(args.max_episode_seconds),
                phase=phase,
                episodes=args.evaluation_episodes,
                seed_start=args.evaluation_seed + phase_index * 1_000_000,
            )
            artifacts.record_evaluation(
                phase=phase,
                total_steps=int(getattr(model, "num_timesteps", 0)),
                metrics=metrics,
                records=records,
                seed_start=args.evaluation_seed + phase_index * 1_000_000,
            )
            completed.add(phase)
            artifacts.write_status(
                state="training",
                phase=phase,
                phase_timesteps=planned_steps,
                total_timesteps=int(getattr(model, "num_timesteps", 0)),
                completed_phases=[spec.name for spec in PHASE_SPECS if spec.name in completed],
            )

        if latest_model_path is None:
            latest_model_path = artifacts.save_checkpoint(
                model,
                phase=selected_phases[-1],
                phase_steps=0,
                label="final",
            )
        else:
            # A stable final filename makes CLI and handoff instructions simple.
            final_base = artifacts.checkpoints / "final_model"
            model.save(str(final_base))
            latest_model_path = final_base.with_suffix(".zip")
            save_replay_buffer = getattr(model, "save_replay_buffer", None)
            if callable(save_replay_buffer):
                save_replay_buffer(str(artifacts.checkpoints / "final_model_replay_buffer.pkl"))

        manifest: dict[str, Any] | None = None
        if not args.skip_qualification:
            manifest = _run_qualification(
                model=model,
                artifacts=artifacts,
                model_path=latest_model_path,
                args=args,
            )
        _make_plots(run_dir)
        artifacts.write_status(
            state="complete",
            phase=selected_phases[-1],
            phase_timesteps=_phase_steps(args, selected_phases[-1]),
            total_timesteps=int(getattr(model, "num_timesteps", 0)),
            final_model=relative_to_package(latest_model_path),
            qualified=bool(manifest and manifest.get("qualified")),
        )
        print(f"Training complete. Final model: {latest_model_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        exact_phase_steps = max(0, int(getattr(model, "num_timesteps", 0)) - phase_progress_origin)
        emergency_path = artifacts.save_checkpoint(
            model,
            phase=active_phase,
            phase_steps=exact_phase_steps,
            label="interrupted",
        )
        artifacts.write_status(
            state="interrupted",
            phase=active_phase,
            phase_timesteps=exact_phase_steps,
            last_checkpoint=relative_to_package(emergency_path),
        )
        print("Training interrupted; a resumable checkpoint was saved.", file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        artifacts.write_status(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Turbo Dodge SAC driver.")
    parser.add_argument("--run-dir", help="Existing/new directory for checkpoints and metrics.")
    parser.add_argument("--run-name", help="Name under artifacts/runs/ when --run-dir is omitted.")
    parser.add_argument("--resume", help="Model .zip checkpoint to resume (defaults to latest in --run-dir).")
    parser.add_argument("--replay-buffer", help="Optional replay-buffer .pkl paired with --resume.")
    parser.add_argument("--phases", nargs="+", choices=tuple(PHASE_BY_NAME), default=list(PHASE_BY_NAME))
    parser.add_argument("--no-multiplayer", action="store_true", help="Do not run the multiplayer curriculum phase.")
    parser.add_argument("--steps-easy", type=int, default=PHASE_BY_NAME["easy"].default_steps)
    parser.add_argument("--steps-progressive", type=int, default=PHASE_BY_NAME["progressive"].default_steps)
    parser.add_argument("--steps-advanced", type=int, default=PHASE_BY_NAME["advanced"].default_steps)
    parser.add_argument("--steps-multiplayer", type=int, default=PHASE_BY_NAME["multiplayer"].default_steps)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto", help="SAC device, for example auto, cpu, or cuda.")
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--evaluation-every", type=int, default=25_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--qualification-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=9_000_000)
    parser.add_argument("--max-episode-seconds", type=float, default=120.0)
    parser.add_argument("--skip-qualification", action="store_true", help="Skip final held-out unlock benchmark.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.checkpoint_every < 1 or args.evaluation_every < 1:
        parser.error("--checkpoint-every and --evaluation-every must be positive")
    if args.evaluation_episodes < 1 or args.qualification_episodes < 1:
        parser.error("evaluation episode counts must be positive")
    if args.max_episode_seconds <= 0:
        parser.error("--max-episode-seconds must be positive")
    try:
        return train(args)
    except DependencyUnavailable as error:
        print(error, file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # Show a concise failure and leave full context for CLI users.
        print(f"Training failed: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
