"""Interactive application coordinator for Turbo Dodge AI's three modes."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .qualification import ARTIFACTS_DIR, PACKAGE_ROOT, load_manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _model_path(manifest: Mapping[str, Any]) -> Path | None:
    raw = manifest.get("model_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve() if path.exists() else None


def _qualified_manifest() -> dict[str, Any] | None:
    manifest = load_manifest()
    if not manifest or not bool(manifest.get("qualified")):
        return None
    return manifest if _model_path(manifest) is not None else None


def _latest_run_directory() -> Path | None:
    runs = ARTIFACTS_DIR / "runs"
    if not runs.exists():
        return None
    # Verification runs are intentionally never resumed from the Play menu.
    options = [path for path in runs.iterdir() if path.is_dir() and path.name != "smoke-test"]
    return max(options, key=lambda path: path.stat().st_mtime) if options else None


def _persist_personal_best(simulation: Any) -> None:
    driver = simulation.drivers.get("human")
    if driver is None:
        return
    path = ARTIFACTS_DIR / "personal_best.json"
    existing = _read_json(path)
    existing["best_survival_seconds"] = max(float(existing.get("best_survival_seconds", 0.0)), simulation.elapsed_time)
    existing["best_passed_groups"] = max(int(existing.get("best_passed_groups", 0)), driver.passed_groups)
    existing["last_survival_seconds"] = simulation.elapsed_time
    existing["last_passed_groups"] = driver.passed_groups
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_sac_model(path: Path) -> Any:
    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise RuntimeError(
            "Human vs AI needs Stable-Baselines3 to load the qualified model. "
            "Install the project's requirements first."
        ) from error
    return SAC.load(str(path), device="auto")


def _training_status() -> dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "training_status.json")


def _show_message(ui: Any, message: str) -> None:
    """Use the existing training-status card as a small non-modal message screen."""

    def status() -> dict[str, Any]:
        return {"phase": "NOTICE", "qualified": False}

    ui.run_training_status(status, message=message)


def _start_training() -> subprocess.Popen[str] | None:
    """Launch one resumable training process and return it for UI polling."""

    run_dir = _latest_run_directory()
    command = [sys.executable, "-m", "turbo_dodge_ai.training"]
    if run_dir is not None:
        command += ["--run-dir", str(run_dir)]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS_DIR / "training_console.log"
    try:
        log = log_path.open("a", encoding="utf-8")
        return subprocess.Popen(
            command,
            cwd=str(PACKAGE_ROOT.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        try:
            log.close()
        except UnboundLocalError:
            pass
        return None


def run_app(project_root: Path | None = None) -> int:
    """Run the menu until the player exits.

    Import Pygame only here so a direct launcher gives a useful missing-package
    message rather than a traceback.
    """

    del project_root  # Package-relative paths are authoritative.
    try:
        from .core import Simulation
        from .ui import MENU_PLAY, MENU_QUIT, MENU_TRAIN, MENU_VERSUS, TurboDodgeUI
    except ImportError as error:
        print(
            "Turbo Dodge AI needs Pygame and NumPy for the playable modes. "
            f"Install requirements with:\n  {sys.executable} -m pip install -r \"{PACKAGE_ROOT / 'requirements.txt'}\"",
            file=sys.stderr,
        )
        return 2

    ui = TurboDodgeUI()
    trainer: subprocess.Popen[str] | None = None
    try:
        while True:
            manifest = _qualified_manifest()
            choice = ui.main_menu(qualified=manifest is not None, qualification=load_manifest())
            if choice == MENU_QUIT:
                return 0
            if choice == MENU_PLAY:
                simulation = Simulation(phase="progressive")
                outcome = ui.run_human_solo(simulation)
                _persist_personal_best(simulation)
                if outcome.destination == MENU_QUIT:
                    return 0
                continue
            if choice == MENU_TRAIN:
                if trainer is None or trainer.poll() is not None:
                    trainer = _start_training()
                if trainer is None:
                    _show_message(ui, "Could not start training. Check the Python installation and training_console.log.")
                    continue

                def status_provider() -> dict[str, Any]:
                    status = _training_status()
                    status["done"] = trainer is not None and trainer.poll() is not None
                    if status["done"] and trainer.returncode not in (0, None):
                        status["phase"] = "TRAINING STOPPED"
                    return status

                destination = ui.run_training_status(
                    status_provider,
                    message="Training runs in the background. Esc returns to the menu; progress is saved automatically.",
                )
                if destination == MENU_QUIT:
                    return 0
                continue
            if choice == MENU_VERSUS:
                # The menu only enables this when a valid qualified model is present.
                assert manifest is not None
                path = _model_path(manifest)
                if path is None:
                    _show_message(ui, "The qualified model checkpoint is missing. Train again to unlock versus mode.")
                    continue
                try:
                    policy = _load_sac_model(path)
                except Exception as error:
                    _show_message(ui, str(error))
                    continue
                outcome = ui.run_human_vs_ai(Simulation(phase="multiplayer"), policy)
                if outcome.destination == MENU_QUIT:
                    return 0
                continue
    finally:
        ui.close()


__all__ = ["run_app"]
