# Turbo Dodge AI

Turbo Dodge AI is an endless, steering-only driving game with three Pygame modes:

- **Play** — drive the Formula 1-style car yourself and survive as long as possible.
- **Train AI** — launch/resume staged SAC training with saved checkpoints and metrics.
- **Human vs AI** — share the same road with the qualified model. The first driver to crash loses; car-to-car contact is a draw.

The playable game and the reinforcement-learning environment use the same fixed-step simulation. The AI receives numerical state and LIDAR values, never rendered pixels.

## Setup (Windows)

From the repository root, create or repair a Python 3.12 virtual environment and install the dependencies:  

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r "Day 5\turbo_dodge_ai\requirements.txt"
```

## Run the game

```powershell
.venv\Scripts\python.exe "Day 5\project.py"
```

Controls in Play and Human vs AI:

- `A` / `D` or arrow keys: steer
- `P`: pause
- `Q`: toggle top-down debug overlay
- `R`: restart after a result
- `Esc`: return to the menu

Human vs AI remains locked until the model passes the held-out solo and shared-road evaluation thresholds. Its manifest is stored at `artifacts/qualified_model.json`.

## Train and evaluate from the terminal

Run these from the `Day 5` directory:

```powershell
..\.venv\Scripts\python.exe -m turbo_dodge_ai.training
..\.venv\Scripts\python.exe -m turbo_dodge_ai.evaluate --phase progressive --episodes 100
```

Training uses four sequential SAC phases: Easy (50k steps), Progressive (200k), Advanced (300k), and Multiplayer (250k). It saves checkpoints, replay buffers, CSV metrics, plots, qualification reports, and a menu-readable status file under `turbo_dodge_ai/artifacts`.

To run a short smoke training session instead of the full curriculum:

```powershell
..\.venv\Scripts\python.exe -m turbo_dodge_ai.training --steps-easy 1000 --steps-progressive 0 --steps-advanced 0 --steps-multiplayer 0 --skip-qualification
```

## Architecture

- `core.py`: deterministic vehicle physics, obstacles, collisions, LIDAR, and one/two-driver match rules.
- `environment.py`: 19-feature Gymnasium adapter and engineered SAC reward.
- `ui.py` / `rendering.py`: menu, keyboard play, third-person road view, and debug overlay.
- `training.py`, `evaluate.py`, and `qualification.py`: training, held-out evaluation, checkpoints, and versus unlocking.

Run the automated tests from `Day 5` with:

```powershell
..\.venv\Scripts\python.exe -m pytest turbo_dodge_ai\tests
```
