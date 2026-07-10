"""Pygame screens and input loops for the Turbo Dodge AI game.

The UI is intentionally injected with a simulation instance and optional AI
policy.  It does not import the simulation or Stable-Baselines3 itself, which
keeps headless training independent of Pygame and makes this module easy to
exercise with a tiny fake simulation in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pygame

try:  # Supports both ``python -m turbo_dodge_ai`` and direct local imports.
    from .rendering import CockpitRenderer
except ImportError:  # pragma: no cover - convenience when run as a script
    from rendering import CockpitRenderer


MENU_PLAY = "play"
MENU_TRAIN = "train"
MENU_VERSUS = "versus"
MENU_QUIT = "quit"
MENU_BACK = "menu"


@dataclass
class ModeOutcome:
    """The value returned when an interactive driving screen ends."""

    destination: str = MENU_BACK
    result: Any = None
    stats: dict[str, Any] = field(default_factory=dict)


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            candidate = getattr(value, name)
            if not callable(candidate):
                return candidate
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class TurboDodgeUI:
    """Own the Pygame display and expose menu, game, and training screens."""

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1280, 720),
        caption: str = "Turbo Dodge AI",
        fps: int = 60,
    ) -> None:
        pygame.init()
        pygame.font.init()
        self.size = size
        self.caption = caption
        self.fps = fps
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.renderer = CockpitRenderer(self.screen)
        self._fonts: dict[int, pygame.font.Font] = {}
        self.show_debug = False

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Main menu

    def main_menu(
        self,
        *,
        qualified: bool = False,
        qualification: Mapping[str, Any] | None = None,
    ) -> str:
        """Run the main menu and return ``play``, ``train``, ``versus``, or ``quit``."""

        selected = 0
        buttons = (MENU_PLAY, MENU_TRAIN, MENU_VERSUS)
        running = True
        while running:
            mouse_choice: str | None = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return MENU_QUIT
                if event.type == pygame.VIDEORESIZE:
                    self._resize(event.size)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return MENU_QUIT
                    if event.key in (pygame.K_UP, pygame.K_w):
                        selected = (selected - 1) % len(buttons)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected = (selected + 1) % len(buttons)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        choice = buttons[selected]
                        if choice != MENU_VERSUS or qualified:
                            return choice
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_choice = self._menu_choice_at(event.pos, qualified)
            hovered = self._menu_choice_at(pygame.mouse.get_pos(), qualified)
            if hovered in buttons:
                selected = buttons.index(hovered)
            if mouse_choice:
                return mouse_choice
            self.draw_main_menu(selected=selected, qualified=qualified, qualification=qualification)
            pygame.display.flip()
            self.clock.tick(self.fps)
        return MENU_QUIT

    # Alias gives a more discoverable API to a simple launcher.
    run_menu = main_menu

    def draw_main_menu(
        self,
        *,
        selected: int = 0,
        qualified: bool = False,
        qualification: Mapping[str, Any] | None = None,
    ) -> None:
        """Draw a menu frame without processing events or flipping the display."""

        width, height = self.screen.get_size()
        self._menu_background()
        self._text("TURBO", (width // 2, int(height * 0.12)), 76, (244, 246, 249), anchor="midtop")
        self._text("DODGE AI", (width // 2, int(height * 0.22)), 76, (222, 47, 50), anchor="midtop")
        self._text("Steer. Survive. Challenge the model.", (width // 2, int(height * 0.345)), 25, (209, 222, 233), anchor="midtop")

        options = (
            (MENU_PLAY, "PLAY", "Take the wheel and dodge hazards"),
            (MENU_TRAIN, "TRAIN AI", "Train or resume the driving model"),
            (MENU_VERSUS, "HUMAN VS AI", "Race the qualified model"),
        )
        button_w, button_h = min(525, int(width * 0.48)), 70
        x = (width - button_w) // 2
        first_y = int(height * 0.43)
        mouse = pygame.mouse.get_pos()
        for index, (choice, title, subtitle) in enumerate(options):
            rect = pygame.Rect(x, first_y + index * 88, button_w, button_h)
            enabled = choice != MENU_VERSUS or qualified
            hovered = rect.collidepoint(mouse)
            active = index == selected or hovered
            self._menu_button(rect, title, subtitle, active=active, enabled=enabled)

        if not qualified:
            reason = self._qualification_message(qualification)
            self._text(reason, (width // 2, first_y + 3 * 88 + 2), 18, (245, 185, 83), anchor="midtop")
        self._text("↑ ↓ / W S choose    Enter select    Esc quit", (width // 2, height - 30), 19, (200, 212, 224), anchor="midbottom")

    def _menu_background(self) -> None:
        width, height = self.screen.get_size()
        self.screen.fill((8, 14, 27))
        # A small static road illustration gives the menu identity without a
        # simulation instance or external artwork.
        horizon = int(height * 0.39)
        pygame.draw.rect(self.screen, (24, 73, 55), (0, horizon, width, height - horizon))
        pygame.draw.polygon(self.screen, (39, 43, 51), [(int(width * 0.43), horizon), (int(width * 0.57), horizon), (int(width * 0.94), height), (int(width * 0.06), height)])
        pygame.draw.line(self.screen, (244, 207, 83), (width // 2, horizon), (width // 2, height), 4)
        for x in (int(width * 0.43), int(width * 0.57)):
            pygame.draw.line(self.screen, (239, 240, 236), (x, horizon), (int(width * (0.04 if x < width / 2 else 0.96)), height), 4)
        pygame.draw.polygon(self.screen, (211, 40, 43), [(int(width * 0.43), height), (int(width * 0.47), int(height * 0.84)), (int(width * 0.53), int(height * 0.84)), (int(width * 0.57), height)])

    def _menu_button(self, rect: pygame.Rect, title: str, subtitle: str, *, active: bool, enabled: bool) -> None:
        if enabled:
            fill = (43, 53, 70) if active else (24, 31, 44)
            edge = (236, 65, 66) if active else (91, 117, 143)
            title_colour = (248, 249, 251)
            sub_colour = (190, 207, 222)
        else:
            fill, edge = (25, 28, 35), (62, 68, 77)
            title_colour, sub_colour = (126, 133, 141), (100, 107, 115)
        pygame.draw.rect(self.screen, fill, rect, border_radius=12)
        pygame.draw.rect(self.screen, edge, rect, 3 if active else 2, border_radius=12)
        self._text(title, (rect.x + 22, rect.y + 12), 30, title_colour)
        self._text(subtitle, (rect.right - 20, rect.bottom - 13), 18, sub_colour, anchor="bottomright")
        if not enabled:
            self._text("LOCKED", (rect.right - 20, rect.y + 15), 18, (245, 185, 83), anchor="topright")

    def _menu_choice_at(self, position: tuple[int, int], qualified: bool) -> str | None:
        width, height = self.screen.get_size()
        button_w, button_h = min(525, int(width * 0.48)), 70
        x = (width - button_w) // 2
        first_y = int(height * 0.43)
        choices = (MENU_PLAY, MENU_TRAIN, MENU_VERSUS)
        for index, choice in enumerate(choices):
            rect = pygame.Rect(x, first_y + index * 88, button_w, button_h)
            if rect.collidepoint(position):
                if choice == MENU_VERSUS and not qualified:
                    return None
                return choice
        return None

    # ------------------------------------------------------------------
    # Human driving modes

    def run_human_solo(self, simulation: Any, *, seed: int | None = None) -> ModeOutcome:
        """Run human-only endless play using the supplied simulation instance."""

        return self.run_drive(simulation, seed=seed, versus=False)

    def run_human_vs_ai(
        self,
        simulation: Any,
        ai_policy: Any = None,
        *,
        seed: int | None = None,
    ) -> ModeOutcome:
        """Run shared-road human-versus-AI play using a model/policy injection."""

        return self.run_drive(simulation, seed=seed, versus=True, ai_policy=ai_policy)

    # Short aliases make common launcher code pleasant to read.
    play_solo = run_human_solo
    play_versus = run_human_vs_ai

    def run_drive(
        self,
        simulation: Any,
        *,
        seed: int | None = None,
        versus: bool = False,
        ai_policy: Any = None,
    ) -> ModeOutcome:
        """Run a fixed-step interactive match until Esc/close is requested.

        ``simulation.step`` is invoked with an actions dictionary.  The
        simulation owns all collision and score rules; the UI only converts
        held keyboard inputs into human steering and asks an injected policy
        for the AI steering value.
        """

        self._reset_simulation(simulation, seed=seed, versus=versus)
        paused = False
        show_debug = self.show_debug
        accumulator = 0.0
        running = True
        outcome: Any = None
        human_id, ai_id = self._driver_ids(simulation, versus)

        while running:
            frame_seconds = min(self.clock.tick(self.fps) / 1000.0, 0.125)
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ModeOutcome(destination=MENU_QUIT, result=self._last_result(simulation), stats=self._stats(simulation))
                if event.type == pygame.VIDEORESIZE:
                    self._resize(event.size)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.show_debug = show_debug
                        return ModeOutcome(destination=MENU_BACK, result=self._last_result(simulation), stats=self._stats(simulation))
                    if event.key == pygame.K_q:
                        show_debug = not show_debug
                    elif event.key == pygame.K_p and not self._is_finished(simulation):
                        paused = not paused
                    elif event.key == pygame.K_r and self._is_finished(simulation):
                        reset_requested = True

            if reset_requested:
                self._reset_simulation(simulation, seed=seed, versus=versus)
                paused = False
                accumulator = 0.0
                outcome = None
                human_id, ai_id = self._driver_ids(simulation, versus)

            if not paused and not self._is_finished(simulation):
                accumulator += frame_seconds
                step_seconds = self._simulation_dt(simulation)
                # Cap catch-up work after a window drag so one long UI stall
                # cannot turn into an invisible multi-second crash.
                steps = 0
                while accumulator >= step_seconds and steps < 5 and not self._is_finished(simulation):
                    human_action = self._human_action()
                    ai_action = self._policy_action(ai_policy, simulation, ai_id) if versus and ai_id is not None else None
                    outcome = self._step_simulation(simulation, human_id, human_action, ai_id, ai_action)
                    accumulator -= step_seconds
                    steps += 1
                if steps == 5:
                    accumulator = min(accumulator, step_seconds)

            mode = "versus" if versus else "solo"
            message = self._result_message(simulation) if self._is_finished(simulation) else None
            self.renderer.render(
                simulation,
                focus_driver=human_id,
                mode=mode,
                paused=paused,
                show_debug=show_debug,
                message=message,
            )
            pygame.display.flip()

        return ModeOutcome(destination=MENU_BACK, result=outcome, stats=self._stats(simulation))

    # ------------------------------------------------------------------
    # Training-status helper

    def draw_training_status(
        self,
        status: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
    ) -> None:
        """Draw one training-progress screen; caller controls event timing."""

        status = status or {}
        width, height = self.screen.get_size()
        self._menu_background()
        tint = pygame.Surface((width, height), pygame.SRCALPHA)
        tint.fill((3, 9, 17, 160))
        self.screen.blit(tint, (0, 0))
        self._text("AI TRAINING", (width // 2, 34), 52, (244, 246, 249), anchor="midtop")
        phase = _field(status, "phase", "curriculum_phase", default="Preparing")
        self._text(str(phase).upper(), (width // 2, 92), 23, (245, 188, 85), anchor="midtop")

        card_w = min(950, int(width * 0.80))
        card = pygame.Rect((width - card_w) // 2, int(height * 0.20), card_w, int(height * 0.54))
        pygame.draw.rect(self.screen, (13, 21, 33), card, border_radius=16)
        pygame.draw.rect(self.screen, (81, 127, 157), card, 2, border_radius=16)

        entries = (
            ("STEPS", _field(status, "total_steps", "steps", "step", default="—")),
            ("RECENT RETURN", _field(status, "recent_reward", "mean_reward", "reward", default="—")),
            ("BEST SURVIVAL", self._format_seconds(_field(status, "best_survival", "survival_time", default=None))),
            ("EVALUATION", self._format_percent(_field(status, "evaluation_success", "success_rate", "eval_success", default=None))),
            ("CHECKPOINT", _field(status, "checkpoint", "checkpoint_path", default="Not saved yet")),
            ("QUALIFICATION", self._qualified_label(status)),
        )
        row_y = card.y + 35
        for label, value in entries:
            self._text(label, (card.x + 34, row_y), 21, (151, 187, 211))
            self._text(str(value), (card.right - 34, row_y), 25, (238, 244, 249), anchor="topright")
            pygame.draw.line(self.screen, (48, 70, 89), (card.x + 30, row_y + 32), (card.right - 30, row_y + 32), 1)
            row_y += 50

        body = message or "Training can continue headlessly. Esc returns to the menu."
        self._text(body, (width // 2, int(height * 0.80)), 21, (210, 224, 235), anchor="midtop")
        self._text("Esc menu    Q debug overlay", (width // 2, height - 30), 18, (190, 207, 222), anchor="midbottom")

    def run_training_status(
        self,
        status_provider: Callable[[], Mapping[str, Any] | None],
        *,
        message: str | None = None,
    ) -> str:
        """Keep a status window alive while a background trainer is running.

        The provider may add ``done=True`` to end this screen automatically.
        It is intentionally polling-only so callers remain responsible for
        starting, stopping, and joining the training job.
        """

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return MENU_QUIT
                if event.type == pygame.VIDEORESIZE:
                    self._resize(event.size)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return MENU_BACK
            status = status_provider() or {}
            self.draw_training_status(status, message=message)
            pygame.display.flip()
            if bool(_field(status, "done", "finished", default=False)):
                return MENU_BACK
            self.clock.tick(12)

    # ------------------------------------------------------------------
    # Input, model adaptation, and simulation adaptation

    def _human_action(self) -> float:
        pressed = pygame.key.get_pressed()
        left = bool(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
        right = bool(pressed[pygame.K_d] or pressed[pygame.K_RIGHT])
        if left == right:
            return 0.0
        return -1.0 if left else 1.0

    def _policy_action(self, policy: Any, simulation: Any, driver_id: str) -> float:
        """Adapt a callable, SB3 policy, or optional simulation helper to float."""

        observation = self._observation_for(simulation, driver_id)
        result: Any = None
        try:
            if policy is not None and hasattr(policy, "predict"):
                result = policy.predict(observation, deterministic=True)
                if isinstance(result, tuple):
                    result = result[0]
            elif callable(policy):
                try:
                    result = policy(observation)
                except TypeError:
                    result = policy(simulation, driver_id)
            elif hasattr(simulation, "ai_action"):
                helper = getattr(simulation, "ai_action")
                result = helper(driver_id) if callable(helper) else helper
            elif hasattr(simulation, "scripted_action"):
                result = simulation.scripted_action(driver_id)
        except (AttributeError, TypeError, ValueError, IndexError):
            result = None

        if result is not None:
            try:
                return float(np.clip(np.asarray(result, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
            except (TypeError, ValueError, IndexError):
                pass

        # A safe fallback lets an unlocked screen still run if model loading
        # failed. It centres the car but makes no claim to be an obstacle AI.
        driver = self._drivers(simulation).get(driver_id)
        road_width = _number(_field(simulation, "road_width", default=18.0), 18.0)
        return float(np.clip(-_number(_field(driver, "x"), 0.0) / max(1.0, road_width / 2.0), -0.45, 0.45))

    def _observation_for(self, simulation: Any, driver_id: str) -> Any:
        for name in ("observation_for", "get_observation", "observation", "observe"):
            method = getattr(simulation, name, None)
            if callable(method):
                try:
                    return method(driver_id)
                except TypeError:
                    try:
                        return method()
                    except TypeError:
                        continue
        return np.zeros(19, dtype=np.float32)

    def _reset_simulation(self, simulation: Any, *, seed: int | None, versus: bool) -> Any:
        reset = getattr(simulation, "reset")
        try:
            return reset(seed=seed, versus=versus)
        except TypeError:
            try:
                return reset(seed=seed, driver_count=2 if versus else 1)
            except TypeError:
                try:
                    return reset(seed, 2 if versus else 1)
                except TypeError:
                    return reset()

    def _step_simulation(
        self,
        simulation: Any,
        human_id: str,
        human_action: float,
        ai_id: str | None,
        ai_action: float | None,
    ) -> Any:
        """Call either the core's sequence API or a mapping-compatible fake.

        The production ``Simulation`` deliberately accepts a sequence indexed
        by driver order. A mapping remains useful for external prototypes and
        tests, so we retain that fallback instead of making UI code brittle.
        """

        raw_drivers = _field(simulation, "drivers", default=())
        if isinstance(raw_drivers, Mapping):
            actions: dict[str, float] = {human_id: human_action}
            if ai_id is not None and ai_action is not None:
                actions[ai_id] = ai_action
            return simulation.step(actions)

        keys = list(self._drivers(simulation))
        values = [0.0] * max(1, len(keys))
        try:
            values[keys.index(human_id)] = human_action
        except ValueError:
            values[0] = human_action
        if ai_id is not None and ai_action is not None:
            try:
                values[keys.index(ai_id)] = ai_action
            except ValueError:
                if len(values) > 1:
                    values[1] = ai_action
        return simulation.step(values)

    def _driver_ids(self, simulation: Any, versus: bool) -> tuple[str, str | None]:
        drivers = self._drivers(simulation)
        human = next((name for name in ("human", "player", "ego") if name in drivers), None)
        if human is None:
            human = next(iter(drivers), "human")
        if not versus:
            return human, None
        ai = next((name for name in ("ai", "opponent", "model") if name in drivers and name != human), None)
        if ai is None:
            ai = next((name for name in drivers if name != human), None)
        return human, ai

    def _drivers(self, simulation: Any) -> dict[str, Any]:
        raw = _field(simulation, "drivers", default={})
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items()}
        try:
            return {str(_field(driver, "id", "name", default=index)): driver for index, driver in enumerate(raw)}
        except TypeError:
            return {}

    def _simulation_dt(self, simulation: Any) -> float:
        return max(1 / 240, min(1 / 10, _number(_field(simulation, "dt", "time_step", default=1 / 30), 1 / 30)))

    def _is_finished(self, simulation: Any) -> bool:
        result = self._last_result(simulation)
        if result is not None:
            complete = _field(result, "finished", "done", "terminated", "complete", default=None)
            if complete is not None:
                return bool(complete)
            # Named outcomes/results normally mean the simulation has resolved.
            if _field(result, "winner", "outcome", "reason", "result", default=None) is not None:
                return True
        drivers = self._drivers(simulation)
        return bool(drivers) and not any(bool(_field(driver, "alive", default=True)) for driver in drivers.values())

    def _last_result(self, simulation: Any) -> Any:
        return _field(simulation, "last_result", "result", "match_result", default=None)

    def _result_message(self, simulation: Any) -> str:
        result = self._last_result(simulation)
        value = _field(result, "message", "outcome", "winner", "reason", "result", default=None)
        if value is not None:
            text = str(value).replace("_", " ").upper()
            if text in {"HUMAN", "PLAYER"}:
                return "YOU WIN"
            if text == "AI":
                return "AI WINS"
            return text
        drivers = self._drivers(simulation)
        human = next((driver for key, driver in drivers.items() if key in {"human", "player", "ego"}), None)
        if human is not None and not bool(_field(human, "alive", default=True)):
            return "CRASHED"
        return "MATCH COMPLETE"

    def _stats(self, simulation: Any) -> dict[str, Any]:
        return {
            "elapsed_time": _field(simulation, "elapsed_time", "time", "episode_time", default=0.0),
            "passed_groups": _field(simulation, "passed_groups", "groups_passed", "score", default=0),
            "result": self._last_result(simulation),
        }

    def _resize(self, size: tuple[int, int]) -> None:
        width, height = max(640, size[0]), max(420, size[1])
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        self.renderer.set_surface(self.screen)

    # ------------------------------------------------------------------
    # Small text/status helpers

    def _font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _text(
        self,
        text: str,
        position: tuple[int | float, int | float],
        size: int,
        colour: tuple[int, int, int],
        *,
        anchor: str = "topleft",
    ) -> None:
        image = self._font(size).render(str(text), True, colour)
        rect = image.get_rect()
        setattr(rect, anchor, (int(position[0]), int(position[1])))
        self.screen.blit(image, rect)

    @staticmethod
    def _format_seconds(value: Any) -> str:
        try:
            return f"{float(value):.1f}s"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _format_percent(value: Any) -> str:
        try:
            value = float(value)
            if value <= 1.0:
                value *= 100.0
            return f"{value:.0f}%"
        except (TypeError, ValueError):
            return "—"

    def _qualified_label(self, status: Mapping[str, Any]) -> str:
        value = _field(status, "qualified", "is_qualified", default=None)
        if value is True:
            return "READY FOR VERSUS"
        if value is False:
            return "IN PROGRESS"
        return "PENDING"

    def _qualification_message(self, qualification: Mapping[str, Any] | None) -> str:
        if not qualification:
            return "Human vs AI unlocks after a qualified model is saved."
        missing = _field(qualification, "missing", "missing_metrics", "reason", default=None)
        if isinstance(missing, (list, tuple)) and missing:
            return f"Locked: {', '.join(map(str, missing))}"
        if missing:
            return f"Locked: {missing}"
        return "Human vs AI unlocks after qualification evaluation passes."


# A descriptive alias accommodates either name in a launcher or a test fixture.
GameUI = TurboDodgeUI

__all__ = [
    "GameUI",
    "MENU_BACK",
    "MENU_PLAY",
    "MENU_QUIT",
    "MENU_TRAIN",
    "MENU_VERSUS",
    "ModeOutcome",
    "TurboDodgeUI",
]
