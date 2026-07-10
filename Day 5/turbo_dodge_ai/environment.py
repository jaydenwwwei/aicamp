"""Gymnasium adapter for :mod:`turbo_dodge_ai.core`.

The adapter deliberately makes only the learning driver an RL action source.
In multiplayer curriculum episodes the opponent is an internal, deterministic
benchmark policy, which keeps the environment compatible with SB3 SAC.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

try:
    from .core import GameConfig, Simulation
except ImportError:  # pragma: no cover - direct script convenience
    from core import GameConfig, Simulation  # type: ignore[no-redef]

try:  # Import lazily enough for the human game to explain missing packages.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised on machines without ML deps
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


if gym is not None:
    _EnvironmentBase = gym.Env[np.ndarray, np.ndarray]
else:  # pragma: no cover - gives a friendly message instead of an import crash
    _EnvironmentBase = object


class TurboDodgeEnv(_EnvironmentBase):
    """A 19-feature, continuous-action environment for SAC.

    Args:
        phase: ``easy``, ``progressive``, ``advanced``, or ``multiplayer``.
        driver_count: one for solo training and two for internal-opponent races.
        max_episode_seconds: training truncation time; human Play has no such
            truncation because it uses :class:`~turbo_dodge_ai.core.Simulation`
            directly.
        opponent_policy: optional callback receiving ``(simulation, driver_id)``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        phase: str = "progressive",
        render_mode: str | None = None,
        driver_count: int | None = None,
        max_episode_seconds: float = 120.0,
        seed: int | None = None,
        opponent_policy: Callable[[Simulation, str], float] | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise RuntimeError(
                "TurboDodgeEnv requires Gymnasium. Install dependencies with "
                "`python -m pip install -r turbo_dodge_ai/requirements.txt`."
            )
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if max_episode_seconds <= 0.0:
            raise ValueError("max_episode_seconds must be positive")
        if driver_count is None:
            driver_count = 2 if phase == "multiplayer" else 1
        if driver_count not in {1, 2}:
            raise ValueError("driver_count must be one or two")

        self.phase = phase
        self.render_mode = render_mode
        self.driver_count = driver_count
        self.max_episode_seconds = float(max_episode_seconds)
        self.opponent_policy = opponent_policy
        self._initial_seed = seed
        self.simulation = self._new_simulation(seed)
        self.dt = self.simulation.dt
        self._renderer: Any | None = None
        self._last_lead = 0.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        lows = np.asarray([-1.0] * 5 + [0.0] * 7 + [0.0] * 2 + [0.0] + [-1.0] * 4, dtype=np.float32)
        highs = np.asarray([1.0] * 5 + [1.0] * 7 + [1.0] * 2 + [1.0] + [1.0] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

    def _new_simulation(self, seed: int | None) -> Simulation:
        # Match timeout belongs to the wrapper rather than hidden state in the
        # simulation, so evaluation can choose a different safe episode length.
        # The Gymnasium timeout is a truncation. Human-versus-AI uses the
        # simulation's separate 120-second draw rule through the UI.
        config = replace(GameConfig(), match_limit_seconds=float("inf"))
        return Simulation(phase=self.phase, seed=seed, config=config)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Explicit seeds reproduce episodes. Unseeded resets draw fresh,
        # deterministic-per-environment seeds so SAC sees varied roads.
        if seed is not None:
            self._initial_seed = seed
            episode_seed = int(seed)
        else:
            episode_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self.simulation = self._new_simulation(episode_seed)
        self.simulation.reset(
            seed=episode_seed,
            versus=self.driver_count == 2,
            driver_ids=("agent", "opponent") if self.driver_count == 2 else ("agent",),
        )
        self._last_lead = 0.0
        observation = self.simulation.observation("agent")
        return observation, self._info(reward_components={})

    def set_phase(self, phase: str, *, driver_count: int | None = None) -> None:
        """Select a curriculum configuration for the next reset."""

        # Let Simulation provide the canonical phase validation.
        Simulation(phase=phase)
        self.phase = phase
        if driver_count is not None:
            if driver_count not in {1, 2}:
                raise ValueError("driver_count must be one or two")
            self.driver_count = driver_count
        elif phase == "multiplayer":
            self.driver_count = 2

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        agent_action = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        actions: dict[str, float] = {"agent": agent_action}
        if self.driver_count == 2:
            if self.opponent_policy is not None:
                opponent_action = float(self.opponent_policy(self.simulation, "opponent"))
            else:
                opponent_action = self.simulation.heuristic_action("opponent")
            actions["opponent"] = float(np.clip(opponent_action, -1.0, 1.0))

        result = self.simulation.step(actions)
        agent = self.simulation.drivers["agent"]
        terminated = not agent.alive or result.match_result is not None
        truncated = result.elapsed_time >= self.max_episode_seconds and not terminated
        reward, components = self._reward(agent_action)
        observation = self.simulation.observation("agent")
        info = self._info(reward_components=components)
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _reward(self, action: float) -> tuple[float, dict[str, float]]:
        agent = self.simulation.drivers["agent"]
        min_lidar = float(np.min(self.simulation.lidar_distances("agent")))
        center_offset = abs(agent.x) / self.simulation.config.usable_half_width
        components = {
            "survival": 0.01 if agent.alive else 0.0,
            "speed": 0.01 * agent.speed / self.simulation.settings.max_speed if agent.alive else 0.0,
            "center": -0.002 * min(1.0, center_offset),
            "proximity": -0.002 * (1.0 - min_lidar / 20.0) if min_lidar < 20.0 else 0.0,
            "evasion": 0.0,
            "crash": -10.0 if not agent.alive else 0.0,
            "lead": 0.0,
            "overtake": 0.0,
            "win": 0.0,
            "contact": 0.0,
        }
        for event in self.simulation.last_result.events:
            if event.get("type") == "group_passed" and event.get("driver") == "agent":
                components["evasion"] += 1.0
            if event.get("type") == "overtake" and event.get("driver") == "agent":
                components["overtake"] += 0.5
            if event.get("type") == "car_contact":
                components["contact"] -= 5.0
        if self.driver_count == 2:
            opponent = self.simulation.drivers["opponent"]
            lead = agent.y - opponent.y
            components["lead"] = 0.001 * float(np.clip(lead / 10.0, -1.0, 1.0))
            self._last_lead = lead
            if self.simulation.last_result.match_result == "agent":
                components["win"] = 5.0
        return sum(components.values()), components

    def _info(self, *, reward_components: dict[str, float]) -> dict[str, Any]:
        agent = self.simulation.drivers.get("agent")
        raw_result = self.simulation.last_result.match_result
        timed_out = self.simulation.elapsed_time >= self.max_episode_seconds and raw_result is None
        if self.driver_count == 2:
            if raw_result == "agent":
                result = "win"
            elif raw_result == "opponent":
                result = "loss"
            elif raw_result == "draw":
                result = "draw"
            elif timed_out:
                result = "draw"
            else:
                result = ""
        else:
            result = "crash" if raw_result == "crash" else ""
        contact = any(event.get("type") == "car_contact" for event in self.simulation.last_result.events)
        return {
            "elapsed_time": self.simulation.elapsed_time,
            "speed": self.simulation.current_speed,
            "spawn_rate": self.simulation.current_spawn_rate,
            "phase": self.phase,
            "passed_groups": agent.passed_groups if agent is not None else 0,
            "collision_reason": agent.crash_reason if agent is not None and agent.crash_reason else "",
            "match_result": result,
            "raw_match_result": raw_result or "",
            "car_contact": contact,
            "reward_components": reward_components,
            "events": list(self.simulation.last_result.events),
            "applied_steering": agent.applied_steering if agent is not None else 0.0,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .rendering import PygameRenderer

            self._renderer = PygameRenderer()
        return self._renderer.render(self.simulation, driver_index=0, mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["TurboDodgeEnv"]
