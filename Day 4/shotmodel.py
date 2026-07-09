"""Q-agent shooter for the Day 4 basketball game.

The game calls the entry method with its interface functions. The model reads
the state, chooses a y-angle offset and a speed, then returns the release mouse
position it used for the shot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import pygame


Position = tuple[float, float]
GameFunction = Callable[[], Any]


@dataclass(frozen=True)
class Action:
    y_offset: float
    speed: float


@dataclass(frozen=True)
class Shot:
    click_pos: Position
    release_pos: Position
    action: Action


class QAgent:
    def __init__(self):
        self.epsilon = 0.75
        self.learning_rate = 0.25
        self.actions = [
            Action(y_offset, speed)
            for y_offset in ( 400,410,450,500,510, 520,530,540,550, 560, 600,620, 640, 680)
            for speed in ( 720, 780, 840, 900, 960, 1020, 1080, 1140, 1200)
        ]
        self.q_table: dict[tuple[int, int, int], float] = {}
        self.last_key: tuple[int, int, int] | None = None
        self.last_action: Action | None = None
        self.attempts = 0
        self.makes = 0

    def entry(
        self,
        clickonball: GameFunction,
        releaseonball: GameFunction,
        reset_ball: GameFunction,
        get_state: GameFunction,
    ) -> Position:
        """Choose and take a shot, then return the release mouse position."""
        reset_ball()
        state = get_state()
        shot = self.choose_shot(state)
        self.last_key = self._state_action_key(state, shot.action)
        print(
            "Model decision:",
            f"y_offset={shot.action.y_offset}",
            f"speed={shot.action.speed}",
            f"release_pos=({int(shot.release_pos[0])}, {int(shot.release_pos[1])})",
            f"attempts={self.attempts}",
            f"makes={self.makes}",
            f"epsilon={self.epsilon:.3f}",
        )
        self._call_with_mouse_pos(clickonball, shot.click_pos)
        self._call_with_mouse_pos(releaseonball, shot.release_pos)
        return shot.release_pos

    def choose_shot(self, state: Any) -> Shot:
        ball_pos, _ball_vel, rim_rect, _backboard_rect = state
        click_pos = (float(ball_pos[0]), float(ball_pos[1]))
        action = self._choose_action(click_pos, rim_rect)
        release_pos = self._release_position(click_pos, rim_rect, action)
        print(release_pos)
        return Shot(click_pos, release_pos, action)

    def learn(self, state: Any, action: Action, reward: float) -> None:
        key = self._state_action_key(state, action)
        self._update_q_value(key, reward)

    def learn_from_result(self, scored: bool) -> None:
        if self.last_key is None:
            return

        reward = 1 if scored else -1
        self._update_q_value(self.last_key, reward)
        self.attempts += 1
        self.makes += int(scored)
        self.last_key = None
        self.epsilon = max(0.4, self.epsilon * 0.999)

    def has_active_shot(self) -> bool:
        return self.last_key is not None

    def _update_q_value(self, key: tuple[int, int, int], reward: float) -> None:
        old_value = self.q_table.get(key, 0)
        self.q_table[key] = old_value + self.learning_rate * (reward - old_value)

    def _choose_action(self, ball_pos: Position, rim_rect: Any) -> Action:
        if random.random() < self.epsilon:
            choices = [action for action in self.actions if action != self.last_action]
            action = random.choice(choices or self.actions)
            self.last_action = action
            return action

        action = max(
            self.actions,
            key=lambda action: self.q_table.get(
                self._state_action_key((ball_pos, None, rim_rect, None), action),
                self._starter_score(action),
            ),
        )
        self.last_action = action
        return action

    def _starter_score(self, action: Action) -> float:
        y_score = -abs(action.y_offset - 520) / 160
        speed_score = -abs(action.speed - 960) / 240
        return y_score + speed_score

    def _release_position(self, ball_pos: Position, rim_rect: Any, action: Action) -> Position:
        aim_pos = (float(rim_rect.centerx), float(rim_rect.centery) - action.y_offset)
        direction_x = aim_pos[0] - ball_pos[0]
        direction_y = aim_pos[1] - ball_pos[1]
        length = max(1, (direction_x**2 + direction_y**2) ** 0.5)
        return (
            ball_pos[0] + direction_x / length * action.speed,
            ball_pos[1] + direction_y / length * action.speed,
        )

    def _state_action_key(self, state: Any, action: Action) -> tuple[int, int, int]:
        ball_pos, _ball_vel, rim_rect, _backboard_rect = state
        distance_bucket = int((rim_rect.centerx - ball_pos[0]) // 100)
        height_bucket = int((rim_rect.centery - ball_pos[1]) // 50)
        action_bucket = self.actions.index(action)
        return (distance_bucket, height_bucket, action_bucket)

    def _call_with_mouse_pos(self, function: GameFunction, pos: Position) -> None:
        original_get_pos = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (int(pos[0]), int(pos[1]))
        try:
            function()
        finally:
            pygame.mouse.get_pos = original_get_pos


agent = QAgent()


def entry(
    clickonball: GameFunction,
    releaseonball: GameFunction,
    reset_ball: GameFunction,
    get_state: GameFunction,
) -> Position:
    return agent.entry(clickonball, releaseonball, reset_ball, get_state)


def shoot_ball(
    clickonball: GameFunction,
    releaseonball: GameFunction,
    reset_ball: GameFunction,
    get_state: GameFunction,
) -> Position:
    return entry(clickonball, releaseonball, reset_ball, get_state)


def learn_from_result(scored: bool) -> None:
    agent.learn_from_result(scored)


def has_active_shot() -> bool:
    return agent.has_active_shot()


def get_stats() -> tuple[int, int, float]:
    attempts = agent.attempts
    makes = agent.makes
    percentage = 0 if attempts == 0 else makes / attempts * 100
    return attempts, makes, percentage
