"""Obstacle data and deterministic layout helpers for Turbo Dodge AI.

This module deliberately contains no rendering code.  A renderer can use the
``kind`` field to choose artwork while the simulator always uses the same
collision dimensions for every hazard type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np


ROAD_WIDTH = 18.0
ROAD_HALF_WIDTH = ROAD_WIDTH / 2.0

# Five evenly spaced virtual columns leave a usable 3.6 m corridor whenever
# one column is open.  Hazards are deliberately narrower than a column.
LANE_CENTERS: tuple[float, ...] = (-7.2, -3.6, 0.0, 3.6, 7.2)
LANE_SPACING = 3.6
HAZARD_WIDTH = 2.4
HAZARD_LENGTH = 3.0
HAZARD_KINDS: tuple[str, ...] = ("barrier", "oil", "traffic", "sign")


@dataclass(slots=True)
class Obstacle:
    """One stationary, axis-aligned world-space hazard.

    Coordinates use metres.  ``y`` increases in the direction cars drive.
    The simulator currently gives all kinds equal geometry so visual variety
    cannot accidentally alter the learning task.
    """

    identifier: int
    group_id: int
    x: float
    y: float
    kind: str
    width: float = HAZARD_WIDTH
    length: float = HAZARD_LENGTH
    heading: float = 0.0


@dataclass(slots=True)
class ObstacleGroup:
    """Hazards that form one dodge decision and score as one pass."""

    identifier: int
    y: float
    obstacles: list[Obstacle]
    open_lanes: tuple[int, ...]
    passed_by: set[int] = field(default_factory=set)

    @property
    def front_y(self) -> float:
        """Furthest forward collision extent in world coordinates."""

        return max(obstacle.y + obstacle.length / 2.0 for obstacle in self.obstacles)

    @property
    def rear_y(self) -> float:
        """Furthest rear collision extent in world coordinates."""

        return min(obstacle.y - obstacle.length / 2.0 for obstacle in self.obstacles)

    @property
    def blocked_lanes(self) -> tuple[int, ...]:
        """The virtual columns occupied by this group."""

        return tuple(index for index in range(len(LANE_CENTERS)) if index not in self.open_lanes)


def open_lane_sets(blocked_count: int) -> Iterable[tuple[int, ...]]:
    """Yield every possible set of open lanes for ``blocked_count`` hazards."""

    lane_count = len(LANE_CENTERS)
    open_count = lane_count - blocked_count
    for open_set in combinations(range(lane_count), open_count):
        yield open_set


def build_group(
    *,
    group_id: int,
    obstacle_start_id: int,
    y: float,
    open_lanes: tuple[int, ...],
    rng: np.random.Generator,
) -> ObstacleGroup:
    """Build a group from selected open virtual columns.

    A small lateral jitter makes patterns look less grid-like while retaining
    a comfortably wide physical gap.  Jitter is generated exclusively from
    the supplied seeded generator.
    """

    obstacles: list[Obstacle] = []
    blocked = [index for index in range(len(LANE_CENTERS)) if index not in open_lanes]
    for offset, lane_index in enumerate(blocked):
        jitter = float(rng.uniform(-0.18, 0.18))
        kind_index = int(rng.integers(0, len(HAZARD_KINDS)))
        obstacles.append(
            Obstacle(
                identifier=obstacle_start_id + offset,
                group_id=group_id,
                x=LANE_CENTERS[lane_index] + jitter,
                y=y,
                kind=HAZARD_KINDS[kind_index],
            )
        )
    return ObstacleGroup(
        identifier=group_id,
        y=y,
        obstacles=obstacles,
        open_lanes=tuple(sorted(open_lanes)),
    )
