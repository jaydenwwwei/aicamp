"""Deterministic simulation shared by the game, renderer, and RL wrappers.

The coordinate system uses +Y as the direction of travel and X across the
road.  Rendering is deliberately kept out of this module so headless training
and human play use exactly the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PhaseSettings:
    initial_speed: float
    acceleration: float
    max_speed: float
    initial_spawn_rate: float
    spawn_growth: float
    max_spawn_rate: float
    max_group_size: int


PHASES: dict[str, PhaseSettings] = {
    "easy": PhaseSettings(6.0, 0.02, 10.0, 0.32, 0.002, 0.60, 2),
    "progressive": PhaseSettings(8.0, 0.06, 18.0, 0.55, 0.005, 1.15, 3),
    "advanced": PhaseSettings(10.0, 0.08, 24.0, 0.75, 0.006, 1.50, 4),
    "multiplayer": PhaseSettings(9.0, 0.07, 20.0, 0.55, 0.005, 1.15, 3),
}


@dataclass(frozen=True)
class GameConfig:
    dt: float = 1.0 / 30.0
    road_width: float = 18.0
    car_width: float = 2.0
    car_length: float = 4.5
    wheelbase: float = 2.7
    max_steering: float = math.radians(30.0)
    steering_rate: float = math.radians(120.0)
    lidar_range: float = 60.0
    spawn_min_ahead: float = 45.0
    spawn_max_ahead: float = 70.0
    match_limit_seconds: float = 120.0

    @property
    def road_half_width(self) -> float:
        return self.road_width / 2.0

    @property
    def usable_half_width(self) -> float:
        return self.road_half_width - self.car_width / 2.0


@dataclass
class DriverState:
    driver_id: str
    x: float
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    applied_steering: float = 0.0
    alive: bool = True
    crash_reason: str | None = None
    passed_groups: int = 0
    clean_overtakes: int = 0

    @property
    def velocity_x(self) -> float:
        return self.speed * math.sin(self.heading)

    @property
    def velocity_y(self) -> float:
        return self.speed * math.cos(self.heading)


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    length: float
    appearance: str


@dataclass
class ObstacleGroup:
    group_id: int
    obstacles: list[Obstacle]
    open_columns: tuple[int, ...] = ()
    passed_by: set[str] = field(default_factory=set)


@dataclass
class StepResult:
    elapsed_time: float
    drivers: dict[str, DriverState]
    events: list[dict[str, object]]
    match_result: str | None
    speed: float
    spawn_rate: float

    @property
    def finished(self) -> bool:
        """Whether a solo run or versus match has reached a terminal result."""

        return self.match_result is not None

    @property
    def outcome(self) -> str | None:
        return self.match_result

    @property
    def winner(self) -> str | None:
        return self.match_result if self.match_result not in {None, "draw", "crash"} else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Simulation:
    """A seeded endless-road simulation for one or two drivers.

    ``reset`` accepts arbitrary player IDs. ``step`` takes a mapping of these
    IDs to normalized steering actions. Missing actions are treated as zero.
    """

    lidar_angles: tuple[float, ...] = tuple(math.radians(value) for value in (-60, -40, -20, 0, 20, 40, 60))
    _column_centers: tuple[float, ...] = (-6.4, -3.2, 0.0, 3.2, 6.4)
    _appearances: tuple[str, ...] = ("barrier", "oil", "traffic", "sign")

    def __init__(
        self,
        phase: str = "progressive",
        seed: int | None = None,
        config: GameConfig | None = None,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {sorted(PHASES)}")
        self.config = config or GameConfig()
        self.phase = phase
        self.settings = PHASES[phase]
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.drivers: dict[str, DriverState] = {}
        self.obstacle_groups: list[ObstacleGroup] = []
        self.elapsed_time = 0.0
        self.versus = False
        self._next_group_id = 1
        self._last_group_y = -math.inf
        self._previous_leader: str | None = None
        self._overtake_candidate: tuple[str, float] | None = None
        self.last_result = StepResult(0.0, {}, [], None, 0.0, 0.0)

    def reset(
        self,
        seed: int | None = None,
        versus: bool = False,
        driver_ids: Sequence[str] | None = None,
        driver_count: int | None = None,
    ) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed
        if driver_count is not None:
            if driver_count not in {1, 2}:
                raise ValueError("driver_count must be one or two")
            versus = driver_count == 2
        self.versus = versus
        ids = tuple(driver_ids or (("human", "ai") if versus else ("human",)))
        if not ids or len(ids) > 2:
            raise ValueError("The simulation supports one or two drivers")
        if versus and len(ids) != 2:
            raise ValueError("Versus simulations need exactly two drivers")

        self.elapsed_time = 0.0
        self.obstacle_groups = []
        self._next_group_id = 1
        self._last_group_y = -math.inf
        self._previous_leader = None
        self._overtake_candidate = None
        initial_speed = self.settings.initial_speed
        if len(ids) == 1:
            positions = (0.0,)
        else:
            positions = (-2.25, 2.25)
            if bool(self.rng.integers(0, 2)):
                positions = tuple(reversed(positions))
        self.drivers = {
            driver_id: DriverState(driver_id=driver_id, x=positions[index], speed=initial_speed)
            for index, driver_id in enumerate(ids)
        }
        self.last_result = self._make_result([])
        return self.last_result

    @property
    def dt(self) -> float:
        """Public fixed timestep used by the UI and Gymnasium wrapper."""

        return self.config.dt

    @property
    def road_width(self) -> float:
        return self.config.road_width

    @property
    def driver_count(self) -> int:
        return len(self.drivers)

    @property
    def passed_groups(self) -> int:
        """Score shown to the current human-facing renderer."""

        preferred = self.drivers.get("human") or self.drivers.get("agent")
        return preferred.passed_groups if preferred is not None else 0

    @property
    def current_speed(self) -> float:
        return min(self.settings.initial_speed + self.settings.acceleration * self.elapsed_time, self.settings.max_speed)

    @property
    def current_spawn_rate(self) -> float:
        return min(
            self.settings.initial_spawn_rate + self.settings.spawn_growth * self.elapsed_time,
            self.settings.max_spawn_rate,
        )

    def step(self, actions: Mapping[str, float] | Sequence[float] | None = None) -> StepResult:
        """Advance one fixed-duration physics step."""
        action_map = self._normalise_actions(actions)
        events: list[dict[str, object]] = []
        speed = self.current_speed
        for driver_id, driver in self.drivers.items():
            if not driver.alive:
                continue
            action = _clip(float(action_map.get(driver_id, 0.0)), -1.0, 1.0)
            target = action * self.config.max_steering
            maximum_change = self.config.steering_rate * self.config.dt
            steering_difference = _clip(target - driver.applied_steering, -maximum_change, maximum_change)
            driver.applied_steering += steering_difference
            driver.speed = speed
            driver.heading = _wrap_angle(
                driver.heading + (driver.speed / self.config.wheelbase) * math.tan(driver.applied_steering) * self.config.dt
            )
            driver.x += driver.velocity_x * self.config.dt
            driver.y += driver.velocity_y * self.config.dt

        self.elapsed_time += self.config.dt
        self._maybe_spawn_group()
        self._detect_collisions(events)
        self._record_passed_groups(events)
        self._record_overtakes(events)
        self._cleanup_groups()
        self.last_result = self._make_result(events)
        return self.last_result

    def _normalise_actions(self, actions: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
        if actions is None:
            return {}
        if isinstance(actions, Mapping):
            return {str(key): float(value) for key, value in actions.items()}
        return {driver_id: float(action) for driver_id, action in zip(self.drivers, actions)}

    def _maybe_spawn_group(self) -> None:
        living = [driver for driver in self.drivers.values() if driver.alive]
        if not living:
            return
        maximum_y = max(driver.y for driver in living)
        minimum_spacing = max(16.0, self.current_speed)
        if maximum_y + self.config.spawn_min_ahead < self._last_group_y + minimum_spacing:
            return
        probability = 1.0 - math.exp(-self.current_spawn_rate * self.config.dt)
        if float(self.rng.random()) > probability:
            return
        spawn_y = maximum_y + float(self.rng.uniform(self.config.spawn_min_ahead, self.config.spawn_max_ahead))
        spawn_y = max(spawn_y, self._last_group_y + minimum_spacing)
        group = self._build_group(spawn_y)
        self.obstacle_groups.append(group)
        self._last_group_y = spawn_y

    def _build_group(self, y: float) -> ObstacleGroup:
        # Two cars need two clear columns. The far spawn distance and clear
        # corridors are a practical reachability guarantee for this compact game.
        required_gaps = 2 if self.versus else 1
        maximum_hazards = min(self.settings.max_group_size, len(self._column_centers) - required_gaps)
        selected: np.ndarray | None = None
        # A generated row is only accepted when every living car can reach at
        # least one advertised open virtual column before the row arrives.
        for _ in range(32):
            hazard_count = int(self.rng.integers(1, maximum_hazards + 1))
            candidate = self.rng.choice(len(self._column_centers), size=hazard_count, replace=False)
            open_columns = tuple(index for index in range(len(self._column_centers)) if index not in set(candidate.tolist()))
            if self._open_columns_reachable(open_columns, y):
                selected = candidate
                break
        chosen_columns = selected if selected is not None else self.rng.choice(
            len(self._column_centers), size=maximum_hazards, replace=False
        )
        open_columns = tuple(index for index in range(len(self._column_centers)) if index not in set(chosen_columns.tolist()))
        obstacles: list[Obstacle] = []
        for column in chosen_columns:
            x = float(self._column_centers[int(column)] + self.rng.uniform(-0.35, 0.35))
            obstacles.append(
                Obstacle(
                    x=x,
                    y=y + float(self.rng.uniform(-0.5, 0.5)),
                    width=float(self.rng.uniform(1.45, 1.85)),
                    length=float(self.rng.uniform(2.0, 3.6)),
                    appearance=str(self.rng.choice(self._appearances)),
                )
            )
        group = ObstacleGroup(self._next_group_id, obstacles, open_columns=open_columns)
        self._next_group_id += 1
        return group

    def _open_columns_reachable(self, open_columns: Sequence[int], spawn_y: float) -> bool:
        for driver in self.drivers.values():
            if not driver.alive:
                continue
            time_to_row = max(0.0, (spawn_y - driver.y) / max(1.0, driver.speed))
            # Conservative lateral distance after steering actuator delay. It
            # is deliberately capped by the road's usable width.
            lateral_reach = min(
                self.config.usable_half_width * 2.0,
                max(1.0, 0.5 * driver.speed * max(0.0, time_to_row - 0.2)),
            )
            if not any(abs(self._column_centers[column] - driver.x) <= lateral_reach for column in open_columns):
                return False
        return True

    def _detect_collisions(self, events: list[dict[str, object]]) -> None:
        for driver in self.drivers.values():
            if not driver.alive:
                continue
            lateral_extent = (
                abs(math.cos(driver.heading)) * self.config.car_width / 2.0
                + abs(math.sin(driver.heading)) * self.config.car_length / 2.0
            )
            if abs(driver.x) + lateral_extent >= self.config.road_half_width:
                self._crash(driver, "road_edge", events)
                continue
            for group in self.obstacle_groups:
                if any(self._overlaps_obstacle(driver, obstacle) for obstacle in group.obstacles):
                    self._crash(driver, "hazard", events)
                    break

        live_drivers = [driver for driver in self.drivers.values() if driver.alive]
        if len(live_drivers) == 2:
            first, second = live_drivers
            if self._cars_overlap(first, second):
                self._crash(first, "car_contact", events)
                self._crash(second, "car_contact", events)
                events.append({"type": "car_contact", "drivers": (first.driver_id, second.driver_id)})

    def _overlaps_obstacle(self, driver: DriverState, obstacle: Obstacle) -> bool:
        return self._oriented_rectangles_overlap(
            driver.x,
            driver.y,
            driver.heading,
            self.config.car_width / 2.0,
            self.config.car_length / 2.0,
            obstacle.x,
            obstacle.y,
            0.0,
            obstacle.width / 2.0,
            obstacle.length / 2.0,
        )

    def _cars_overlap(self, first: DriverState, second: DriverState) -> bool:
        return self._oriented_rectangles_overlap(
            first.x,
            first.y,
            first.heading,
            self.config.car_width / 2.0,
            self.config.car_length / 2.0,
            second.x,
            second.y,
            second.heading,
            self.config.car_width / 2.0,
            self.config.car_length / 2.0,
        )

    @staticmethod
    def _oriented_rectangles_overlap(
        first_x: float,
        first_y: float,
        first_heading: float,
        first_half_width: float,
        first_half_length: float,
        second_x: float,
        second_y: float,
        second_heading: float,
        second_half_width: float,
        second_half_length: float,
    ) -> bool:
        """Separating-axis test for two road-aligned/rotated car rectangles."""

        def axes(heading: float) -> tuple[tuple[float, float], tuple[float, float]]:
            return (math.cos(heading), -math.sin(heading)), (math.sin(heading), math.cos(heading))

        first_lateral, first_forward = axes(first_heading)
        second_lateral, second_forward = axes(second_heading)
        delta_x, delta_y = second_x - first_x, second_y - first_y
        for axis_x, axis_y in (first_lateral, first_forward, second_lateral, second_forward):
            center_distance = abs(delta_x * axis_x + delta_y * axis_y)
            first_radius = (
                first_half_width * abs(first_lateral[0] * axis_x + first_lateral[1] * axis_y)
                + first_half_length * abs(first_forward[0] * axis_x + first_forward[1] * axis_y)
            )
            second_radius = (
                second_half_width * abs(second_lateral[0] * axis_x + second_lateral[1] * axis_y)
                + second_half_length * abs(second_forward[0] * axis_x + second_forward[1] * axis_y)
            )
            if center_distance >= first_radius + second_radius:
                return False
        return True

    def _crash(self, driver: DriverState, reason: str, events: list[dict[str, object]]) -> None:
        if not driver.alive:
            return
        driver.alive = False
        driver.crash_reason = reason
        events.append({"type": "crash", "driver": driver.driver_id, "reason": reason})

    def _record_passed_groups(self, events: list[dict[str, object]]) -> None:
        rear_margin = self.config.car_length / 2.0
        for group in self.obstacle_groups:
            group_rear = max(obstacle.y + obstacle.length / 2.0 for obstacle in group.obstacles)
            for driver in self.drivers.values():
                if driver.alive and driver.driver_id not in group.passed_by and group_rear < driver.y - rear_margin:
                    group.passed_by.add(driver.driver_id)
                    driver.passed_groups += 1
                    events.append({"type": "group_passed", "driver": driver.driver_id, "group_id": group.group_id})

    def _record_overtakes(self, events: list[dict[str, object]]) -> None:
        if len(self.drivers) != 2:
            return
        active = [driver for driver in self.drivers.values() if driver.alive]
        if len(active) != 2:
            return
        leader = max(active, key=lambda driver: driver.y).driver_id
        if self._previous_leader is None:
            self._previous_leader = leader
            return
        if leader != self._previous_leader:
            self._previous_leader = leader
            self._overtake_candidate = (leader, self.elapsed_time)
            return
        if self._overtake_candidate is not None and self._overtake_candidate[0] == leader:
            other = next(driver for driver in active if driver.driver_id != leader)
            new_leader = self.drivers[leader]
            candidate_time = self._overtake_candidate[1]
            if new_leader.y - other.y >= self.config.car_length and self.elapsed_time - candidate_time >= 1.0:
                new_leader.clean_overtakes += 1
                events.append({"type": "overtake", "driver": leader})
                self._overtake_candidate = None

    def _cleanup_groups(self) -> None:
        if not self.drivers:
            return
        floor = min(driver.y for driver in self.drivers.values()) - 25.0
        self.obstacle_groups = [
            group
            for group in self.obstacle_groups
            if any(obstacle.y + obstacle.length / 2.0 >= floor for obstacle in group.obstacles)
        ]

    def _make_result(self, events: list[dict[str, object]]) -> StepResult:
        match_result: str | None = None
        if len(self.drivers) == 2:
            alive = [driver.driver_id for driver in self.drivers.values() if driver.alive]
            if not alive:
                match_result = "draw"
            elif len(alive) == 1:
                match_result = alive[0]
            elif self.elapsed_time >= self.config.match_limit_seconds:
                match_result = "draw"
        elif len(self.drivers) == 1:
            only_driver = next(iter(self.drivers.values()))
            if not only_driver.alive:
                match_result = "crash"
        return StepResult(
            elapsed_time=self.elapsed_time,
            drivers=self.drivers,
            events=events,
            match_result=match_result,
            speed=self.current_speed,
            spawn_rate=self.current_spawn_rate,
        )

    def lidar_distances(self, driver_id: str) -> np.ndarray:
        driver = self.drivers[driver_id]
        distances = [self._raycast(driver, driver.heading + angle) for angle in self.lidar_angles]
        return np.asarray(distances, dtype=np.float32)

    def _raycast(self, driver: DriverState, angle: float) -> float:
        direction_x, direction_y = math.sin(angle), math.cos(angle)
        closest = self.config.lidar_range
        targets: list[tuple[float, float, float, float]] = []
        for group in self.obstacle_groups:
            for obstacle in group.obstacles:
                targets.append((obstacle.x, obstacle.y, obstacle.width / 2.0, obstacle.length / 2.0))
        for other in self.drivers.values():
            if other.driver_id != driver.driver_id and other.alive:
                targets.append((other.x, other.y, self.config.car_width / 2.0, self.config.car_length / 2.0))
        for center_x, center_y, half_width, half_length in targets:
            hit = self._ray_aabb_distance(
                driver.x, driver.y, direction_x, direction_y, center_x - half_width, center_x + half_width, center_y - half_length, center_y + half_length
            )
            if hit is not None:
                closest = min(closest, hit)
        return closest

    def _ray_aabb_distance(
        self,
        origin_x: float,
        origin_y: float,
        direction_x: float,
        direction_y: float,
        minimum_x: float,
        maximum_x: float,
        minimum_y: float,
        maximum_y: float,
    ) -> float | None:
        lower, upper = -math.inf, math.inf
        for origin, direction, minimum, maximum in (
            (origin_x, direction_x, minimum_x, maximum_x),
            (origin_y, direction_y, minimum_y, maximum_y),
        ):
            if abs(direction) < 1e-9:
                if origin < minimum or origin > maximum:
                    return None
                continue
            first, second = (minimum - origin) / direction, (maximum - origin) / direction
            if first > second:
                first, second = second, first
            lower, upper = max(lower, first), min(upper, second)
            if lower > upper:
                return None
        if upper < 0.0:
            return None
        distance = max(0.0, lower)
        return distance if distance <= self.config.lidar_range else None

    def observation(self, driver_id: str) -> np.ndarray:
        """Return the documented 19-feature normalized observation vector."""
        driver = self.drivers[driver_id]
        maximum_speed = self.settings.max_speed
        usable = self.config.usable_half_width
        lidar = self.lidar_distances(driver_id) / self.config.lidar_range
        left_distance = _clip((driver.x + usable) / (2.0 * usable), 0.0, 1.0)
        right_distance = _clip((usable - driver.x) / (2.0 * usable), 0.0, 1.0)
        other = next((candidate for candidate in self.drivers.values() if candidate.driver_id != driver_id), None)
        if other is None:
            opponent = (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            opponent = (
                1.0,
                _clip((other.x - driver.x) / (2.0 * usable), -1.0, 1.0),
                _clip((other.y - driver.y) / self.config.lidar_range, -1.0, 1.0),
                _clip((other.velocity_x - driver.velocity_x) / maximum_speed, -1.0, 1.0),
                _clip((other.velocity_y - driver.velocity_y) / maximum_speed, -1.0, 1.0),
            )
        values = np.asarray(
            [
                _clip(driver.x / usable, -1.0, 1.0),
                _clip(driver.velocity_x / maximum_speed, -1.0, 1.0),
                _clip(driver.velocity_y / maximum_speed, -1.0, 1.0),
                _wrap_angle(driver.heading) / math.pi,
                _clip(driver.applied_steering / self.config.max_steering, -1.0, 1.0),
                *lidar.tolist(),
                left_distance,
                right_distance,
                *opponent,
            ],
            dtype=np.float32,
        )
        return values

    # Adapters intentionally keep the Pygame UI and SB3 wrapper independent
    # from the exact internal method spelling.
    observation_for = observation
    get_observation = observation

    def ai_action(self, driver_id: str = "ai") -> float:
        return self.heuristic_action(driver_id)

    def heuristic_action(self, driver_id: str) -> float:
        """A safe fallback policy used as a benchmark and when no model exists."""
        driver = self.drivers[driver_id]
        lidar = self.lidar_distances(driver_id)
        forward = float(lidar[3])
        left_clearance = float(np.mean(lidar[:3]))
        right_clearance = float(np.mean(lidar[4:]))
        steer = -0.22 * driver.x / self.config.usable_half_width
        if forward < 24.0:
            steer += 0.8 if right_clearance > left_clearance else -0.8
        for other in self.drivers.values():
            if other.driver_id != driver_id and other.alive and 0.0 < other.y - driver.y < 10.0:
                steer += -0.35 if other.x > driver.x else 0.35
        return _clip(steer, -1.0, 1.0)
