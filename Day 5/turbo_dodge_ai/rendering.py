"""Procedural Pygame rendering for Turbo Dodge AI.

This module deliberately knows very little about the simulation implementation.
It reads the small public surface used by the game (`drivers`,
`obstacle_groups`, and a few optional progress fields) and degrades gracefully
when a field is absent.  Keeping the renderer this loose makes it safe to use
for the Gymnasium environment as well as the interactive game.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pygame


Color = tuple[int, int, int]


@dataclass(frozen=True)
class RenderTheme:
    """Colour palette for the procedural cockpit view."""

    sky_top: Color = (14, 24, 51)
    sky_bottom: Color = (74, 121, 174)
    grass: Color = (33, 98, 58)
    road: Color = (42, 46, 54)
    road_edge: Color = (236, 238, 241)
    lane: Color = (244, 204, 64)
    cockpit_dark: Color = (11, 13, 19)
    cockpit_mid: Color = (35, 39, 49)
    f1_red: Color = (211, 40, 43)
    f1_red_highlight: Color = (255, 81, 69)
    hud_text: Color = (241, 247, 255)
    warning: Color = (255, 186, 73)
    danger: Color = (255, 89, 89)


def _field(value: Any, *names: str, default: Any = None) -> Any:
    """Read an attribute or mapping item without tying rendering to a model."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            candidate = getattr(value, name)
            if callable(candidate):
                continue
            return candidate
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _iter_values(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, (str, bytes)):
        return ()
    try:
        return iter(value)
    except TypeError:
        return (value,)


class CockpitRenderer:
    """Draw a lightweight third-person road view from simulation coordinates.

    Coordinates are interpreted as metres on a road where heading 0 faces
    positive Y.  If a simulation provides ``relative_y`` / ``distance`` on an
    obstacle, that value takes precedence.  This makes the drawing code usable
    with both world-space and ego-relative obstacle implementations.
    """

    def __init__(
        self,
        surface: pygame.Surface,
        *,
        road_width: float = 18.0,
        max_view_distance: float = 70.0,
        theme: RenderTheme | None = None,
    ) -> None:
        self.surface = surface
        self.road_width = road_width
        self.max_view_distance = max_view_distance
        self.theme = theme or RenderTheme()
        self._fonts: dict[int, pygame.font.Font] = {}

    def set_surface(self, surface: pygame.Surface) -> None:
        """Use a resized/recreated display surface."""

        self.surface = surface

    def render(
        self,
        simulation: Any,
        *,
        focus_driver: str | int | None = "human",
        mode: str = "solo",
        paused: bool = False,
        show_debug: bool = False,
        message: str | None = None,
        training_status: Mapping[str, Any] | None = None,
    ) -> None:
        """Draw one complete simulation frame, without flipping the display."""

        driver_id, driver = self._focus_driver(simulation, focus_driver)
        self._draw_sky()
        self._draw_road(driver, simulation)
        self._draw_obstacles(simulation, driver)
        self._draw_other_drivers(simulation, driver_id, driver)
        self._draw_player_car(driver, driver_id)
        self._draw_hud(simulation, driver_id, driver, mode, training_status)

        if show_debug:
            self._draw_debug_overlay(simulation, driver_id, driver)
        if paused:
            self._draw_center_banner("PAUSED", "Press P to continue")
        if message:
            self._draw_result_banner(message)

    def rgb_array(self) -> np.ndarray:
        """Return the current surface in Gymnasium's H×W×RGB arrangement."""

        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1).copy()

    # ------------------------------------------------------------------
    # Scene geometry

    @property
    def _size(self) -> tuple[int, int]:
        return self.surface.get_size()

    @property
    def _horizon_y(self) -> int:
        return int(self._size[1] * 0.245)

    @property
    def _road_bottom_y(self) -> int:
        return int(self._size[1] * 0.98)

    def _font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _road_metrics(self) -> tuple[float, float, float, float]:
        width, _ = self._size
        centre = width / 2.0
        return centre, width * 0.074, width * 0.465, self._horizon_y

    def _project(self, lateral_metres: float, forward_metres: float) -> tuple[int, int, float]:
        """Project a local-road coordinate to screen x/y and perspective size."""

        centre, horizon_half, bottom_half, horizon = self._road_metrics()
        distance = _clamp(forward_metres, 0.0, self.max_view_distance)
        closeness = 1.0 - distance / self.max_view_distance
        # Slightly curved depth helps the road feel deeper than a flat trapezoid.
        depth = closeness**1.55
        half_road = horizon_half + (bottom_half - horizon_half) * depth
        road_y = horizon + (self._road_bottom_y - horizon) * depth
        x = centre + lateral_metres / (self.road_width / 2.0) * half_road
        return int(x), int(road_y), depth

    def _road_curve_offset(self, world_y: float) -> float:
        """Visual-only centreline curve for the road ahead."""

        return 2.4 * math.sin(world_y * 0.035) + 1.35 * math.sin(world_y * 0.011 + 1.8)

    def _visual_road_x(self, road_x: float, world_y: float) -> float:
        return road_x + self._road_curve_offset(world_y)

    def _driver_visual_x(self, driver: Any) -> float:
        driver_x = _number(_field(driver, "x", "position_x", "center_x"), 0.0)
        driver_y = _number(_field(driver, "y", "position_y", "center_y"), 0.0)
        return self._visual_road_x(driver_x, driver_y)

    def _camera_visual_x(self, driver: Any) -> float:
        driver_y = _number(_field(driver, "y", "position_y", "center_y"), 0.0)
        return self._road_curve_offset(driver_y)

    def _road_point(self, road_x: float, distance_ahead: float, driver: Any) -> tuple[float, float]:
        """Convert a road-layout coordinate into the driver's forward view."""

        driver_y = _number(_field(driver, "y", "position_y", "center_y"), 0.0)
        world_y = driver_y + distance_ahead
        lateral = self._visual_road_x(road_x, world_y) - self._camera_visual_x(driver)
        return lateral, distance_ahead

    def _draw_sky(self) -> None:
        width, height = self._size
        horizon = self._horizon_y
        top = self.theme.sky_top
        bottom = self.theme.sky_bottom
        for y in range(max(1, horizon)):
            mix = y / max(1, horizon - 1)
            colour = tuple(int(top[i] + (bottom[i] - top[i]) * mix) for i in range(3))
            pygame.draw.line(self.surface, colour, (0, y), (width, y))
        pygame.draw.rect(self.surface, self.theme.grass, (0, horizon, width, height - horizon))

        # Simple, deliberately abstract mountains at the horizon give the road a
        # destination without requiring an image asset.
        ridge = [
            (0, horizon + 20),
            (int(width * 0.12), horizon - 8),
            (int(width * 0.24), horizon + 10),
            (int(width * 0.38), horizon - 15),
            (int(width * 0.56), horizon + 12),
            (int(width * 0.72), horizon - 5),
            (width, horizon + 18),
        ]
        pygame.draw.polygon(self.surface, (36, 67, 82), ridge + [(width, horizon + 35), (0, horizon + 35)])

    def _draw_road(self, driver: Any, simulation: Any) -> None:
        width, _ = self._size
        road_bottom = self._road_bottom_y
        half_road = self.road_width / 2.0
        distances = np.linspace(-3.0, self.max_view_distance, 34)

        def projected_edge(edge_x: float) -> list[tuple[int, int]]:
            points: list[tuple[int, int]] = []
            for distance in distances:
                lateral, forward = self._road_point(edge_x, float(distance), driver)
                if forward >= -1.0:
                    points.append(self._project(lateral, forward))
            return [(x, y) for x, y, _ in points]

        left_edge = projected_edge(-half_road)
        right_edge = projected_edge(half_road)
        if len(left_edge) >= 2 and len(right_edge) >= 2:
            road = left_edge + list(reversed(right_edge))
        else:
            centre, horizon_half, bottom_half, horizon = self._road_metrics()
            road = [
                (int(centre - horizon_half), horizon),
                (int(centre + horizon_half), horizon),
                (int(centre + bottom_half), road_bottom),
                (int(centre - bottom_half), road_bottom),
            ]
        pygame.draw.polygon(self.surface, self.theme.road, road)

        # White road-edge strips.
        edge_width = max(4, int(width * 0.008))
        if len(left_edge) >= 2:
            pygame.draw.lines(self.surface, self.theme.road_edge, False, left_edge, edge_width)
        if len(right_edge) >= 2:
            pygame.draw.lines(self.surface, self.theme.road_edge, False, right_edge, edge_width)

        # Dashed centre and lane markers are world-spaced. Curves come from the
        # road layout, not from rotating the road with the car's steering.
        elapsed = self._elapsed_time(simulation)
        phase = (elapsed * max(1.0, _number(_field(driver, "speed", "forward_speed"), 0.0))) % 8.0
        lanes = (-self.road_width / 6.0, self.road_width / 6.0)
        for lane_x in lanes:
            for start in np.arange(4.0 - phase, self.max_view_distance, 8.0):
                end = start + 3.3
                if end <= 0:
                    continue
                lateral1, forward1 = self._road_point(lane_x, max(0.0, float(start)), driver)
                lateral2, forward2 = self._road_point(lane_x, min(self.max_view_distance, float(end)), driver)
                if forward1 < 0.0 or forward2 <= 0.0:
                    continue
                x1, y1, depth1 = self._project(lateral1, forward1)
                x2, y2, depth2 = self._project(lateral2, forward2)
                line_width = max(1, int(2 + 8 * max(depth1, depth2)))
                pygame.draw.line(self.surface, self.theme.lane, (x1, y1), (x2, y2), line_width)

    def _draw_obstacles(self, simulation: Any, driver: Any) -> None:
        drawables: list[tuple[float, float, Any]] = []
        for obstacle in self._iter_obstacles(simulation):
            lateral, forward = self._relative_position(obstacle, driver)
            if 0.0 < forward <= self.max_view_distance:
                drawables.append((forward, lateral, obstacle))

        # Far objects first makes closer Pygame primitives naturally occlude them.
        for forward, lateral, obstacle in sorted(drawables, key=lambda item: item[0], reverse=True):
            self._draw_obstacle(obstacle, lateral, forward)

    def _draw_obstacle(self, obstacle: Any, lateral: float, forward: float) -> None:
        x, y, depth = self._project(lateral, forward)
        width_m = _number(_field(obstacle, "width", "w", "size_x"), 2.0)
        length_m = _number(_field(obstacle, "length", "height", "h", "size_y"), 2.0)
        scale = 6.0 + depth * min(self._size) * 0.105
        half_w = max(3, int(width_m * scale / 2.0))
        h = max(4, int(length_m * scale * 0.68))
        kind = str(_field(obstacle, "kind", "type", "appearance", "name", default="barrier")).lower()

        shadow = pygame.Rect(x - int(half_w * 1.05), y - int(h * 0.15), int(half_w * 2.1), max(3, int(h * 0.32)))
        pygame.draw.ellipse(self.surface, (22, 27, 29), shadow)

        if "oil" in kind or "slick" in kind:
            slick = pygame.Rect(x - half_w, y - h // 2, half_w * 2, max(4, h))
            pygame.draw.ellipse(self.surface, (10, 11, 16), slick)
            pygame.draw.ellipse(self.surface, (52, 60, 72), slick, max(1, int(depth * 3)))
            return

        if "sign" in kind:
            pole_w = max(2, half_w // 5)
            pygame.draw.rect(self.surface, (126, 132, 139), (x - pole_w // 2, y - h, pole_w, h))
            face = pygame.Rect(x - half_w, y - h * 2, half_w * 2, h)
            pygame.draw.rect(self.surface, (245, 194, 58), face, border_radius=max(1, half_w // 5))
            pygame.draw.rect(self.surface, (40, 43, 50), face, max(1, int(depth * 3)))
            return

        if "car" in kind or "traffic" in kind or "vehicle" in kind:
            body = [(x - half_w, y), (x - int(half_w * 0.65), y - h), (x + int(half_w * 0.65), y - h), (x + half_w, y)]
            pygame.draw.polygon(self.surface, (51, 152, 213), body)
            pygame.draw.polygon(self.surface, (193, 231, 249), [(x - int(half_w * 0.42), y - int(h * 0.78)), (x + int(half_w * 0.42), y - int(h * 0.78)), (x + int(half_w * 0.28), y - int(h * 0.47)), (x - int(half_w * 0.28), y - int(h * 0.47))])
            pygame.draw.circle(self.surface, (255, 67, 65), (x - int(half_w * 0.54), y - int(h * 0.12)), max(1, half_w // 6))
            pygame.draw.circle(self.surface, (255, 67, 65), (x + int(half_w * 0.54), y - int(h * 0.12)), max(1, half_w // 6))
            return

        # Barrier is the default and deliberately has high contrast at speed.
        rect = pygame.Rect(x - half_w, y - h, half_w * 2, h)
        pygame.draw.rect(self.surface, (232, 113, 43), rect, border_radius=max(1, half_w // 4))
        stripe_h = max(2, h // 4)
        for offset in range(-half_w * 2, half_w * 2, max(4, half_w // 2)):
            points = [(x + offset, y), (x + offset + half_w // 2, y), (x + offset + half_w * 2, y - h), (x + offset + int(half_w * 1.5), y - h)]
            pygame.draw.polygon(self.surface, (248, 222, 91), points)
        pygame.draw.rect(self.surface, (55, 43, 34), rect, max(1, int(depth * 3)))

    def _draw_other_drivers(self, simulation: Any, focus_id: str | None, focus: Any) -> None:
        cars: list[tuple[float, float, str, Any]] = []
        for identifier, driver in self._drivers(simulation).items():
            if identifier == focus_id or not bool(_field(driver, "alive", default=True)):
                continue
            lateral, forward = self._relative_position(driver, focus)
            if 0.0 < forward <= self.max_view_distance:
                cars.append((forward, lateral, str(identifier), driver))
        for forward, lateral, identifier, _ in sorted(cars, key=lambda item: item[0], reverse=True):
            x, y, depth = self._project(lateral, forward)
            scale = 13.0 + depth * min(self._size) * 0.145
            half_w = int(scale * 0.62)
            height = int(scale * 1.32)
            shadow = pygame.Rect(x - int(half_w * 1.05), y - int(height * 0.12), int(half_w * 2.1), int(height * 0.25))
            pygame.draw.ellipse(self.surface, (15, 19, 22), shadow)
            body = pygame.Rect(x - half_w, y - height, half_w * 2, height)
            pygame.draw.rect(self.surface, (37, 133, 202), body, border_radius=max(3, half_w // 5))
            pygame.draw.rect(self.surface, (24, 72, 126), (x - half_w, y - int(height * 0.28), half_w * 2, int(height * 0.28)), border_radius=max(2, half_w // 6))
            cabin = [
                (x - int(half_w * 0.55), y - int(height * 0.55)),
                (x - int(half_w * 0.33), y - int(height * 0.82)),
                (x + int(half_w * 0.33), y - int(height * 0.82)),
                (x + int(half_w * 0.55), y - int(height * 0.55)),
            ]
            pygame.draw.polygon(self.surface, (18, 35, 57), cabin)
            for wheel_x in (x - half_w - max(2, half_w // 8), x + half_w - max(2, half_w // 8)):
                pygame.draw.rect(self.surface, (10, 11, 15), (wheel_x, y - int(height * 0.74), max(3, half_w // 4), int(height * 0.22)), border_radius=2)
                pygame.draw.rect(self.surface, (10, 11, 15), (wheel_x, y - int(height * 0.22), max(3, half_w // 4), int(height * 0.22)), border_radius=2)
            pygame.draw.rect(self.surface, (255, 244, 174), (x - int(half_w * 0.48), y - height + 3, int(half_w * 0.32), max(2, int(height * 0.06))), border_radius=2)
            pygame.draw.rect(self.surface, (255, 244, 174), (x + int(half_w * 0.16), y - height + 3, int(half_w * 0.32), max(2, int(height * 0.06))), border_radius=2)
            label = self._font(max(12, int(14 + depth * 8))).render(identifier.upper(), True, (228, 247, 255))
            self.surface.blit(label, label.get_rect(center=(x, y - height - 8)))

    def _draw_player_car(self, driver: Any, driver_id: str | None) -> None:
        width, height = self._size
        lateral = self._driver_visual_x(driver) - self._camera_visual_x(driver)
        _, _, bottom_half, _ = self._road_metrics()
        centre_x = width / 2.0 + lateral / (self.road_width / 2.0) * bottom_half
        centre_y = height * 0.785
        heading = _number(_field(driver, "heading", "angle", "theta"), 0.0)
        steering = _number(_field(driver, "applied_steering", "steering"), 0.0)
        visual_angle = _clamp(heading * 0.30 + steering * 0.45, -0.36, 0.36)
        scale = max(72, min(width, height) * 0.18)

        def rotate(local_x: float, local_y: float) -> tuple[int, int]:
            sin_a, cos_a = math.sin(visual_angle), math.cos(visual_angle)
            return (
                int(centre_x + local_x * cos_a + local_y * sin_a),
                int(centre_y + local_x * sin_a - local_y * cos_a),
            )

        shadow = pygame.Rect(0, 0, int(scale * 1.35), int(scale * 0.72))
        shadow.center = (int(centre_x), int(centre_y + scale * 0.28))
        pygame.draw.ellipse(self.surface, (12, 16, 20), shadow)

        body = [
            rotate(-scale * 0.48, scale * 0.72),
            rotate(-scale * 0.56, scale * 0.20),
            rotate(-scale * 0.42, -scale * 0.72),
            rotate(0.0, -scale * 0.90),
            rotate(scale * 0.42, -scale * 0.72),
            rotate(scale * 0.56, scale * 0.20),
            rotate(scale * 0.48, scale * 0.72),
        ]
        pygame.draw.polygon(self.surface, (176, 31, 37), body)
        pygame.draw.polygon(self.surface, (227, 48, 51), [rotate(-scale * 0.38, scale * 0.58), rotate(-scale * 0.44, scale * 0.02), rotate(0, -scale * 0.18), rotate(scale * 0.44, scale * 0.02), rotate(scale * 0.38, scale * 0.58)])
        pygame.draw.polygon(self.surface, self.theme.f1_red_highlight, [rotate(-scale * 0.30, -scale * 0.18), rotate(-scale * 0.22, -scale * 0.55), rotate(0, -scale * 0.70), rotate(scale * 0.22, -scale * 0.55), rotate(scale * 0.30, -scale * 0.18)])
        pygame.draw.polygon(self.surface, (95, 17, 22), [rotate(-scale * 0.48, scale * 0.72), rotate(-scale * 0.56, scale * 0.20), rotate(-scale * 0.42, scale * 0.30), rotate(-scale * 0.35, scale * 0.76)])
        pygame.draw.polygon(self.surface, (122, 20, 25), [rotate(scale * 0.48, scale * 0.72), rotate(scale * 0.56, scale * 0.20), rotate(scale * 0.42, scale * 0.30), rotate(scale * 0.35, scale * 0.76)])

        windshield = [rotate(-scale * 0.26, -scale * 0.08), rotate(-scale * 0.20, -scale * 0.45), rotate(scale * 0.20, -scale * 0.45), rotate(scale * 0.26, -scale * 0.08)]
        rear_window = [rotate(-scale * 0.28, scale * 0.12), rotate(-scale * 0.22, scale * 0.39), rotate(scale * 0.22, scale * 0.39), rotate(scale * 0.28, scale * 0.12)]
        pygame.draw.polygon(self.surface, (23, 38, 58), windshield)
        pygame.draw.polygon(self.surface, (34, 55, 78), rear_window)
        pygame.draw.line(self.surface, (180, 215, 232), rotate(-scale * 0.18, -scale * 0.38), rotate(scale * 0.10, -scale * 0.38), max(2, int(scale * 0.025)))

        for wheel_x in (-scale * 0.55, scale * 0.55):
            for wheel_y in (-scale * 0.43, scale * 0.42):
                points = [
                    rotate(wheel_x, wheel_y - scale * 0.19),
                    rotate(wheel_x + math.copysign(scale * 0.13, wheel_x), wheel_y - scale * 0.13),
                    rotate(wheel_x + math.copysign(scale * 0.13, wheel_x), wheel_y + scale * 0.16),
                    rotate(wheel_x, wheel_y + scale * 0.22),
                ]
                pygame.draw.polygon(self.surface, (12, 13, 16), points)
        pygame.draw.polygon(self.surface, (255, 245, 176), [rotate(-scale * 0.28, -scale * 0.80), rotate(-scale * 0.10, -scale * 0.86), rotate(-scale * 0.13, -scale * 0.74)])
        pygame.draw.polygon(self.surface, (255, 245, 176), [rotate(scale * 0.28, -scale * 0.80), rotate(scale * 0.10, -scale * 0.86), rotate(scale * 0.13, -scale * 0.74)])
        pygame.draw.line(self.surface, (255, 98, 95), rotate(-scale * 0.25, scale * 0.72), rotate(-scale * 0.05, scale * 0.76), max(2, int(scale * 0.035)))
        pygame.draw.line(self.surface, (255, 98, 95), rotate(scale * 0.25, scale * 0.72), rotate(scale * 0.05, scale * 0.76), max(2, int(scale * 0.035)))
        if driver_id:
            label = self._font(18).render(str(driver_id).upper(), True, (250, 252, 255))
            self.surface.blit(label, label.get_rect(center=(int(centre_x), int(centre_y + scale * 1.02))))

    def _draw_cockpit(self) -> None:
        width, height = self._size
        bottom = height
        # Halo / windshield surround.
        pygame.draw.polygon(
            self.surface,
            self.theme.cockpit_dark,
            [(0, bottom), (0, int(height * 0.70)), (int(width * 0.19), int(height * 0.83)), (int(width * 0.37), bottom)],
        )
        pygame.draw.polygon(
            self.surface,
            self.theme.cockpit_dark,
            [(width, bottom), (width, int(height * 0.70)), (int(width * 0.81), int(height * 0.83)), (int(width * 0.63), bottom)],
        )
        pygame.draw.polygon(
            self.surface,
            self.theme.cockpit_mid,
            [(int(width * 0.31), bottom), (int(width * 0.43), int(height * 0.77)), (int(width * 0.57), int(height * 0.77)), (int(width * 0.69), bottom)],
        )
        pygame.draw.polygon(
            self.surface,
            self.theme.f1_red,
            [(int(width * 0.38), bottom), (int(width * 0.46), int(height * 0.745)), (int(width * 0.54), int(height * 0.745)), (int(width * 0.62), bottom)],
        )
        pygame.draw.polygon(
            self.surface,
            self.theme.f1_red_highlight,
            [(int(width * 0.47), bottom), (int(width * 0.492), int(height * 0.76)), (int(width * 0.508), int(height * 0.76)), (int(width * 0.53), bottom)],
        )
        pygame.draw.rect(self.surface, (15, 16, 20), (int(width * 0.455), int(height * 0.80), int(width * 0.09), int(height * 0.06)), border_radius=8)

    # ------------------------------------------------------------------
    # HUD / overlays

    def _draw_hud(
        self,
        simulation: Any,
        driver_id: str | None,
        driver: Any,
        mode: str,
        training_status: Mapping[str, Any] | None,
    ) -> None:
        width, _ = self._size
        elapsed = self._elapsed_time(simulation)
        speed = _number(_field(driver, "speed", "forward_speed", "velocity"), 0.0)
        passed = _field(simulation, "passed_groups", "groups_passed", "score", default=0)
        phase = _field(simulation, "phase", "curriculum_phase", default="FREE DRIVE")
        title = "HUMAN VS AI" if mode.lower() in {"versus", "vs", "human_vs_ai"} else "TURBO DODGE"

        self._label(title, (28, 24), 28, self.theme.hud_text)
        self._label(f"{speed * 3.6:05.1f} km/h", (28, 57), 42, self.theme.hud_text)
        self._label(f"TIME  {elapsed:05.1f}s", (30, 105), 22, self.theme.hud_text)
        self._label(f"PASSED  {passed}", (30, 132), 22, self.theme.hud_text)

        right_x = width - 28
        self._label(f"{str(phase).upper()}", (right_x, 26), 22, self.theme.warning, anchor="topright")
        if driver_id:
            self._label(str(driver_id).upper(), (right_x, 53), 18, (197, 222, 242), anchor="topright")
        steering = math.degrees(_number(_field(driver, "applied_steering", "steering"), 0.0))
        self._label(f"STEER {steering:+.0f}\N{DEGREE SIGN}", (right_x, 78), 18, (197, 222, 242), anchor="topright")

        if training_status:
            step = _field(training_status, "total_steps", "steps", "step", default="—")
            reward = _field(training_status, "recent_reward", "reward", "mean_reward", default="—")
            self._label(f"TRAINING  {step} steps", (right_x, 106), 18, (153, 232, 195), anchor="topright")
            self._label(f"RETURN  {reward}", (right_x, 130), 18, (153, 232, 195), anchor="topright")

        hint = "A / D or ← / → steer   P pause   Q debug   Esc menu"
        self._label(hint, (width // 2, 16), 18, (222, 231, 238), anchor="midtop")

    def _draw_debug_overlay(self, simulation: Any, focus_id: str | None, focus: Any) -> None:
        width, height = self._size
        panel = pygame.Rect(18, int(height * 0.23), min(335, int(width * 0.3)), min(300, int(height * 0.43)))
        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        overlay.fill((5, 11, 18, 218))
        self.surface.blit(overlay, panel.topleft)
        pygame.draw.rect(self.surface, (126, 170, 196), panel, 2, border_radius=8)
        self._label("TOP-DOWN DEBUG", (panel.x + 12, panel.y + 10), 18, (222, 239, 252))

        road = pygame.Rect(panel.x + 44, panel.y + 40, panel.width - 88, panel.height - 58)
        origin_y = road.bottom - 15
        half_road = self.road_width / 2.0
        focus_x = _number(_field(focus, "x", "position_x", "center_x"), 0.0)
        focus_y = _number(_field(focus, "y", "position_y", "center_y"), 0.0)
        focus_heading = _number(_field(focus, "heading", "angle", "theta"), 0.0)
        camera_x = self._camera_visual_x(focus)

        def map_top_down(world_x: float, world_y: float) -> tuple[int, int]:
            forward = world_y - focus_y
            visual_x = self._visual_road_x(world_x, world_y)
            x = road.centerx + int((visual_x - camera_x) / half_road * road.width / 2)
            y = origin_y - int(forward / self.max_view_distance * (road.height - 25))
            return x, y

        pygame.draw.rect(self.surface, (25, 66, 46), road)
        samples = np.linspace(0.0, self.max_view_distance, 26)
        left_edge = [map_top_down(-half_road, focus_y + float(distance)) for distance in samples]
        right_edge = [map_top_down(half_road, focus_y + float(distance)) for distance in samples]
        pygame.draw.polygon(self.surface, (48, 53, 61), left_edge + list(reversed(right_edge)))
        pygame.draw.lines(self.surface, (238, 239, 236), False, left_edge, 2)
        pygame.draw.lines(self.surface, (238, 239, 236), False, right_edge, 2)

        def rotate_point(x: float, y: float) -> tuple[int, int]:
            sin_h, cos_h = math.sin(focus_heading), math.cos(focus_heading)
            return (
                int(car_x + x * cos_h + y * sin_h),
                int(car_y + x * sin_h - y * cos_h),
            )

        car_x, car_y = map_top_down(focus_x, focus_y)
        car_shape = [rotate_point(-7, 12), rotate_point(-7, -13), rotate_point(7, -13), rotate_point(7, 12)]
        pygame.draw.polygon(self.surface, self.theme.f1_red, car_shape)
        pygame.draw.polygon(self.surface, (24, 35, 50), [rotate_point(-5, 2), rotate_point(-4, -8), rotate_point(4, -8), rotate_point(5, 2)])

        for obstacle in self._iter_obstacles(simulation):
            obstacle_x = _number(_field(obstacle, "x", "position_x", "center_x"), 0.0)
            obstacle_y = _number(_field(obstacle, "y", "position_y", "center_y"), 0.0)
            forward = obstacle_y - focus_y
            if 0.0 <= forward <= self.max_view_distance:
                x, y = map_top_down(obstacle_x, obstacle_y)
                pygame.draw.rect(self.surface, (247, 161, 65), (x - 4, y - 5, 8, 10))
        for identifier, driver in self._drivers(simulation).items():
            if identifier == focus_id:
                continue
            driver_x = _number(_field(driver, "x", "position_x", "center_x"), 0.0)
            driver_y = _number(_field(driver, "y", "position_y", "center_y"), 0.0)
            forward = driver_y - focus_y
            if 0.0 <= forward <= self.max_view_distance:
                x, y = map_top_down(driver_x, driver_y)
                pygame.draw.circle(self.surface, (64, 191, 238), (x, y), 6)

        alive = _field(focus, "alive", default=True)
        self._label(f"Driver: {focus_id or 'unknown'}", (panel.x + 12, panel.bottom - 38), 16, (197, 222, 242))
        self._label(f"Alive: {bool(alive)}", (panel.x + 12, panel.bottom - 20), 16, (197, 222, 242))

    def _draw_center_banner(self, title: str, subtitle: str = "") -> None:
        width, height = self._size
        box = pygame.Rect(int(width * 0.29), int(height * 0.35), int(width * 0.42), int(height * 0.18))
        shade = pygame.Surface(box.size, pygame.SRCALPHA)
        shade.fill((4, 8, 14, 224))
        self.surface.blit(shade, box.topleft)
        pygame.draw.rect(self.surface, self.theme.f1_red, box, 3, border_radius=12)
        self._label(title, (box.centerx, box.y + 23), 44, self.theme.hud_text, anchor="midtop")
        if subtitle:
            self._label(subtitle, (box.centerx, box.bottom - 24), 21, (206, 222, 234), anchor="midbottom")

    def _draw_result_banner(self, message: str) -> None:
        self._draw_center_banner(message, "R restart   Esc menu")

    def _label(
        self,
        text: str,
        position: tuple[int | float, int | float],
        size: int,
        colour: Color,
        *,
        anchor: str = "topleft",
    ) -> None:
        image = self._font(size).render(str(text), True, colour)
        rect = image.get_rect()
        setattr(rect, anchor, (int(position[0]), int(position[1])))
        self.surface.blit(image, rect)

    # ------------------------------------------------------------------
    # Simulation adaptation helpers

    def _drivers(self, simulation: Any) -> dict[str, Any]:
        raw = _field(simulation, "drivers", default={})
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items()}
        result: dict[str, Any] = {}
        for index, driver in enumerate(_iter_values(raw)):
            identifier = str(_field(driver, "id", "name", "driver_id", default=f"driver_{index}"))
            result[identifier] = driver
        return result

    def _focus_driver(self, simulation: Any, preferred: str | int | None) -> tuple[str | None, Any]:
        drivers = self._drivers(simulation)
        if isinstance(preferred, int):
            identifiers = list(drivers)
            if 0 <= preferred < len(identifiers):
                identifier = identifiers[preferred]
                return identifier, drivers[identifier]
        if preferred is not None and str(preferred) in drivers:
            identifier = str(preferred)
            return identifier, drivers[identifier]
        for candidate in ("human", "player", "ego", "agent"):
            if candidate in drivers:
                return candidate, drivers[candidate]
        if drivers:
            identifier = next(iter(drivers))
            return identifier, drivers[identifier]
        return None, None

    def _iter_obstacles(self, simulation: Any) -> Iterable[Any]:
        groups = _field(simulation, "obstacle_groups", "groups", default=None)
        source = groups if groups is not None else _field(simulation, "obstacles", "hazards", default=())
        seen: set[int] = set()
        for group in _iter_values(source):
            children = _field(group, "obstacles", "hazards", "items", "members", default=None)
            targets = _iter_values(children) if children is not None else (group,)
            for obstacle in targets:
                identity = id(obstacle)
                if identity not in seen:
                    seen.add(identity)
                    yield obstacle

    def _relative_position(self, object_: Any, driver: Any) -> tuple[float, float]:
        direct_lateral = _field(object_, "relative_x", "lateral_offset", default=None)
        direct_forward = _field(object_, "relative_y", "distance_ahead", "forward_distance", "distance", default=None)
        if direct_lateral is not None and direct_forward is not None:
            return _number(direct_lateral), _number(direct_forward)

        obj_x = _number(_field(object_, "x", "position_x", "center_x"), 0.0)
        obj_y = _number(_field(object_, "y", "position_y", "center_y"), 0.0)
        drv_y = _number(_field(driver, "y", "position_y", "center_y"), 0.0)
        forward = obj_y - drv_y
        lateral = self._visual_road_x(obj_x, obj_y) - self._camera_visual_x(driver)
        return lateral, forward

    def _elapsed_time(self, simulation: Any) -> float:
        direct = _field(simulation, "elapsed_time", "time", "episode_time", default=None)
        if direct is not None:
            return _number(direct)
        return _number(_field(simulation, "steps", "step_count"), 0.0) * _number(_field(simulation, "dt", "time_step"), 1 / 30)


class PygameRenderer:
    """Environment-facing renderer with Gymnasium-friendly output modes.

    ``TurboDodgeEnv`` can instantiate this class lazily without owning a
    Pygame window. ``rgb_array`` renders into an offscreen surface; ``human``
    creates a resizable window only when it is first requested.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1280, 720),
        caption: str = "Turbo Dodge AI",
    ) -> None:
        self.size = size
        self.caption = caption
        self._surface: pygame.Surface | None = None
        self._cockpit: CockpitRenderer | None = None
        self._display_active = False

    def render(self, simulation: Any, driver_index: int = 0, mode: str = "human") -> np.ndarray | None:
        """Render a simulation and optionally return an H×W×3 RGB frame."""

        mode = mode or "human"
        if mode not in {"human", "rgb_array"}:
            raise ValueError("mode must be 'human' or 'rgb_array'")
        self._ensure_surface(mode)
        assert self._cockpit is not None
        self._cockpit.render(
            simulation,
            focus_driver=driver_index,
            mode="versus" if len(self._cockpit._drivers(simulation)) > 1 else "solo",
        )
        if mode == "rgb_array":
            return self._cockpit.rgb_array()
        pygame.display.flip()
        return None

    def close(self) -> None:
        """Release display resources while allowing Pygame to be re-used later."""

        if self._display_active:
            pygame.display.quit()
        self._display_active = False
        self._surface = None
        self._cockpit = None

    def _ensure_surface(self, mode: str) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if mode == "human":
            if not pygame.display.get_init():
                pygame.display.init()
            existing = pygame.display.get_surface() if self._display_active else None
            if existing is None:
                surface = pygame.display.set_mode(self.size, pygame.RESIZABLE)
                pygame.display.set_caption(self.caption)
                self._display_active = True
            else:
                surface = existing
        elif self._surface is not None and not self._display_active:
            surface = self._surface
        else:
            surface = pygame.Surface(self.size)
            self._display_active = False
        self._surface = surface
        if self._cockpit is None:
            self._cockpit = CockpitRenderer(surface)
        else:
            self._cockpit.set_surface(surface)


__all__ = ["CockpitRenderer", "PygameRenderer", "RenderTheme"]
