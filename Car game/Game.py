"""Pseudo-3D rear-view road-dodging game.

Steer with A/D or the left/right arrow keys.  Press R after a crash to play
again. The car sprite is a transparent PNG beside this file.
"""

from pathlib import Path
import math
import random

import pygame


WIDTH, HEIGHT = 1280, 720
FPS = 60
HORIZON_Y = 132
ROAD_TOP_WIDTH = 150
ROAD_BOTTOM_WIDTH = 1120
LANES = 3
ROAD_SEGMENTS = 42
PLAYER_WIDTH = 245
MAX_DRIVE_SPEED = 0.95
THROTTLE_ACCELERATION = 0.52
BRAKE_DECELERATION = 1.35
COAST_DECELERATION = 0.085

SKY_TOP = (60, 150, 219)
SKY_HORIZON = (189, 222, 243)
GRASS = (34, 123, 60)
ROAD = (50, 54, 63)
ROAD_ALT = (55, 59, 68)
ROAD_EDGE = (235, 238, 222)
LANE_MARKING = (255, 240, 178)
HUD_TEXT = (247, 249, 255)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rear View Road Dodge - 3D")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 38)
small_font = pygame.font.Font(None, 27)
large_font = pygame.font.Font(None, 76)


def load_player_car() -> pygame.Surface:
    """Load the supplied rear-view car art with its transparent background."""
    image_path = Path(__file__).with_name("car_transparent.png")
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find the transparent car image: {image_path}")

    source = pygame.image.load(image_path).convert_alpha()
    height = round(PLAYER_WIDTH * source.get_height() / source.get_width())
    return pygame.transform.smoothscale(source, (PLAYER_WIDTH, height))


player_car = load_player_car()
PLAYER_HEIGHT = player_car.get_height()


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def screen_y(depth: float) -> float:
    """Project a world depth (0=farthest, 1=at the camera) onto the screen."""
    return lerp(HORIZON_Y, HEIGHT + 85, depth ** 1.55)


def road_curve(depth: float, distance: float) -> float:
    """Make the far road bend while the road stays centred under the camera."""
    current_bend = math.sin(distance * 0.010)
    distant_bend = math.sin(distance * 0.010 + (1 - depth) * 2.2)
    return (distant_bend - current_bend) * (1 - depth) * 150


def road_section(depth: float, distance: float) -> tuple[float, float, float, float]:
    """Return left edge, right edge, centre x, and y for a road depth."""
    depth = clamp(depth, 0.0, 1.0)
    centre = WIDTH / 2 + road_curve(depth, distance)
    width = lerp(ROAD_TOP_WIDTH, ROAD_BOTTOM_WIDTH, depth ** 0.92)
    return centre - width / 2, centre + width / 2, centre, screen_y(depth)


def lane_x(depth: float, lane: float, distance: float) -> float:
    left, right, _, _ = road_section(depth, distance)
    return lerp(left, right, (lane + 0.5) / LANES)


def shade(colour: tuple[int, int, int], multiplier: float) -> tuple[int, int, int]:
    return tuple(round(channel * multiplier) for channel in colour)


def draw_sky() -> None:
    """Paint a simple sky and distant hills behind the 3D road."""
    for index in range(HORIZON_Y):
        colour = tuple(
            round(lerp(SKY_TOP[channel], SKY_HORIZON[channel], index / HORIZON_Y))
            for channel in range(3)
        )
        pygame.draw.line(screen, colour, (0, index), (WIDTH, index))

    screen.fill(GRASS, pygame.Rect(0, HORIZON_Y, WIDTH, HEIGHT - HORIZON_Y))

    hill_points = [(0, HORIZON_Y + 24)]
    for x in range(0, WIDTH + 80, 80):
        hill_y = HORIZON_Y + 19 + math.sin(x * 0.011) * 18 + math.sin(x * 0.029) * 9
        hill_points.append((x, hill_y))
    hill_points.extend([(WIDTH, HORIZON_Y + 75), (0, HORIZON_Y + 75)])
    pygame.draw.polygon(screen, (33, 101, 76), hill_points)


def draw_lane_markers(distance: float) -> None:
    """Draw fixed lane dashes so the player car, not the road, leads motion."""
    for lane in range(1, LANES):
        for index in range(1, 20):
            far_depth = index / 20
            near_depth = far_depth + 0.026
            far_depth = clamp(far_depth, 0.0, 1.0)
            near_depth = clamp(near_depth, 0.0, 1.0)
            far_left, far_right, _, far_y = road_section(far_depth, distance)
            near_left, near_right, _, near_y = road_section(near_depth, distance)
            far_x = lerp(far_left, far_right, lane / LANES)
            near_x = lerp(near_left, near_right, lane / LANES)
            far_width = max(1, (far_right - far_left) * 0.010)
            near_width = max(2, (near_right - near_left) * 0.010)
            pygame.draw.polygon(
                screen,
                LANE_MARKING,
                [
                    (far_x - far_width, far_y),
                    (far_x + far_width, far_y),
                    (near_x + near_width, near_y),
                    (near_x - near_width, near_y),
                ],
            )


def draw_road(distance: float) -> None:
    """Draw the road in many depth segments, producing a curved 3D effect."""
    draw_sky()

    for index in range(ROAD_SEGMENTS):
        far_depth = index / ROAD_SEGMENTS
        near_depth = (index + 1) / ROAD_SEGMENTS
        far_left, far_right, _, far_y = road_section(far_depth, distance)
        near_left, near_right, _, near_y = road_section(near_depth, distance)
        stripe = index % 2
        road_colour = ROAD_ALT if stripe else ROAD
        pygame.draw.polygon(
            screen,
            road_colour,
            [(far_left, far_y), (far_right, far_y), (near_right, near_y), (near_left, near_y)],
        )

        edge_width = max(1, round(2 + near_depth * 7))
        pygame.draw.line(screen, ROAD_EDGE, (far_left, far_y), (near_left, near_y), edge_width)
        pygame.draw.line(screen, ROAD_EDGE, (far_right, far_y), (near_right, near_y), edge_width)

    draw_lane_markers(distance)


def draw_roadside_objects(distance: float) -> None:
    """Place trees and lamp posts on the same depth scale as the road."""
    for index in range(1, 14):
        depth = index / 14
        if not 0.02 < depth < 1.0:
            continue
        left, right, _, y = road_section(depth, distance)
        size = round(7 + depth * 34)
        for side, road_edge in ((-1, left), (1, right)):
            x = road_edge + side * size * 1.7
            trunk = pygame.Rect(round(x - size * 0.13), round(y - size * 1.1), max(2, size // 4), round(size * 1.25))
            pygame.draw.rect(screen, (92, 61, 37), trunk)
            pygame.draw.circle(screen, (18, 83, 42), (round(x), round(y - size * 1.15)), size)
            pygame.draw.circle(screen, (49, 151, 69), (round(x - size * 0.28), round(y - size * 1.4)), round(size * 0.7))

            if index % 3 == 0:
                pole_x = road_edge + side * size * 3.3
                pygame.draw.line(screen, (91, 98, 102), (pole_x, y), (pole_x, y - size * 3), max(1, size // 7))
                pygame.draw.circle(screen, (255, 236, 147), (round(pole_x), round(y - size * 3)), max(2, size // 5))


def obstacle_position(obstacle: dict, distance: float) -> tuple[float, float, int, int]:
    """Project a distant traffic car from its lane/depth into screen space."""
    depth = obstacle["depth"]
    _, _, _, y = road_section(depth, distance)
    x = lane_x(depth, obstacle["lane"], distance)
    width = round(18 + 120 * depth ** 1.35)
    height = round(width * 0.68)
    return x, y, width, height


def draw_traffic_car(obstacle: dict, distance: float) -> pygame.Rect:
    """Draw a depth-shaded rear-view traffic car and return its fair hitbox."""
    x, y, width, height = obstacle_position(obstacle, distance)
    body = pygame.Rect(round(x - width / 2), round(y - height), width, height)
    depth = obstacle["depth"]
    colour = shade(obstacle["colour"], 0.55 + depth * 0.45)

    pygame.draw.ellipse(screen, (31, 37, 34), (body.left - width * 0.1, body.bottom - height * 0.12, width * 1.2, max(2, height * 0.22)))
    tire_width = max(3, width // 7)
    tire_height = max(4, height // 3)
    pygame.draw.rect(screen, (15, 18, 22), (body.left, body.bottom - tire_height, tire_width, tire_height), border_radius=3)
    pygame.draw.rect(screen, (15, 18, 22), (body.right - tire_width, body.bottom - tire_height, tire_width, tire_height), border_radius=3)
    pygame.draw.rect(screen, colour, body, border_radius=max(3, width // 10))

    window = pygame.Rect(body.left + width // 4, body.top + height // 7, width // 2, max(3, height // 3))
    pygame.draw.rect(screen, (22, 33, 47), window, border_radius=max(2, width // 16))
    light_width = max(2, width // 6)
    light_height = max(2, height // 8)
    pygame.draw.rect(screen, (255, 89, 71), (body.left + width // 9, body.bottom - height // 3, light_width, light_height), border_radius=2)
    pygame.draw.rect(screen, (255, 89, 71), (body.right - width // 9 - light_width, body.bottom - height // 3, light_width, light_height), border_radius=2)

    return body.inflate(-max(4, width // 5), -max(4, height // 7))


def reset_game() -> tuple[float, float, list[dict], float, bool, float, float, float]:
    """Start stationary so the player controls every bit of forward speed."""
    return WIDTH / 2, 0.0, [], 0.8, False, 0.0, 0.0, 0.0


player_x, distance, obstacles, spawn_timer, game_over, steering_visual, current_speed, forward_visual = reset_game()
running = True

while running:
    delta = clock.tick(FPS) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif game_over and event.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                player_x, distance, obstacles, spawn_timer, game_over, steering_visual, current_speed, forward_visual = reset_game()

    keys = pygame.key.get_pressed()
    if not game_over:
        steering = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
        steering_visual = lerp(steering_visual, steering, min(1.0, delta * 10))
        player_x += steering * 620 * delta
        road_left, road_right, _, _ = road_section(0.96, distance)
        player_x = clamp(player_x, road_left + PLAYER_WIDTH * 0.42, road_right - PLAYER_WIDTH * 0.42)

        throttle = keys[pygame.K_w]
        braking = keys[pygame.K_s]
        if braking:
            current_speed = max(0.0, current_speed - BRAKE_DECELERATION * delta)
        elif throttle:
            current_speed = min(MAX_DRIVE_SPEED, current_speed + THROTTLE_ACCELERATION * delta)
        else:
            current_speed = max(0.0, current_speed - COAST_DECELERATION * delta)

        # The car physically drives farther into the scene as it accelerates.
        # This keeps the visual focus on the vehicle rather than scrolling the
        # road surface toward the camera.
        forward_target = current_speed / MAX_DRIVE_SPEED
        forward_visual = lerp(forward_visual, forward_target, min(1.0, delta * 3.8))

        distance += 100 * current_speed * delta
        spawn_timer -= current_speed * delta
        if spawn_timer <= 0:
            obstacles.append(
                {
                    "lane": random.randrange(LANES),
                    "depth": 0.018,
                    "colour": random.choice([(50, 124, 222), (233, 171, 38), (174, 66, 73), (101, 165, 110)]),
                }
            )
            spawn_timer = max(0.42, random.uniform(0.72, 1.12) - min(distance * 0.00005, 0.30))

        for obstacle in obstacles:
            obstacle["depth"] += current_speed * delta
        obstacles = [obstacle for obstacle in obstacles if obstacle["depth"] < 1.10]
    else:
        steering_visual = lerp(steering_visual, 0.0, min(1.0, delta * 5))
        forward_visual = lerp(forward_visual, 0.0, min(1.0, delta * 3))

    draw_road(distance)
    draw_roadside_objects(distance)
    obstacle_hitboxes = [draw_traffic_car(obstacle, distance) for obstacle in sorted(obstacles, key=lambda item: item["depth"])]

    car_scale = 1.0 - forward_visual * 0.19
    tilted_car = pygame.transform.rotozoom(player_car, -steering_visual * 5, car_scale)
    car_forward_offset = round(forward_visual * 118)
    player_rect = tilted_car.get_rect(center=(round(player_x), HEIGHT - tilted_car.get_height() // 2 - 23 - car_forward_offset))
    player_hitbox = pygame.Rect(
        player_rect.left + round(player_rect.width * 0.22),
        player_rect.top + round(player_rect.height * 0.15),
        round(player_rect.width * 0.56),
        round(player_rect.height * 0.75),
    )
    screen.blit(tilted_car, player_rect)

    if not game_over and any(player_hitbox.colliderect(hitbox) for hitbox in obstacle_hitboxes):
        game_over = True

    speed_display = round(current_speed / MAX_DRIVE_SPEED * 185)
    screen.blit(font.render(f"Distance: {int(distance):04d} m", True, HUD_TEXT), (30, 25))
    screen.blit(font.render(f"Speed: {speed_display} km/h", True, HUD_TEXT), (30, 62))
    screen.blit(small_font.render("W: drive forward   S: brake   Steer: A / D or arrow keys   Esc: quit", True, HUD_TEXT), (30, HEIGHT - 34))

    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 12, 18, 175))
        screen.blit(overlay, (0, 0))
        title = large_font.render("CRASH!", True, (255, 111, 82))
        prompt = font.render("Press R, Enter, or Space to drive again", True, HUD_TEXT)
        screen.blit(title, title.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 35)))
        screen.blit(prompt, prompt.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 35)))

    pygame.display.flip()

pygame.quit()
