import random
import sys

import pygame


pygame.init()

# Window settings
WIDTH = 480
HEIGHT = 700
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

# Colors
SKY_BLUE = (120, 200, 255)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
YELLOW = (255, 220, 40)
ORANGE = (255, 140, 20)
GREEN = (60, 190, 80)
DARK_GREEN = (20, 130, 50)
GROUND_COLOR = (220, 190, 100)

# Fonts
score_font = pygame.font.SysFont("arial", 48, bold=True)
message_font = pygame.font.SysFont("arial", 30, bold=True)
small_font = pygame.font.SysFont("arial", 22)

# Game settings
GRAVITY = 0.45
FLAP_STRENGTH = -8.5
PIPE_SPEED = 4
PIPE_WIDTH = 80
PIPE_GAP = 180
GROUND_HEIGHT = 80
PIPE_INTERVAL = 1500


class Bird:
    def __init__(self):
        self.x = 110
        self.y = HEIGHT // 2
        self.radius = 18
        self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self):
        # Body
        pygame.draw.circle(
            screen,
            YELLOW,
            (int(self.x), int(self.y)),
            self.radius
        )

        # Wing
        pygame.draw.ellipse(
            screen,
            ORANGE,
            (
                int(self.x - 14),
                int(self.y),
                22,
                14
            )
        )

        # Eye
        pygame.draw.circle(
            screen,
            WHITE,
            (int(self.x + 8), int(self.y - 7)),
            7
        )
        pygame.draw.circle(
            screen,
            BLACK,
            (int(self.x + 10), int(self.y - 7)),
            3
        )

        # Beak
        pygame.draw.polygon(
            screen,
            ORANGE,
            [
                (int(self.x + 17), int(self.y - 2)),
                (int(self.x + 32), int(self.y + 4)),
                (int(self.x + 17), int(self.y + 8)),
            ]
        )

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2
        )


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(160, HEIGHT - GROUND_HEIGHT - 160)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self):
        top_height = self.gap_y - PIPE_GAP // 2
        bottom_y = self.gap_y + PIPE_GAP // 2
        bottom_height = HEIGHT - GROUND_HEIGHT - bottom_y

        # Main pipes
        pygame.draw.rect(
            screen,
            GREEN,
            (self.x, 0, PIPE_WIDTH, top_height)
        )
        pygame.draw.rect(
            screen,
            GREEN,
            (self.x, bottom_y, PIPE_WIDTH, bottom_height)
        )

        # Dark edges
        pygame.draw.rect(
            screen,
            DARK_GREEN,
            (self.x, 0, 8, top_height)
        )
        pygame.draw.rect(
            screen,
            DARK_GREEN,
            (self.x, bottom_y, 8, bottom_height)
        )

        # Pipe ends
        pygame.draw.rect(
            screen,
            GREEN,
            (self.x - 6, top_height - 28, PIPE_WIDTH + 12, 28)
        )
        pygame.draw.rect(
            screen,
            GREEN,
            (self.x - 6, bottom_y, PIPE_WIDTH + 12, 28)
        )

        pygame.draw.rect(
            screen,
            DARK_GREEN,
            (self.x - 6, top_height - 28, 8, 28)
        )
        pygame.draw.rect(
            screen,
            DARK_GREEN,
            (self.x - 6, bottom_y, 8, 28)
        )

    def get_rects(self):
        top_height = self.gap_y - PIPE_GAP // 2
        bottom_y = self.gap_y + PIPE_GAP // 2

        top_rect = pygame.Rect(
            int(self.x),
            0,
            PIPE_WIDTH,
            top_height
        )

        bottom_rect = pygame.Rect(
            int(self.x),
            bottom_y,
            PIPE_WIDTH,
            HEIGHT - GROUND_HEIGHT - bottom_y
        )

        return top_rect, bottom_rect

    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0


def draw_background():
    screen.fill(SKY_BLUE)

    # Clouds
    pygame.draw.circle(screen, WHITE, (75, 110), 28)
    pygame.draw.circle(screen, WHITE, (105, 100), 35)
    pygame.draw.circle(screen, WHITE, (140, 115), 25)

    pygame.draw.circle(screen, WHITE, (330, 180), 24)
    pygame.draw.circle(screen, WHITE, (355, 165), 32)
    pygame.draw.circle(screen, WHITE, (390, 185), 22)

    # Ground
    pygame.draw.rect(
        screen,
        GROUND_COLOR,
        (0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT)
    )
    pygame.draw.rect(
        screen,
        GREEN,
        (0, HEIGHT - GROUND_HEIGHT, WIDTH, 15)
    )


def draw_centered_text(text, font, color, y):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(WIDTH // 2, y))
    screen.blit(surface, rect)


def reset_game():
    bird = Bird()
    pipes = []
    score = 0
    game_started = False
    game_over = False
    last_pipe_time = pygame.time.get_ticks()

    return bird, pipes, score, game_started, game_over, last_pipe_time


bird, pipes, score, game_started, game_over, last_pipe_time = reset_game()

running = True

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_UP):
                if game_over:
                    (
                        bird,
                        pipes,
                        score,
                        game_started,
                        game_over,
                        last_pipe_time
                    ) = reset_game()
                else:
                    game_started = True
                    bird.flap()

            if event.key == pygame.K_r and game_over:
                (
                    bird,
                    pipes,
                    score,
                    game_started,
                    game_over,
                    last_pipe_time
                ) = reset_game()

            if event.key == pygame.K_ESCAPE:
                running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if game_over:
                (
                    bird,
                    pipes,
                    score,
                    game_started,
                    game_over,
                    last_pipe_time
                ) = reset_game()
            else:
                game_started = True
                bird.flap()

    if game_started and not game_over:
        bird.update()

        current_time = pygame.time.get_ticks()

        if current_time - last_pipe_time >= PIPE_INTERVAL:
            pipes.append(Pipe(WIDTH + 20))
            last_pipe_time = current_time

        for pipe in pipes:
            pipe.update()

            if not pipe.passed and pipe.x + PIPE_WIDTH < bird.x:
                pipe.passed = True
                score += 1

            top_rect, bottom_rect = pipe.get_rects()

            if bird.get_rect().colliderect(top_rect):
                game_over = True

            if bird.get_rect().colliderect(bottom_rect):
                game_over = True

        pipes = [pipe for pipe in pipes if not pipe.is_off_screen()]

        # Ceiling and ground collision
        if bird.y - bird.radius <= 0:
            game_over = True

        if bird.y + bird.radius >= HEIGHT - GROUND_HEIGHT:
            game_over = True

    draw_background()

    for pipe in pipes:
        pipe.draw()

    bird.draw()

    score_surface = score_font.render(str(score), True, WHITE)
    score_shadow = score_font.render(str(score), True, BLACK)

    score_rect = score_surface.get_rect(center=(WIDTH // 2, 70))
    shadow_rect = score_shadow.get_rect(center=(WIDTH // 2 + 3, 73))

    screen.blit(score_shadow, shadow_rect)
    screen.blit(score_surface, score_rect)

    if not game_started and not game_over:
        draw_centered_text(
            "Flappy Bird",
            message_font,
            BLACK,
            250
        )
        draw_centered_text(
            "Press SPACE or click to flap",
            small_font,
            BLACK,
            300
        )

    if game_over:
        draw_centered_text(
            "Game Over",
            message_font,
            BLACK,
            250
        )
        draw_centered_text(
            f"Score: {score}",
            small_font,
            BLACK,
            295
        )
        draw_centered_text(
            "Press SPACE, R, or click to restart",
            small_font,
            BLACK,
            335
        )

    pygame.display.flip()

pygame.quit()
sys.exit()