import pygame
import math
import shotmodel




def clickonball():
    global shooting, shootorigin

    shooting=True
    shootorigin = pygame.mouse.get_pos()
       
def releaseonball():
    global shooting, ball_active

    shooting=False
    shoottarget = pygame.mouse.get_pos()
    ball_vel[0] += (shoottarget[0] - shootorigin[0])*2
    ball_vel[1] += (shoottarget[1] - shootorigin[1])*2
    print((shoottarget[0] - shootorigin[0])*2 , (shoottarget[1] - shootorigin[1])*2)
    ball_active = True


def get_state():
    return ball_pos, ball_vel, rim_rect, None










def bounce():
    width, height = screen.get_size()
    hit_bottom = False

    if ball_pos[1] + rad >= height:
        ball_pos[1] = height - rad
        ball_vel[1] = -abs(ball_vel[1]) * bounciness
        hit_bottom = True
        if abs(ball_vel[1]) < 10:
            ball_pos[1]= screen.get_size()[1] - rad-1
            ball_vel[1] = 0

    if ball_pos[0] + rad >= width:
        ball_pos[0] = width - rad
        ball_vel[0] = -abs(ball_vel[0]) * bounciness

    if ball_pos[0] - rad <= 0:
        ball_pos[0] = rad
        ball_vel[0] = abs(ball_vel[0])*bounciness

    return hit_bottom


def bounce_off_rim_post(post_pos):
    dx = ball_pos[0] - post_pos[0]
    dy = ball_pos[1] - post_pos[1]
    distance = math.hypot(dx, dy)
    min_distance = rad + rim_radius

    if distance == 0 or distance > min_distance:
        return

    normal_x = dx / distance
    normal_y = dy / distance
    speed_toward_post = ball_vel[0] * normal_x + ball_vel[1] * normal_y

    if speed_toward_post < 0:
        ball_vel[0] -= (1 + bounciness) * speed_toward_post * normal_x
        ball_vel[1] -= (1 + bounciness) * speed_toward_post * normal_y

    ball_pos[0] = post_pos[0] + normal_x * min_distance
    ball_pos[1] = post_pos[1] + normal_y * min_distance


def bounce_off_rim():
    bounce_off_rim_post(left_rim_post)
    bounce_off_rim_post(right_rim_post)


def reset_ball():
    global ball_pos, ball_vel, scored_this_shot, ball_active, next_model_shot_time

    ball_pos = start_pos.copy()
    ball_vel = start_vel.copy()
    scored_this_shot = False
    ball_active = False
    next_model_shot_time = pygame.time.get_ticks() + model_shot_delay

show=True
def change():
    global show
    if pygame.key.get_pressed()[pygame.K_a]:
        show=False

def check_score(previous_pos):
    global score, scored_this_shot

    rim_y = rim_rect.centery
    opening_left = rim_rect.left + rad
    opening_right = rim_rect.right - rad
    crossed_down = previous_pos[1] < rim_y <= ball_pos[1] and ball_vel[1] > 0
    inside_opening = opening_left < ball_pos[0] < opening_right

    if crossed_down and inside_opening and not scored_this_shot:
        score += 1
        scored_this_shot = True
        shotmodel.learn_from_result(True)
        reset_ball()

    if ball_pos[1] < rim_rect.top - rad:
        scored_this_shot = False


def check_miss(hit_bottom):
    if hit_bottom and shotmodel.has_active_shot():
        shotmodel.learn_from_result(False)
        reset_ball()


def mouseonball():
    mouse_pos = pygame.mouse.get_pos()
    a=mouse_pos[0] - ball_pos[0]
    b=mouse_pos[1] - ball_pos[1]
    c2=a**2 + b**2
    if c2<=rad**2:
        return True
    return False


def maybe_model_shoot():
    if auto_model and not ball_active and not shooting and not shotmodel.has_active_shot():
        if pygame.time.get_ticks() >= next_model_shot_time:
            shotmodel.shoot_ball(clickonball, releaseonball, reset_ball, get_state)


def toggle_play_mode():
    global auto_model

    auto_model = not auto_model
    reset_ball()


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1300, 800))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)
running = True
start_pos = [25,500]
start_vel=[0,0]
ball_pos = start_pos.copy()
ball_vel = start_vel.copy()
ball_active = False
delta=0.0
gravity=700
rad=25
bounciness=1
score=0
scored_this_shot=False
rim_radius=8
rim_rect=pygame.Rect(
                screen.get_width()-rad*5-5,
                screen.get_height()//2-50,
                rad*5,
                rad
            )
left_rim_post=(rim_rect.left, rim_rect.centery)
right_rim_post=(rim_rect.right, rim_rect.centery)

shootorigin=[0,0]
shooting=False
auto_model=True
model_shot_delay=50
next_model_shot_time=pygame.time.get_ticks() + model_shot_delay
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and mouseonball():
            clickonball()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and shooting:
            releaseonball()
        if event.type==pygame.KEYDOWN and event.key==pygame.K_r:
            reset_ball()
        if event.type==pygame.KEYDOWN and event.key==pygame.K_SPACE and auto_model:
            shotmodel.shoot_ball(clickonball, releaseonball, reset_ball, get_state)
        if event.type==pygame.KEYDOWN and event.key==pygame.K_s:
            toggle_play_mode()
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    pygame.draw.ellipse(screen, (0,0,0),
                        rim_rect,
                        8)
    pygame.draw.circle(screen, "blue", ball_pos, rad)
    score_text = font.render(f"Score: {score}", True, "black")
    screen.blit(score_text, (20, 20))
    attempts, makes, percentage = shotmodel.get_stats()
    percent_text = font.render(f"Model: {makes}/{attempts}  {percentage:.1f}%", True, "black")
    screen.blit(percent_text, (20, 65))
    mode_text = font.render("Mode: Model" if auto_model else "Mode: Player", True, "black")
    screen.blit(mode_text, (20, 110))


    maybe_model_shoot()

    if ball_active:
        ball_vel[1] += gravity/100
        previous_pos = ball_pos.copy()
        ball_pos[0] += ball_vel[0]/100
        ball_pos[1] += ball_vel[1]//100
        hit_bottom = bounce()
        bounce_off_rim()
        check_score(previous_pos)
        check_miss(hit_bottom)
    if show:
        pygame.display.flip()


        delta = clock.tick(100000000000000000000000) / 1000

pygame.quit()
