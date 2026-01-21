import pygame
import math
import time
from text import drawText
from fontDict import fonts
import controlPanel as cp
from droneManager import DroneHandler

pygame.init()

# ---------------- Setup
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 1
screen_width = int(screen.get_width() / scaleDownFactor)
screen_height = int(screen.get_height() / scaleDownFactor)

# Layers
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT.set_alpha(100)
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()

# Adaptive Fonts
montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratBoldAdaptive = fonts[f"bold{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]

# State
timer = 0
shake = [0, 0]
toggle = True
drone_handler = DroneHandler()

# ---------------- Main Loop
last_time = time.time()
running = True

while running:
    # Input
    mx, my = pygame.mouse.get_pos()
    mx, my = mx / scaleDownFactor, my / scaleDownFactor

    # Delta Time
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0.1: dt = 0.1

    # --- Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
            # Regenerate Map
            if event.key == pygame.K_r:
                drone_handler.generate_map()

    # --- Update ---
    drone_handler.update(dt)

    # --- Rendering ---
    screen.fill(cp.Endesga.my_blue)
    screen2.fill(cp.Endesga.my_blue)
    screenT.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))

    # Calculate Centering Offset
    map_pixel_width = cp.ARENA_WIDTH_FT * cp.PX_PER_FOOT
    map_pixel_height = cp.ARENA_HEIGHT_FT * cp.PX_PER_FOOT

    offset_x = (screen_width - map_pixel_width) / 2
    offset_y = (screen_height - map_pixel_height) / 2

    # Draw Simulation
    drone_handler.draw(screen2, (offset_x, offset_y))

    # UI Overlay
    if toggle:
        items = {
            "FPS": round(clock.get_fps()),
            "Time": f"{drone_handler.elapsed:.1f}s",
            "Detected": f"{len(drone_handler.mines_detected)} / {len(drone_handler.mines_truth)}",
            "Controls": "[R] Reset  [ESC] Quit"
        }
        for i, label in enumerate(items.keys()):
            full_str = f"{label}: {items[label]}"
            drawText(screenUI, cp.Endesga.debug_red, montserratRegularAdaptive, 5,
                     screen_height - (30 + 25 * i), full_str, cp.Endesga.black, 1, antiAliasing=False)

        # Cursor
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, cp.Endesga.black, (mx + 1, my + 1), 2, 1)
        pygame.draw.circle(screenUI, cp.Endesga.white, (mx, my), 2, 1)

    # Composite
    screen.blit(pygame.transform.scale(screen2, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)),
                (0, 0))
    screen.blit(pygame.transform.scale(screenT, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)),
                (0, 0))
    screen.blit(pygame.transform.scale(screenUI, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)),
                (0, 0))

    pygame.display.update()
    clock.tick(fps)