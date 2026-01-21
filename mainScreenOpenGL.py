import pygame
import moderngl
import struct
import math
import time
import sys
import array

# Import Framework Utilities (Assumes these files exist in the directory)
try:
    from text import drawText
    from fontDict import fonts
    from calcs import linearGradient, distance, normalize
except ImportError:
    print("Error: Could not import 'text' or 'fontDict'. Ensure Framework files are in the same directory.")
    sys.exit(1)


# --- SHADER LOADING UTILITY ---
def load_shader(ctx, vert_path, frag_path):
    try:
        with open(vert_path, 'r') as f:
            vert_src = f.read()
        with open(frag_path, 'r') as f:
            frag_src = f.read()
        return ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    except FileNotFoundError:
        print(f"Error: Shader files not found at {vert_path} or {frag_path}")
        sys.exit(1)


# --- COLORS (From Framework) ---
class Endesga:
    maroon_red = [87, 28, 39]
    lighter_maroon_red = [127, 36, 51]
    dark_green = [9, 26, 23]
    light_brown = [191, 111, 74]
    black = [19, 19, 19]
    grey_blue = [66, 76, 110]
    cream = [237, 171, 80]
    white = [255, 255, 255]
    greyL = [200, 200, 200]
    grey = [150, 150, 150]
    greyD = [100, 100, 100]
    greyVD = [50, 50, 50]
    very_light_blue = [199, 207, 221]
    my_blue = [32, 36, 46]
    darker_my_blue = [16, 18, 21]
    debug_red = [255, 96, 141]


# --- INIT ---
pygame.init()

# 1. CONFIGURE OPENGL ATTRIBUTES (Fixes the Version 0 Error)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# 2. Setup Window
# Note: On Retina displays, use pygame.HIGHDPI if needed, though raw pixel scaling handles it here
screen = pygame.display.set_mode((0, 0), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
info = pygame.display.Info()
WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h

# 3. Create ModernGL Context
# require=330 forces check for OpenGL 3.3
try:
    ctx = moderngl.create_context(require=330)
except Exception as e:
    print(f"Error creating ModernGL context: {e}")
    sys.exit(1)

ctx.enable(moderngl.BLEND)

# 4. Game Logic Variables
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 3

# Virtual Resolution (Low Res)
VIRTUAL_W = int(WINDOW_WIDTH / scaleDownFactor)
VIRTUAL_H = int(WINDOW_HEIGHT / scaleDownFactor)

# Pygame Surfaces (CPU side)
screen2 = pygame.Surface((VIRTUAL_W, VIRTUAL_H)).convert_alpha()
screenUI = pygame.Surface((VIRTUAL_W, VIRTUAL_H)).convert_alpha()

# 5. OpenGL Textures
game_texture = ctx.texture((VIRTUAL_W, VIRTUAL_H), 4)
game_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
game_texture.swizzle = 'BGRA'

# 6. Geometry (Full Screen Quad)
# Format: x, y, u, v
quad_buffer = ctx.buffer(data=array.array('f', [
    # Position (x,y)   # UV Coords (u,v)
    -1.0, 1.0, 0.0, 0.0,  # Top Left
    -1.0, -1.0, 0.0, 1.0,  # Bottom Left
    1.0, 1.0, 1.0, 0.0,  # Top Right
    1.0, -1.0, 1.0, 1.0,  # Bottom Right
]))

# 7. Load Shaders
prog = load_shader(ctx, 'shaders/basic.vert', 'shaders/basic.frag')

vao = ctx.vertex_array(prog, [
    (quad_buffer, '2f 2f', 'in_vert', 'in_text')
])

# Font Setup
try:
    montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
except KeyError:
    print("Warning: Requested font size not found in fontDict. Using default SysFont.")
    montserratRegularAdaptive = pygame.font.SysFont("Arial", 20)

# --- GRID CONFIGURATION ---
LOGICAL_W, LOGICAL_H = 160, 100
TILE_SIZE = 2  # Width/Height of each tile in pixels
TILE_GAP = 0  # Gap between tiles in pixels
EFFECTIVE_STEP = TILE_SIZE + TILE_GAP

# Calculate physical pixel dimensions based on tiles + gaps
PHYSICAL_W = LOGICAL_W * EFFECTIVE_STEP
PHYSICAL_H = LOGICAL_H * EFFECTIVE_STEP

# Create a specific surface for the heatmap
grid_surface = pygame.Surface((PHYSICAL_W, PHYSICAL_H))

# Colors for Heatmap: Red (Near) -> Green (Far)
HEAT_PALETTE = [[150, 50, 80], [80, 150, 50]]

# Pre-calculate centering offsets for placing grid on screen2
grid_x_offset = (VIRTUAL_W - PHYSICAL_W) // 2
grid_y_offset = (VIRTUAL_H - PHYSICAL_H) // 2

# Loop Variables
timer = 0
shake = [0, 0]
oscillating_random_thing = 0
toggle = True
click = False
last_time = time.time()
running = True

# --- MAIN LOOP ---
while running:
    # A. Time Management
    dt = time.time() - last_time
    dt *= fps
    last_time = time.time()

    # B. Input Handling
    mx, my = pygame.mouse.get_pos()
    vmx, vmy = mx / scaleDownFactor, my / scaleDownFactor

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle

    # C. Logic
    timer -= 1 * dt
    oscillating_random_thing += math.pi / fps * dt

    # D. Rendering to Pygame Surfaces
    screen2.fill(Endesga.darker_my_blue)
    screenUI.fill((0, 0, 0, 0))

    # 1. Input: Get mouse relative to the grid surface
    rel_mx = vmx - grid_x_offset
    rel_my = vmy - grid_y_offset

    # Convert pixel mouse coordinate to logical tile coordinate for heatmap math
    mouse_tile_x = rel_mx // EFFECTIVE_STEP
    mouse_tile_y = rel_my // EFFECTIVE_STEP

    # 2. Logic: Heat Map Calculation
    # Clear surface to black (or gap color) before drawing tiles
    grid_surface.fill((0, 0, 0))

    # We use PixelArray for faster direct pixel access
    px_array = pygame.PixelArray(grid_surface)

    # Max distance possible (diagonal of logical grid)
    max_dist = math.hypot(LOGICAL_W, LOGICAL_H)

    for x in range(LOGICAL_W):
        for y in range(LOGICAL_H):
            # Calculate logical distance from mouse tile to current tile
            dist = math.hypot(x - mouse_tile_x, y - mouse_tile_y)

            # Normalize 0.0 to 1.0 based on map size
            norm_dist = normalize(dist, 0, max_dist, True)

            # Use calcs.linearGradient
            col_list = linearGradient(HEAT_PALETTE, norm_dist)
            col_tuple = (col_list[0], col_list[1], col_list[2])

            # Calculate start pixel positions
            sx = x * EFFECTIVE_STEP
            sy = y * EFFECTIVE_STEP

            # Use slice assignment to color a block of pixels (The Tile)
            # The gap is preserved because we don't write to the pixels in the gap region
            px_array[sx: sx + TILE_SIZE, sy: sy + TILE_SIZE] = col_tuple

    px_array.close()  # Unlock surface

    # 3. Rendering
    # Draw the Grid Centered on the Game Screen
    screen2.blit(grid_surface, (grid_x_offset, grid_y_offset))

    # E. OpenGL Rendering
    texture_data = pygame.image.tobytes(screen2, 'BGRA')
    game_texture.write(texture_data)
    game_texture.use(location=0)

    ctx.clear(0.0, 0.0, 0.0)
    vao.render(mode=moderngl.TRIANGLE_STRIP)

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
sys.exit()