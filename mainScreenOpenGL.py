import pygame
import moderngl
import struct
import math
import time
import sys
import array
import numpy as np

# Import Framework Utilities
try:
    from text import drawText
    from fontDict import fonts
    from calcs import linearGradient, normalize
    import controlPanel as cp
    from droneManager import DroneHandler
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
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


# --- HEATMAP CONFIGURATION ---
# The heatmap covers the specific Arena dimensions
HEAT_TILE_SIZE = 11  # Size of one heat tile in pixels
HEAT_TILE_GAP = 1  # Gap between tiles
HEAT_EFFECTIVE_STEP = HEAT_TILE_SIZE + HEAT_TILE_GAP

# Calculate Grid Dimensions based on Arena Size
ARENA_PIXEL_W = int(cp.ARENA_WIDTH_FT * cp.PX_PER_FOOT)
ARENA_PIXEL_H = int(cp.ARENA_HEIGHT_FT * cp.PX_PER_FOOT)

GRID_COLS = ARENA_PIXEL_W // HEAT_EFFECTIVE_STEP
GRID_ROWS = ARENA_PIXEL_H // HEAT_EFFECTIVE_STEP

# Colors for Heatmap: Red (Close to Mine) -> Green (Safe)
HEAT_PALETTE = [[200, 40, 80], [40, 100, 120]]
# Distance (in pixels) at which the heatmap turns fully "Cold" (Green)
# 50px = 10ft at 5px/ft
HEAT_MAX_DIST = 50.0


def generate_heatmap_surface(mines_truth):
    """
    Generates a Pygame surface representing the proximity to mines.
    Uses NumPy for vectorized distance calculation to handle hundreds of mines efficiently.
    """
    width = GRID_COLS * HEAT_EFFECTIVE_STEP
    height = GRID_ROWS * HEAT_EFFECTIVE_STEP
    surf = pygame.Surface((width, height))
    surf.fill(cp.Endesga.darker_my_blue)  # Background color (gaps)

    if not mines_truth:
        return surf

    # 1. Setup Coordinates
    # Create a meshgrid of tile center coordinates
    x_coords = np.arange(GRID_COLS) * HEAT_EFFECTIVE_STEP + (HEAT_TILE_SIZE / 2)
    y_coords = np.arange(GRID_ROWS) * HEAT_EFFECTIVE_STEP + (HEAT_TILE_SIZE / 2)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # Shape: (Rows * Cols, 2)
    tile_points = np.stack([xv.flatten(), yv.flatten()], axis=1)

    # Mines: Convert list of [x, y] (feet) to pixels
    mines_px = np.array(mines_truth) * cp.PX_PER_FOOT

    if len(mines_px) == 0:
        return surf

    # 2. Vectorized Distance Calculation
    # We want for every tile, the distance to the CLOSEST mine.
    # This can still be heavy if Grid is huge.
    # Tile Points: ~4800 (150*32). Mines: ~400.
    # Broadcast: (4800, 1, 2) - (1, 400, 2) -> (4800, 400, 2) floats is ~30MB RAM, totally fine.

    diff = tile_points[:, np.newaxis, :] - mines_px[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)  # (Tiles, Mines)
    min_dists = np.sqrt(np.min(dists_sq, axis=1))  # (Tiles,)

    # 3. Color Mapping
    # Normalize distances
    norm_dists = np.clip(min_dists / HEAT_MAX_DIST, 0.0, 1.0)

    # Manual Linear Gradient (Red to Blue/Green)
    # P0 = HEAT_PALETTE[0], P1 = HEAT_PALETTE[1]
    # Col = P0 + (P1 - P0) * t
    c0 = np.array(HEAT_PALETTE[0])
    c1 = np.array(HEAT_PALETTE[1])

    # Shape: (Tiles, 3)
    colors = c0 + (c1 - c0) * norm_dists[:, np.newaxis]
    colors = colors.astype(np.uint8)

    # 4. Drawing to Surface
    # We use PixelArray for block drawing
    px_array = pygame.PixelArray(surf)

    # We iterate manually to draw the blocks (PixelArray slice assignment is fast)
    # Reshape colors back to (Rows, Cols, 3)
    colors_grid = colors.reshape((GRID_ROWS, GRID_COLS, 3))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            col = tuple(colors_grid[r, c])
            sx = c * HEAT_EFFECTIVE_STEP
            sy = r * HEAT_EFFECTIVE_STEP
            # Draw the tile (leaving the gap)
            px_array[sx: sx + HEAT_TILE_SIZE, sy: sy + HEAT_TILE_SIZE] = col

    px_array.close()
    return surf


# --- INIT ---
pygame.init()

# 1. CONFIGURE OPENGL ATTRIBUTES
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# 2. Setup Window
screen = pygame.display.set_mode((0, 0), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
info = pygame.display.Info()
WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h

# 3. Create ModernGL Context
try:
    ctx = moderngl.create_context(require=330)
except Exception as e:
    print(f"Error creating ModernGL context: {e}")
    sys.exit(1)

ctx.enable(moderngl.BLEND)

# 4. Game Logic Variables
clock = pygame.time.Clock()
fps = 60
# Keep scale 1:1 for simulation accuracy, or adjust if you want pixel art style
# Given the Sim uses 5px per foot, 1:1 is usually best for clarity.
scaleDownFactor = 1

# Virtual Resolution
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
quad_buffer = ctx.buffer(data=array.array('f', [
    -1.0, 1.0, 0.0, 0.0,  # TL
    -1.0, -1.0, 0.0, 1.0,  # BL
    1.0, 1.0, 1.0, 0.0,  # TR
    1.0, -1.0, 1.0, 1.0,  # BR
]))

# 7. Load Shaders
prog = load_shader(ctx, 'shaders/basic.vert', 'shaders/basic.frag')
vao = ctx.vertex_array(prog, [(quad_buffer, '2f 2f', 'in_vert', 'in_text')])

# Font Setup
try:
    montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
except KeyError:
    montserratRegularAdaptive = pygame.font.SysFont("Arial", 20)

# --- SIMULATION STATE ---
drone_handler = DroneHandler()
timer = 0
toggle = True
running = True
last_time = time.time()

# Heatmap State
heatmap_surface = None
heatmap_needs_update = True

# Calculate Centering Offset for the Arena
offset_x = (VIRTUAL_W - ARENA_PIXEL_W) // 2
offset_y = (VIRTUAL_H - ARENA_PIXEL_H) // 2

pygame.mouse.set_visible(False)

# --- MAIN LOOP ---
while running:
    # A. Time Management
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0.1: dt = 0.1

    # B. Input Handling
    mx, my = pygame.mouse.get_pos()
    vmx, vmy = mx / scaleDownFactor, my / scaleDownFactor

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
            if event.key == pygame.K_r:
                drone_handler.generate_map()
                heatmap_needs_update = True  # Flag to regenerate heatmap

    # C. Update Simulation
    drone_handler.update(dt)

    # Update Heatmap if needed (Static map, so only on regen)
    if heatmap_needs_update:
        print("Regenerating Heatmap...")
        t0 = time.time()
        heatmap_surface = generate_heatmap_surface(drone_handler.mines_truth)
        print(f"Heatmap generated in {time.time() - t0:.4f}s")
        heatmap_needs_update = False

    # D. Rendering to Pygame Surfaces
    screen2.fill(cp.Endesga.darker_my_blue)
    screenUI.fill((0, 0, 0, 0))

    # 1. Draw Heatmap (Underlay)
    if heatmap_surface:
        screen2.blit(heatmap_surface, (offset_x, offset_y))

    # 2. Draw Simulation (Drones, detected mines, etc.)
    # Note: drone_handler draws its own background/grid which might cover the heatmap.
    # We should modify how it draws or draw heatmap on top with blending if we want it visible.
    # However, looking at DroneHandler.draw:
    # It draws a Rect with BACKGROUND_COLOR.
    # Let's draw the Heatmap *instead* of the background rect inside the arena area,
    # or pass a flag to DroneHandler to skip background clearing.
    # For now, simply drawing DroneHandler AFTER heatmap will obscure the heatmap.
    # To fix this without modifying DroneHandler deeply, we can use blending or just let the heatmap act as the "floor".
    # BUT DroneHandler draws grid lines and trees too.

    # Strategy: Draw Heatmap. Then Draw DroneHandler elements manually?
    # Or rely on transparency?
    # Let's assume for this specific OpenGL main screen, we want the heatmap to be the dominant visual for the floor.

    # To allow the heatmap to show through, we can manually draw the essential parts of DroneHandler here
    # OR we draw DroneHandler first, then blend Heatmap on top using Multiply or Add.

    # Let's try: Draw DroneHandler (Standard), then Overlay Heatmap with transparency.
    drone_handler.draw(screen2, (offset_x, offset_y))

    if heatmap_surface:
        # Draw heatmap with partial transparency on top of the arena
        heatmap_overlay = heatmap_surface.copy()
        heatmap_overlay.set_alpha(150)  # Semi-transparent
        screen2.blit(heatmap_overlay, (offset_x, offset_y))

    # 3. Draw UI
    if toggle:
        items = {
            "FPS": round(clock.get_fps()),
            "Time": f"{drone_handler.elapsed:.1f}s",
            "Detected": f"{len(drone_handler.mines_detected)} / {len(drone_handler.mines_truth)}",
            "Controls": "[R] Reset  [ESC] Quit  [SPACE] UI"
        }

        for i, (label, val) in enumerate(items.items()):
            full_str = f"{label}: {val}"
            drawText(screenUI, cp.Endesga.debug_red, montserratRegularAdaptive,
                     5, VIRTUAL_H - (30 + 25 * i),
                     full_str, cp.Endesga.black, 1, antiAliasing=False)

        # Custom Cursor
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, cp.Endesga.black, (vmx + 1, vmy + 1), 2, 1)
        pygame.draw.circle(screenUI, cp.Endesga.white, (vmx, vmy), 2, 1)

    screen2.blit(screenUI, (0, 0))

    # E. OpenGL Rendering
    try:
        texture_data = pygame.image.tobytes(screen2, 'BGRA')
        game_texture.write(texture_data)
        game_texture.use(location=0)

        ctx.clear(0.0, 0.0, 0.0)
        vao.render(mode=moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(fps)
    except Exception as e:
        print(f"Render Error: {e}")
        running = False

pygame.quit()
sys.exit()