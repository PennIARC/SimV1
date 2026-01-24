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
    from path_planning.greedy_path_planning import GreedyBottleneckPlanner
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
# Planner freeze state
planning_enabled = True
freeze_until = None

# Calculate Centering Offset for the Arena
offset_x = (VIRTUAL_W - ARENA_PIXEL_W) // 2
offset_y = (VIRTUAL_H - ARENA_PIXEL_H) // 2

# Planner Setup
MAP_HEIGHT = int(cp.ARENA_HEIGHT_FT) + 1
MAP_WIDTH = int(cp.ARENA_WIDTH_FT) + 1
start_cells = [(0, y) for y in range(MAP_HEIGHT)]
goal_cells = [(MAP_WIDTH - 1, y) for y in range(MAP_HEIGHT)]
planner = GreedyBottleneckPlanner(
    MAP_HEIGHT, MAP_WIDTH, start_cells, goal_cells
)

# Cache the last live path so transient planner failures don't immediately hide the path
last_live_path = None

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

    # C. Do Path planning and update simulation
    waypoints = None
    result = None
    if planning_enabled:
        confidence_map, clearance_map = drone_handler.get_confidence_and_clearance_maps()
        result = planner.plan(confidence_map, clearance_map)

        # Update cached live path: only replace when planner returns a valid path
        if result and isinstance(result, dict) and result.get("path"):
            last_live_path = result.get("path")
        else:
            # keep previous `last_live_path` so transient failures don't hide the path
            pass

        # If planner reached goal, stop drones and begin freeze timer (show final frame)
        if result and isinstance(result, dict) and result.get("reached"):
            print("Planner has found a path to the goal!")
            print("Final Path:", result)
            print("Simulation complete — freezing 3s then exit.")
            drone_handler.stop_all_drones()
            planning_enabled = False
            freeze_until = time.time() + 3.0

        # # produce waypoints only while planning enabled
        # if planning_enabled and result and result.get('path'):
        #     waypoints = planner.suggest_exploration_targets(result['path'], confidence_map)
        # else:
        #     waypoints = None
        
        # produce fixed waypoints (evenly spaced along height at right edge)
        # one target per drone
        waypoints = planner.fixed_targets(num_drones=len(drone_handler.drones))

        # physics update only while planning enabled
        if planning_enabled:
            drone_handler.update(dt, waypoints)
    else:
        # planning disabled: don't update physics so the scene appears frozen
        pass

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

    # 2.a Draw planner paths: persistent (saved) and live (current) if available
    live_path = None
    persistent = None
    persistent = None
    if result and isinstance(result, dict):
        persistent = result.get("persistent_path") if "persistent_path" in result else getattr(planner, "persistent_best", None)
        live_path = result.get("path") or last_live_path
    else:
        persistent = getattr(planner, "persistent_best", None)
        live_path = last_live_path

    # Helper to convert path cells to screen points
    def to_screen(path):
        pts = []
        for px, py in path:
            sx = offset_x + (px * cp.PX_PER_FOOT)
            sy = offset_y + (py * cp.PX_PER_FOOT)
            pts.append((int(sx), int(sy)))
        return pts

    # Draw persistent path first (if present) in a cool color
    if persistent and len(persistent) >= 2:
        persistent_pts = to_screen(persistent)
        try:
            pygame.draw.aalines(screen2, (100, 180, 255), False, persistent_pts)
        except Exception:
            pass
        for p in persistent_pts:
            pygame.draw.circle(screen2, (80, 150, 230), (int(p[0]), int(p[1])), 4)

    # Draw live path on top (if present) in gold
    if live_path and len(live_path) >= 2:
        live_pts = to_screen(live_path)
        try:
            pygame.draw.aalines(screen2, (255, 215, 0), False, live_pts)
        except Exception:
            pass
        for p in live_pts:
            pygame.draw.circle(screen2, (255, 200, 50), (int(p[0]), int(p[1])), 5)

    # Draw drone start markers for reference
    for d in drone_handler.drones:
        sx = int(offset_x + d.pos[0] * cp.PX_PER_FOOT)
        sy = int(offset_y + d.pos[1] * cp.PX_PER_FOOT)
        pygame.draw.circle(screen2, (255, 100, 100), (sx, sy), 6)

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

    # If we entered freeze mode, ensure we hold the final frame for the remaining time
    if freeze_until is not None:
        remaining = freeze_until - time.time()
        if remaining > 0:
            time.sleep(remaining)
        # after freeze, exit main loop
        running = False

pygame.quit()
sys.exit()