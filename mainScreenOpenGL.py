import pygame
import moderngl
import math
import time
import sys
import array
import numpy as np
from path_planning.waypoint_allocator import WaypointAllocator

# Import Framework Utilities
try:
    from text import drawText
    from fontDict import fonts
    import controlPanel as cp
    from droneManager import DroneHandler
    from path_planning.greedy_path_planning import GreedyBottleneckPlanner
    from path_planning.rrt_planner import RRTPlanner
    from path_planning.a_star_planner import AStarPlanner
    from scoring import compute_score, format_score
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
    x_coords = np.arange(GRID_COLS) * HEAT_EFFECTIVE_STEP + (HEAT_EFFECTIVE_STEP / 2)
    y_coords = np.arange(GRID_ROWS) * HEAT_EFFECTIVE_STEP + (HEAT_EFFECTIVE_STEP / 2)
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
screen = pygame.display.set_mode((0, 0), pygame.OPENGL | pygame.DOUBLEBUF|pygame.RESIZABLE)
info = pygame.display.Info()
WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h

# --- Dynamic scale: fit arena as large as possible in the window ---
_scale_w = (WINDOW_WIDTH * 0.95) / cp.ARENA_WIDTH_FT
_scale_h = (WINDOW_HEIGHT * 0.88) / cp.ARENA_HEIGHT_FT
_cell_px = max(4, int(cp.COMP_CELL_SIZE_FT * min(_scale_w, _scale_h)))
cp.PX_PER_FOOT = _cell_px / cp.COMP_CELL_SIZE_FT

# --- HEATMAP CONFIGURATION ---
# Grid matches competition rules: 150 x 40 cells, each 2x2 ft
GRID_COLS = cp.COMP_FIELD_LENGTH_CELLS   # 150
GRID_ROWS = cp.COMP_FIELD_WIDTH_CELLS    # 40
HEAT_PALETTE = [[200, 40, 80], [40, 100, 120]]


HEAT_TILE_SIZE = _cell_px - 1        # 1px gap for visible grid lines
HEAT_EFFECTIVE_STEP = _cell_px
ARENA_PIXEL_W = GRID_COLS * _cell_px
ARENA_PIXEL_H = GRID_ROWS * _cell_px
HEAT_MAX_DIST = 10.0 * cp.PX_PER_FOOT

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

# Planner Setup — uses 2ft competition grid
start_cells = [(0, y) for y in range(GRID_ROWS)]
goal_cells = [(GRID_COLS - 1, y) for y in range(GRID_ROWS)]


##DIFFERENT PLANNER ALGORITHMS
if cp.PLANNER_ALGORITHM == "greedy":
    planner = GreedyBottleneckPlanner(
        GRID_ROWS, GRID_COLS, start_cells, goal_cells
    )
elif cp.PLANNER_ALGORITHM == "rrt":
    planner = RRTPlanner(
        GRID_ROWS, GRID_COLS, start_cells, goal_cells,
        max_iter=3000, step_size=5, goal_bias=0.15,
    )
elif cp.PLANNER_ALGORITHM == "astar":
    planner = AStarPlanner(
        GRID_ROWS, GRID_COLS, start_cells, goal_cells,
    )
else:
    raise ValueError(f"Unknown PLANNER_ALGORITHM: {cp.PLANNER_ALGORITHM}")

# Waypoint allocator — drone movement is independent of path planning
allocator = WaypointAllocator(GRID_ROWS, GRID_COLS)

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
        confidence_map, clearance_map, mines_detected_high, mines_detected_low = drone_handler.get_maps()
        result = planner.plan(confidence_map, clearance_map)

        # Update cached live path: only replace when planner returns a valid path
        if result and isinstance(result, dict) and result.get("path"):
            last_live_path = result.get("path")


        # If planner reached goal, stop drones and begin freeze timer (show final frame)
        if result and isinstance(result, dict) and result.get("reached"):
            print("Planner has found a path to the goal!")
            path = result["path"]
            A_min = drone_handler.elapsed / 60.0

            # Score using ground-truth mines (final evaluation)
            score_truth, G_truth, info_truth = compute_score(
                path, drone_handler.mines_truth, A_min,
                grid_w=GRID_COLS, grid_h=GRID_ROWS, N_oz=cp.WEIGHT_N_OZ)
            # Score using detected mines (what we knew during planning)
            score_det, G_det, info_det = compute_score(
                path, drone_handler.mines_detected, A_min,
                grid_w=GRID_COLS, grid_h=GRID_ROWS,  N_oz=cp.WEIGHT_N_OZ)

            print(format_score(score_truth, G_truth, info_truth,
                               algorithm=cp.PLANNER_ALGORITHM, seed=cp.MINE_SEED))
            print(f"  Bottleneck: {result.get('bottleneck', '?')}")
            print(f"=== SCORE (detected mines only) ===")
            print(f"  Score: {score_det:.2f}  |  G={G_det}  B={info_det.get('B', 0)}")
            print(f"Final Path length: {len(path)} cells")
            print("Simulation complete — freezing 3s then exit.")
            drone_handler.stop_all_drones()
            planning_enabled = False
            freeze_until = time.time() + 3.0

        # produce waypoints for drones (independent of path planning)
        half = cp.COMP_CELL_SIZE_FT / 2.0
        drone_positions_grid = []
        for d in drone_handler.drones:
            grid_x = int(d.pos[0] / cp.COMP_CELL_SIZE_FT)
            grid_y = int(d.pos[1] / cp.COMP_CELL_SIZE_FT)
            drone_positions_grid.append((grid_x, grid_y))

        waypoints_grid = allocator.assign(
            drone_positions_grid=drone_positions_grid,
            confidence_map=confidence_map,
            clearance_map=clearance_map,
            mines_detected_high=mines_detected_high,
            sensing_radius_high=int(cp.DETECTION_RADIUS_FT_LARGE / cp.COMP_CELL_SIZE_FT),
            sensing_radius_low=int(cp.DETECTION_RADIUS_FT_SMALL / cp.COMP_CELL_SIZE_FT))
        waypoints = [(x * cp.COMP_CELL_SIZE_FT + half, y * cp.COMP_CELL_SIZE_FT + half) for x, y in waypoints_grid]

    else:
        half = cp.COMP_CELL_SIZE_FT / 2.0
        waypoints_grid = allocator.fixed_targets(num_drones=len(drone_handler.drones))
        waypoints = [(x * cp.COMP_CELL_SIZE_FT + half, y * cp.COMP_CELL_SIZE_FT + half) for x, y in waypoints_grid]
        
    # physics update
    drone_handler.update(dt, waypoints)

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
    drone_handler.draw(screen2, (offset_x, offset_y))

    if heatmap_surface:
        # Draw heatmap with partial transparency on top of the arena
        heatmap_overlay = heatmap_surface.copy()
        heatmap_overlay.set_alpha(150)  # Semi-transparent
        screen2.blit(heatmap_overlay, (offset_x, offset_y))

    # 2.a Draw planner paths: persistent (saved) and live (current) if available
    live_path = None
    persistent = None
    if result and isinstance(result, dict):
        persistent = result.get("persistent_path") if "persistent_path" in result else getattr(planner, "persistent_best", None)
        live_path = result.get("path") or last_live_path
    else:
        persistent = getattr(planner, "persistent_best", None)
        live_path = last_live_path

    # Helper to convert path cells (2ft grid) to screen points (centered in cell)
    _half_cell_px = HEAT_EFFECTIVE_STEP / 2.0
    def to_screen(path):
        pts = []
        for px, py in path:
            sx = offset_x + px * HEAT_EFFECTIVE_STEP + _half_cell_px
            sy = offset_y + py * HEAT_EFFECTIVE_STEP + _half_cell_px
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