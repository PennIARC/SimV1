import pygame

# --- Simulation Configuration ---
PX_PER_FOOT = 5.0            # Visual scale
ARENA_WIDTH_FT = 300.0
ARENA_HEIGHT_FT = 80.0
BACKGROUND_COLOR = (20, 25, 30)

# --- Physics & Rules ---
NUM_DRONES = 4
DRONE_RADIUS_FT = 0.5        # Physical collision radius
DETECTION_RADIUS_FT = 5.0   # Sensing radius
MAX_SPEED_FT = 12.0          # ft/s
TURN_RATE_RAD = 3.0          # rad/s
TICK_RATE = 1/60.0

# --- Map Generation ---
MINE_COUNT_MIN = 300
MINE_COUNT_MAX = 400
MINE_RADIUS_FT = 0.5
SAFE_DIST_FT = 1.0

# --- Visuals ---
VISUAL_DRONE_SIZE = 1.0      # Scale factor for drone drawing
GRID_LINE_SPACING = 50       # Feet





# --- Colors (Endesga Palette) ---
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
    sebastian_lague_purple = [70, 74, 124]
    sebastian_lague_light_purple = [137, 133, 181]
    network_green = [64, 128, 67]
    network_red = [127, 45, 41]
