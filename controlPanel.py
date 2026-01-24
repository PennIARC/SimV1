from config_loader import get_config

# Load Initial Config
config = get_config()

# Helper to reload constants if config updates
def reload():
    global config
    config = get_config()
    
    # --- Simulation Configuration ---
    global PX_PER_FOOT, ARENA_WIDTH_FT, ARENA_HEIGHT_FT, BACKGROUND_COLOR
    PX_PER_FOOT = float(config['visuals']['px_per_foot'])
    ARENA_WIDTH_FT = float(config['arena']['width_ft'])
    ARENA_HEIGHT_FT = float(config['arena']['height_ft'])
    BACKGROUND_COLOR = tuple(config['visuals']['background_color'])
    
    # --- Physics & Rules ---
    global NUM_DRONES, DETECTION_RADIUS_FT, MAX_SPEED_FT, MAX_ACCEL_FT, TURN_RATE_RAD, DRONE_RADIUS_FT
    NUM_DRONES = int(config['drone']['count'])
    DETECTION_RADIUS_FT = float(config['drone']['detection_radius_ft'])
    MAX_SPEED_FT = float(config['physics']['max_speed_ft'])
    MAX_ACCEL_FT = float(config['physics']['max_accel_ft'])
    TURN_RATE_RAD = float(config['physics']['turn_rate_rad'])
    DRONE_RADIUS_FT = float(config['drone']['radius_ft'])
    
    # --- PID Control ---
    global PID_KP, PID_KI, PID_KD, TICK_RATE
    PID_KP = float(config['physics']['pid']['kp'])
    PID_KI = float(config['physics']['pid']['ki'])
    PID_KD = float(config['physics']['pid']['kd'])
    TICK_RATE = float(config['physics']['tick_rate'])
    
    # --- Map Generation ---
    global MINE_COUNT_MIN, MINE_COUNT_MAX, MINE_RADIUS_FT, SAFE_DIST_FT
    MINE_COUNT_MIN = int(config['map']['mine_count_min'])
    MINE_COUNT_MAX = int(config['map']['mine_count_max'])
    MINE_RADIUS_FT = float(config['map']['mine_radius_ft'])
    SAFE_DIST_FT = float(config['map']['safe_dist_ft'])
    
    # --- Visuals ---
    global VISUAL_DRONE_SIZE, GRID_LINE_SPACING
    VISUAL_DRONE_SIZE = float(config['visuals']['drone_size'])
    GRID_LINE_SPACING = int(config['visuals']['grid_line_spacing_ft'])
    
    # --- IARC Scoring ---
    global GREEN_ZONE_SQUARES, WEIGHT_PENALTY_OZ
    GREEN_ZONE_SQUARES = int(config['scoring']['green_zone_squares'])
    WEIGHT_PENALTY_OZ = float(config['scoring']['weight_penalty_oz'])

# Execute initial load
reload()





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
