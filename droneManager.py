import pygame
import math
import random
from calcs import distance, normalize_angle
import controlPanel as cp

# Import new modular components
from config_loader import get_config, get_algorithm_config
from algorithms import get_algorithm
from algorithms.safe_path_finder import SafePathFinder
from scoring import ScoreTracker


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def update(self, error, dt):
        if dt <= 0: return 0.0
        
        self.integral += error * dt
        
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.prev_error) / dt
            
        self.prev_error = error
        
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True


class Drone:
    # Class-level font cache
    FONT = None

    def __init__(self, id, start_x, start_y):
        self.id = id
        if Drone.FONT is None and pygame.font.get_init():
             try:
                 Drone.FONT = pygame.font.SysFont("Arial", 12, bold=True)
             except:
                 pass
        
        self.pos = [float(start_x), float(start_y)]
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        
        self.active = True
        
        # Visual orientation (preserved for drawing)
        self.draw_angle = 0.0 
        
        # PID Controllers (Position -> Acceleration)
        self.pid_x = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.pid_y = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        
        self.waypoints = []  # Queue of (x, y) coordinates

    def set_pid_params(self, kp, ki, kd):
        self.pid_x.kp = kp
        self.pid_x.ki = ki
        self.pid_x.kd = kd
        self.pid_y.kp = kp
        self.pid_y.ki = ki
        self.pid_y.kd = kd

    def add_waypoint(self, x, y):
        self.waypoints.append((float(x), float(y)))
    
    def set_waypoints(self, waypoints):
        """Set the entire waypoint list at once."""
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        
    def clear_waypoints(self):
        self.waypoints = []
        self.pid_x.reset()
        self.pid_y.reset()
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]

    def update_physics(self, dt):
        if not self.active: return
        
        # 1. Determine Target
        target = None
        if self.waypoints:
            target = self.waypoints[0]
            
        if target:
            # 2. Calculate Error
            error_x = target[0] - self.pos[0]
            error_y = target[1] - self.pos[1]
            
            # 3. PID Update (Output is Desired Acceleration)
            acc_x = self.pid_x.update(error_x, dt)
            acc_y = self.pid_y.update(error_y, dt)
            
            # 4. Clamp Acceleration
            current_acc_mag = math.sqrt(acc_x**2 + acc_y**2)
            if current_acc_mag > cp.MAX_ACCEL_FT:
                scale = cp.MAX_ACCEL_FT / current_acc_mag
                acc_x *= scale
                acc_y *= scale
                
            self.acc = [acc_x, acc_y]
            
            # Check for waypoint completion (within small radius)
            dist_to_target = math.sqrt(error_x**2 + error_y**2)
            if dist_to_target < 1.0:  # 1 foot tolerance
                self.waypoints.pop(0) 
                if not self.waypoints:
                    # Stop if no more waypoints
                    self.pid_x.reset()
                    self.pid_y.reset()
                    self.vel = [0.0, 0.0]
                    self.acc = [0.0, 0.0]
        else:
            # No target, apply friction/damping to stop
            self.acc = [0.0, 0.0]
            self.vel[0] *= 0.9
            self.vel[1] *= 0.9
            
        # 5. Integrate Acceleration -> Velocity
        self.vel[0] += self.acc[0] * dt
        self.vel[1] += self.acc[1] * dt
        
        # 6. Clamp Velocity
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > cp.MAX_SPEED_FT:
            scale = cp.MAX_SPEED_FT / speed
            self.vel[0] *= scale
            self.vel[1] *= scale
            
        # 7. Integrate Velocity -> Position
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # 8. Arena Bounds Clamping (Hard limit)
        self.pos[0] = max(0.0, min(cp.ARENA_WIDTH_FT, self.pos[0]))
        self.pos[1] = max(0.0, min(cp.ARENA_HEIGHT_FT, self.pos[1]))
        
        # 9. Update Draw Angle (visual only)
        if speed > 0.1:
            self.draw_angle = math.atan2(self.vel[1], self.vel[0])

    def draw(self, surface, arena_offset):
        sx = (self.pos[0] * cp.PX_PER_FOOT) + arena_offset[0]
        sy = (self.pos[1] * cp.PX_PER_FOOT) + arena_offset[1]
        
        # Draw Detection Radius
        pygame.draw.circle(surface, cp.Endesga.sebastian_lague_light_purple,
                           (int(sx), int(sy)), int(cp.DETECTION_RADIUS_FT * cp.PX_PER_FOOT), 1)

        # Draw Drone Shape
        size = cp.VISUAL_DRONE_SIZE * cp.PX_PER_FOOT
        points = [
            (sx + math.cos(self.draw_angle) * size, sy + math.sin(self.draw_angle) * size),
            (sx + math.cos(self.draw_angle + 2.6) * size, sy + math.sin(self.draw_angle + 2.6) * size),
            (sx - math.cos(self.draw_angle) * (size * 0.5), sy - math.sin(self.draw_angle) * (size * 0.5)),
            (sx + math.cos(self.draw_angle - 2.6) * size, sy + math.sin(self.draw_angle - 2.6) * size)
        ]
        pygame.draw.polygon(surface, cp.Endesga.white, points)
        
        # Draw waypoints (debug) with numbered circles
        # Draw lines between waypoints
        if self.waypoints:
             # Line from drone to first WP
             drone_screen_pos = (sx, sy)
             prev_pos = drone_screen_pos
             
             for i, (wx, wy) in enumerate(self.waypoints):
                 wsx = (wx * cp.PX_PER_FOOT) + arena_offset[0]
                 wsy = (wy * cp.PX_PER_FOOT) + arena_offset[1]
                 
                 # Draw line
                 pygame.draw.line(surface, cp.Endesga.grey, prev_pos, (wsx, wsy), 1)
                 prev_pos = (wsx, wsy)
                 
                 # Draw Circle
                 pygame.draw.circle(surface, cp.Endesga.debug_red, (int(wsx), int(wsy)), 8)
                 
                 # Draw Number
                 if Drone.FONT:
                     text = Drone.FONT.render(str(self.id + 1), True, cp.Endesga.white)
                     text_rect = text.get_rect(center=(int(wsx), int(wsy)))
                     surface.blit(text, text_rect)


class DroneHandler:
    def __init__(self):
        # Load configuration
        self.config = get_config()
        
        # Initialize state
        self.drones = []
        self.mines_truth = []
        self.mines_detected = []
        self.trees = []
        self.safe_path = []
        self.elapsed = 0.0
        
        # Algorithm state
        self.algorithm = None
        self.phase = "initial"  # initial, scanning, returning, complete
        self.paths_initialized = False
        
        # Scoring
        self.score_tracker = ScoreTracker(self.config)
        
        # Safe Path Finder (Human Planning)
        # Use 'human_planning' section or fallback
        human_algo_name = self.config.get('human_planning', {}).get('algorithm', 'safe_path_bottleneck')
        # Config key for params might be under 'algorithms.safe_path_bottleneck'
        try:
             safe_path_res = float(self.config['algorithms'][human_algo_name]['resolution_ft'])
        except KeyError:
             safe_path_res = 1.0 # default
             
        self.safe_path_finder = SafePathFinder(cp.ARENA_WIDTH_FT, cp.ARENA_HEIGHT_FT, resolution=safe_path_res)
        self.truth_path_finder = SafePathFinder(cp.ARENA_WIDTH_FT, cp.ARENA_HEIGHT_FT, resolution=safe_path_res) # For ground truth
        self.safe_path_timer = 0.0 # Throttle updates
        
        # Initialize Drone Algorithm
        self._init_algorithm()
        
        # Generate initial map (will also initialize drones)
        self.generate_map()

    def _init_algorithm(self):
        """Initialize the drone path-finding algorithm from config."""
        # Config structure changed: drone_planning.algorithm
        if 'drone_planning' in self.config:
            algorithm_name = self.config['drone_planning'].get('algorithm', 'simple_spacing')
        else:
            algorithm_name = self.config.get("algorithm", "simple_spacing") # Fallback
            
        try:
            AlgorithmClass = get_algorithm(algorithm_name)
            self.algorithm = AlgorithmClass(
                config=self.config,
                arena_width=cp.ARENA_WIDTH_FT,
                arena_height=cp.ARENA_HEIGHT_FT
            )
            print(f"[DroneHandler] Using algorithm: {algorithm_name}")
        except ValueError as e:
            print(f"[DroneHandler] Algorithm error: {e}")
            print("[DroneHandler] Falling back to simple_spacing")
            from algorithms.simple_spacing import SimpleSpacingAlgorithm
            self.algorithm = SimpleSpacingAlgorithm(
                config=self.config,
                arena_width=cp.ARENA_WIDTH_FT,
                arena_height=cp.ARENA_HEIGHT_FT
            )

    def generate_map(self):
        """Generate a new map with mines and reset drones."""
        self.trees = []
        self.mines_truth = []
        self.mines_detected = []
        self.elapsed = 0.0
        self.phase = "initial"
        self.paths_initialized = False
        
        # Reset scoring
        self.score_tracker.reset()
        self.safe_path_finder.current_path = [] # Clear path visualization
        
        # Generate mines
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(5, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(1, cp.ARENA_HEIGHT_FT - 1)
            self.mines_truth.append([mx, my])
            
        # Calculate Truth Path (Blue) once
        self.truth_path_finder.update(self.mines_truth)
        
        # Initialize drones with algorithm-based positions
        num_drones = self.config.get("drone", {}).get("count", cp.NUM_DRONES)
        initial_positions = self.algorithm.calculate_initial_positions(num_drones)
        
        self.drones = []
        for i, (x, y) in enumerate(initial_positions):
            drone = Drone(i, x, y)
            self.drones.append(drone)
        
        print(f"[DroneHandler] Generated map with {count} mines and {len(self.drones)} drones")

    def plan_paths(self):
        """
        Plan paths for all drones based on current phase and algorithm.
        """
        # Get current drone positions
        drone_positions = [(d.pos[0], d.pos[1]) for d in self.drones]
        detected_mines = [(m[0], m[1]) for m in self.mines_detected]
        
        if self.phase == "initial":
            # First time: initialize scanning paths
            if not self.paths_initialized:
                waypoints_list = self.algorithm.plan_scan_path(drone_positions)
                for i, drone in enumerate(self.drones):
                    if i < len(waypoints_list):
                        drone.set_waypoints(waypoints_list[i])
                self.phase = "scanning"
                self.paths_initialized = True
                print("[DroneHandler] Phase: SCANNING - Moving across arena")
        
        elif self.phase == "scanning":
            # Check if all drones have completed their waypoints
            all_done = all(len(d.waypoints) == 0 for d in self.drones)
            if all_done:
                # Check if return is enabled
                if self.config.get("path", {}).get("return_after_scan", True):
                    # Plan return paths
                    waypoints_list = self.algorithm.plan_return_path(drone_positions, detected_mines)
                    for i, drone in enumerate(self.drones):
                        if i < len(waypoints_list):
                            drone.set_waypoints(waypoints_list[i])
                    self.phase = "returning"
                    print("[DroneHandler] Phase: RETURNING - Heading back to start")
                else:
                    self.phase = "complete"
                    print("[DroneHandler] Phase: COMPLETE - Scan finished")
        
        elif self.phase == "returning":
            # Check if all drones have returned
            all_done = all(len(d.waypoints) == 0 for d in self.drones)
            if all_done:
                self.phase = "complete"
                print("[DroneHandler] Phase: COMPLETE - Mission finished")
                self._print_final_score()

    def _print_final_score(self):
        """Print the final score breakdown."""
        metrics = self.score_tracker.get_metrics_dict()
        print("\n" + "="*50)
        print("MISSION COMPLETE - FINAL SCORE")
        print("="*50)
        print(f"  Path Width (W):     {metrics['W']:.1f} ft")
        print(f"  Path Length (L):    {metrics['L']:.1f} ft")
        print(f"  Missed Mines (B):   {metrics['B']}")
        print(f"  Time (A):           {metrics['A']:.2f} min")
        print(f"  Weight Penalty (N): {metrics['N']:.1f} oz")
        print("-"*50)
        print(f"  FINAL SCORE:        {metrics['score']:.4f}")
        print("="*50 + "\n")

    def update(self, dt):
        """Update simulation state."""
        self.elapsed += dt
        
        # Update Control Parameters from Global Config
        for drone in self.drones:
            drone.set_pid_params(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        
        # Path Planning
        self.plan_paths()
        
        # Physics Update
        for drone in self.drones:
            drone.update_physics(dt)
            
            # Sensing - detect mines
            for mine in self.mines_truth:
                d = distance(drone.pos, (mine[0], mine[1]))
                if d < cp.DETECTION_RADIUS_FT:
                    if mine not in self.mines_detected:
                        self.mines_detected.append(mine)
        
        # Update Safe Path (throttle to every 0.2s to save performance)
        self.safe_path_timer -= dt
        if self.safe_path_timer <= 0:
            self.safe_path_finder.update(self.mines_detected)
            metrics = self.safe_path_finder.get_path_metrics()
            self.score_tracker.set_safe_path_metrics(
                metrics["clearance_ft"], 
                metrics["length_ft"]
            )
            self.safe_path_timer = 0.2
        
        # Update score tracker
        drone_positions = [(d.pos[0], d.pos[1]) for d in self.drones]
        self.score_tracker.update(
            drone_positions, 
            dt, 
            self.mines_truth, 
            self.mines_detected
        )

    def get_score_string(self) -> str:
        """Get formatted score string for UI display."""
        return self.score_tracker.format_score_string()

    def get_phase(self) -> str:
        """Get current mission phase."""
        return self.phase

    def draw(self, surface, offset=(0, 0)):
        ox, oy = offset

        # Draw Arena
        rect = pygame.Rect(ox, oy, cp.ARENA_WIDTH_FT * cp.PX_PER_FOOT, cp.ARENA_HEIGHT_FT * cp.PX_PER_FOOT)
        pygame.draw.rect(surface, cp.BACKGROUND_COLOR, rect)
        pygame.draw.rect(surface, cp.Endesga.grey_blue, rect, 2)

        # Grid
        for x_ft in range(0, int(cp.ARENA_WIDTH_FT) + 1, cp.GRID_LINE_SPACING):
            lx = ox + (x_ft * cp.PX_PER_FOOT)
            pygame.draw.line(surface, cp.Endesga.my_blue, (lx, rect.top), (lx, rect.bottom), 1)

        # Draw Known Area (Bounding Box of detected mines)
        if self.mines_detected:
            min_x = min(m[0] for m in self.mines_detected)
            max_x = max(m[0] for m in self.mines_detected)
            min_y = min(m[1] for m in self.mines_detected)
            max_y = max(m[1] for m in self.mines_detected)
            
            bbox_rect = pygame.Rect(
                ox + min_x * cp.PX_PER_FOOT, 
                oy + min_y * cp.PX_PER_FOOT, 
                (max_x - min_x) * cp.PX_PER_FOOT, 
                (max_y - min_y) * cp.PX_PER_FOOT
            )
            pygame.draw.rect(surface, (0, 255, 255), bbox_rect, 1) # Cyan box

        # Draw Truth Path (Blue Corridor - Optimal Baseline)
        truth_points = self.truth_path_finder.current_path
        if truth_points:
             # Draw centerline in Blue
             screen_points = [(ox + p[0]*cp.PX_PER_FOOT, oy + p[1]*cp.PX_PER_FOOT) for p in truth_points]
             if len(screen_points) > 1:
                 # Dash effect or thinner line to distinguish
                 pygame.draw.lines(surface, (50, 100, 255), False, screen_points, 1)

        # Draw Safe Path (Green Corridor)
        path_points = self.safe_path_finder.current_path
        if path_points:
            # Draw centerline
            screen_points = [(ox + p[0]*cp.PX_PER_FOOT, oy + p[1]*cp.PX_PER_FOOT) for p in path_points]
            if len(screen_points) > 1:
                pygame.draw.lines(surface, (50, 200, 100), False, screen_points, 3)
                
            # Draw clearance circles along the path
            radius_px = self.safe_path_finder.path_clearance * cp.PX_PER_FOOT
            for p in screen_points[::10]: # Sample points to avoid drawing too many circles
                 pygame.draw.circle(surface, (50, 200, 100, 100), (int(p[0]), int(p[1])), int(radius_px), 1)

        # Mines on top of path
        for m in self.mines_truth:
            if m not in self.mines_detected:
                mx = ox + (m[0] * cp.PX_PER_FOOT)
                my = oy + (m[1] * cp.PX_PER_FOOT)
                pygame.draw.circle(surface, (40, 45, 55), (int(mx), int(my)), 1)
                
        for m in self.mines_detected:
            mx = ox + (m[0] * cp.PX_PER_FOOT)
            my = oy + (m[1] * cp.PX_PER_FOOT)
            pygame.draw.circle(surface, cp.Endesga.network_red, (int(mx), int(my)), 2)

        # Drones on top
        for d in self.drones:
            d.draw(surface, (ox, oy))