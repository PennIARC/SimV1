import pygame
import math
import random
from calcs import distance, normalize_angle
import controlPanel as cp
from iarc.integrations.simv1_adapter import plan_for_simv1
from iarc.scoring.score import score_path
from iarc.core.types import PathPlan
from iarc.swarm.strategy_frontier import allocate_frontier

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
    def __init__(self, id, start_x, start_y):
        self.id = id
        self.pos = [float(start_x), float(start_y)]
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        
        self.active = True
        
        # Visual orientation (preserved for drawing)
        self.draw_angle = 0.0 
        
        # PID Controllers (Position -> Acceleration)
        self.pid_x = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.pid_y = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        
        self.waypoints = [] # Queue of (x, y) coordinates

    def set_pid_params(self, kp, ki, kd):
        self.pid_x.kp = kp
        self.pid_x.ki = ki
        self.pid_x.kd = kd
        self.pid_y.kp = kp
        self.pid_y.ki = ki
        self.pid_y.kd = kd

    def add_waypoint(self, x, y):
        self.waypoints.append((float(x), float(y)))
        
    def clear_waypoints(self):
        self.waypoints = []
        self.pid_x.reset()
        self.pid_y.reset()
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]

    def update_physics(self, dt):
        if not self.active: return
        
        target = self.waypoints[0] if self.waypoints else None
        
        if target:
            error_x = target[0] - self.pos[0]
            error_y = target[1] - self.pos[1]
            
            acc_x = self.pid_x.update(error_x, dt)
            acc_y = self.pid_y.update(error_y, dt)
            
            # Clamp acceleration
            acc_mag = math.sqrt(acc_x**2 + acc_y**2)
            if acc_mag > cp.MAX_ACCEL_FT:
                scale = cp.MAX_ACCEL_FT / acc_mag
                acc_x *= scale
                acc_y *= scale
            self.acc = [acc_x, acc_y]
            
            # Check waypoint completion
            if math.sqrt(error_x**2 + error_y**2) < 1.0:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.pid_x.reset()
                    self.pid_y.reset()
                    self.vel = [0.0, 0.0]
                    self.acc = [0.0, 0.0]
        else:
            self.acc = [0.0, 0.0]
            self.vel[0] *= 0.9
            self.vel[1] *= 0.9
            
        # Integrate velocity
        self.vel[0] += self.acc[0] * dt
        self.vel[1] += self.acc[1] * dt
        
        # Clamp velocity
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > cp.MAX_SPEED_FT:
            scale = cp.MAX_SPEED_FT / speed
            self.vel[0] *= scale
            self.vel[1] *= scale
            
        # Integrate position
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # Clamp to arena bounds
        self.pos[0] = max(0.0, min(cp.ARENA_WIDTH_FT, self.pos[0]))
        self.pos[1] = max(0.0, min(cp.ARENA_HEIGHT_FT, self.pos[1]))
        
        # Update visual angle
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
        
        # Draw waypoints (debug)
        if self.waypoints:
             for wx, wy in self.waypoints:
                 wsx = (wx * cp.PX_PER_FOOT) + arena_offset[0]
                 wsy = (wy * cp.PX_PER_FOOT) + arena_offset[1]
                 pygame.draw.circle(surface, cp.Endesga.debug_red, (int(wsx), int(wsy)), 3)


class DroneHandler:
    def __init__(self):
        self.drones = []
        # Spread drones vertically across the arena
        spacing = cp.ARENA_HEIGHT_FT / (cp.NUM_DRONES + 1)
        for i in range(cp.NUM_DRONES):
            start_y = spacing * (i + 1)  # Spread across height
            self.drones.append(Drone(i, 3.0, start_y))
        self.generate_map()

    def generate_map(self):
        # Core state
        self.mines_truth = []
        self.mines_detected = []
        self.safe_path = []
        self.world_waypoints = []
        self.current_corridor_width = 0
        
        # Timing
        self.elapsed = 0.0
        self.tick = 0
        self.plan_elapsed = 0.0
        self.plan_interval = 0.5
        
        # Generate mines
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(5, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(1, cp.ARENA_HEIGHT_FT - 1)
            self.mines_truth.append([mx, my])

        # Initial scan: drones spread out and move forward
        for i, d in enumerate(self.drones):
            d.clear_waypoints()
            # Spread drones in a fan pattern forward
            forward_dist = 30.0 + (i * 10.0)  # Different distances for each drone
            target_x = min(cp.ARENA_WIDTH_FT - 5.0, d.pos[0] + forward_dist)
            d.add_waypoint(target_x, d.pos[1])

    def get_belief_grid(self):
        """
        Create a belief grid from detected mines.
        Grid convention: grid[row][col] where row=Y, col=X
        Cell size: 2 ft per cell
        Dimensions: 40 rows (80ft height) × 150 cols (300ft width)
        """
        num_rows = int(cp.ARENA_HEIGHT_FT / 2)   # 40 rows
        num_cols = int(cp.ARENA_WIDTH_FT / 2)    # 150 cols

        # Initialize grid: -1 = unknown, 0 = known free, 1 = mine/obstacle
        grid = [[-1 for _ in range(num_cols)] for _ in range(num_rows)]

        # Mark nearby cells around each drone as known free (sensed area)
        sensed_cells = int(cp.DETECTION_RADIUS_FT / 2)
        for drone in self.drones:
            dc = int(drone.pos[0] / 2)
            dr = int(drone.pos[1] / 2)
            for rr in range(dr - sensed_cells, dr + sensed_cells + 1):
                for cc in range(dc - sensed_cells, dc + sensed_cells + 1):
                    if 0 <= rr < num_rows and 0 <= cc < num_cols:
                        grid[rr][cc] = 0

        # Mark detected mines in the grid
        for mine in self.mines_detected:
            mx, my = mine[0], mine[1]
            col = int(mx / 2)
            row = int(my / 2)
            if 0 <= row < num_rows and 0 <= col < num_cols:
                grid[row][col] = 1

        return grid

    def plan_paths(self):
        """
        Plan paths using greedy_bottleneck algorithm with FIXED START.
        Drones fly left to right (along X axis = columns).
        """
        belief_grid = self.get_belief_grid()
        # For planning, treat unknown as traversable but track it in `unknown_mask`.
        unknown_mask = [[cell < 0 for cell in row] for row in belief_grid]
        grid_for_plan = [[0 if cell < 0 else cell for cell in row] for row in belief_grid]

        num_rows = int(cp.ARENA_HEIGHT_FT / 2)   # 40
        num_cols = int(cp.ARENA_WIDTH_FT / 2)    # 150

        # FIXED START: middle of Y-axis, left edge (Row 20, Col 0)
        start = (20, 0)

        # Goals: right edge (col=149), all rows
        goals = [(r, num_cols - 1) for r in range(num_rows)]

        # Detected hazards in grid coordinates
        hazards = []
        for mine in self.mines_detected:
            col = int(mine[0] / 2)
            row = int(mine[1] / 2)
            if 0 <= row < num_rows and 0 <= col < num_cols:
                hazards.append((row, col))

        plan = plan_for_simv1(
            grid_for_plan,
            start,
            goals,
            hazards=hazards,
            unknown=unknown_mask,
            planner_type=cp.PLANNER_TYPE,
            current_path=self.safe_path,
            time_budget=cp.RRT_TIME_BUDGET,
            turning_radius=cp.DUBINS_TURN_RADIUS,
        )

        self.safe_path = plan.path
        self.current_corridor_width = plan.g_width

        # Convert grid path to world coordinates (feet)
        self.world_waypoints = [(col * 2 + 1, row * 2 + 1) for (row, col) in plan.path]

        # Log path changes
        if getattr(self, '_last_path_len', None) != len(plan.path):
            self._last_path_len = len(plan.path)
            print(f"[Planner] {len(plan.path)} pts, G={plan.g_width}, mines={len(self.mines_detected)}")

        # Allocate drones to frontier targets
        drone_positions = [(int(d.pos[1] / 2), int(d.pos[0] / 2)) for d in self.drones]
        assignments = allocate_frontier(drone_positions, belief_grid)
        
        if assignments:
            targets_dict = {}
            for i, (drone_pos, target_pos) in enumerate(assignments):
                if i < len(self.drones):
                    target_x = target_pos[1] * 2 + 1  # col to feet
                    target_y = target_pos[0] * 2 + 1  # row to feet
                    targets_dict[i] = (target_x, target_y)
                    self.drones[i].clear_waypoints()
                    self.drones[i].add_waypoint(target_x, target_y)
            if self.tick % 100 == 0:  # Print every 100 ticks
                print(f"[Frontier] tick={self.tick} targets={targets_dict}")
        else:
            # Fallback: follow the planned path
            for drone in self.drones:
                if not drone.waypoints and self.world_waypoints:
                    drone.clear_waypoints()
                    for wp in self.world_waypoints[:5]:  # Add first 5 waypoints
                        drone.add_waypoint(wp[0], wp[1])

    def update(self, dt):
        self.elapsed += dt
        self.plan_elapsed += dt
        self.tick += 1
        
        for drone in self.drones:
            drone.set_pid_params(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        
        # Replan periodically
        if self.plan_elapsed >= self.plan_interval:
            self.plan_elapsed = 0.0
            self.plan_paths()
        
        # Physics + sensing
        for drone in self.drones:
            drone.update_physics(dt)
            for mine in self.mines_truth:
                if distance(drone.pos, mine) < cp.DETECTION_RADIUS_FT:
                    if mine not in self.mines_detected:
                        self.mines_detected.append(mine)

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

        # Mines
        for m in self.mines_truth:
            if m not in self.mines_detected:
                mx = ox + (m[0] * cp.PX_PER_FOOT)
                my = oy + (m[1] * cp.PX_PER_FOOT)
                pygame.draw.circle(surface, (40, 45, 55), (int(mx), int(my)), 1)
                
        for m in self.mines_detected:
            mx = ox + (m[0] * cp.PX_PER_FOOT)
            my = oy + (m[1] * cp.PX_PER_FOOT)
            pygame.draw.circle(surface, cp.Endesga.network_red, (int(mx), int(my)), 2)

        # ===== Draw Planned Path (skip unexplored start, show future planning) =====
        if len(self.world_waypoints) > 1 and self.drones:
            belief_grid = self.get_belief_grid()
            num_rows = len(belief_grid)
            num_cols = len(belief_grid[0]) if belief_grid else 0
            
            # Find the leftmost explored X position
            min_explored_x = float('inf')
            for r in range(num_rows):
                for c in range(num_cols):
                    if belief_grid[r][c] >= 0:  # Explored cell
                        cell_x = c * 2 + 1
                        if cell_x < min_explored_x:
                            min_explored_x = cell_x
            
            # Find the furthest drone X position
            max_drone_x = max(d.pos[0] for d in self.drones)
            
            # Draw waypoints from explored area up to current frontier
            explored_waypoints = []
            for wx, wy in self.world_waypoints:
                # Skip waypoints before the explored region
                if wx < min_explored_x - 5:
                    continue
                # Show waypoints only up to where drones have reached (small buffer)
                if wx <= max_drone_x + 5:  # Just 5ft buffer ahead
                    explored_waypoints.append((wx, wy))
                else:
                    break
            
            if len(explored_waypoints) > 1:
                screen_points = []
                for wx, wy in explored_waypoints:
                    sx = ox + (wx * cp.PX_PER_FOOT)
                    sy = oy + (wy * cp.PX_PER_FOOT)
                    screen_points.append((sx, sy))

                # Draw semi-transparent corridor width
                width_cells = (self.current_corridor_width * 2) + 1
                width_px = max(1, int(width_cells * cp.PX_PER_FOOT))

                path_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                pygame.draw.lines(path_surface, (50, 200, 100, 80), False, screen_points, width_px)
                surface.blit(path_surface, (0, 0))

                # Draw solid green centerline
                pygame.draw.lines(surface, cp.Endesga.network_green, False, screen_points, 3)

                # Draw waypoint markers
                for i, pt in enumerate(screen_points):
                    if i % 5 == 0:  # Every 5th point
                        pygame.draw.circle(surface, cp.Endesga.network_green, (int(pt[0]), int(pt[1])), 3)

        # Drones on top (drawn last so they're visible)
        for d in self.drones:
            d.draw(surface, (ox, oy))

    def _ensure_4_connected(self, path):
        if not path:
            return []
        fixed = [path[0]]
        for r, c in path[1:]:
            pr, pc = fixed[-1]
            while pr != r:
                pr += 1 if r > pr else -1
                fixed.append((pr, pc))
            while pc != c:
                pc += 1 if c > pc else -1
                fixed.append((pr, pc))
        return fixed