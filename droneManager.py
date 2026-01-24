import pygame
import math
import random
from calcs import distance, normalize_angle
import controlPanel as cp
from iarc.integrations.simv1_adapter import plan_for_simv1
from iarc.scoring.score import score_path
from iarc.core.types import PathPlan
from iarc.swarm.allocator import DroneState, allocate_receding

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
            if dist_to_target < 1.0: # 1 foot tolerance
                self.waypoints.pop(0) 
                # Don't reset PID immediately to maintain flow? 
                # Actually, if we pop, the next waypoint becomes target. 
                # If no waypoints left, we might overshoot if not handled.
                if not self.waypoints:
                    # Stop if no more waypoints
                    self.pid_x.reset()
                    self.pid_y.reset()
                    self.vel = [0.0, 0.0] # Hard stop or friction?
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
        
        # Draw waypoints (debug)
        if self.waypoints:
             for wx, wy in self.waypoints:
                 wsx = (wx * cp.PX_PER_FOOT) + arena_offset[0]
                 wsy = (wy * cp.PX_PER_FOOT) + arena_offset[1]
                 pygame.draw.circle(surface, cp.Endesga.debug_red, (int(wsx), int(wsy)), 3)


class DroneHandler:
    def __init__(self):
        self.drones = []
        self.mines_truth = []
        self.mines_detected = []
        self.trees = []
        self.safe_path = []
        self.elapsed = 0.0
        self.tick = 0
        self.last_alloc_tick = -999
        self.plan_interval = 0.5
        self.plan_elapsed = 0.0
        self.last_allocations = {}
        self.world_waypoints = []
        self.best_world_waypoints = []
        self.best_score = None
        self.best_valid_world_waypoints = []
        self.best_valid_score = None
        self.best_valid_corridor_width = 0
        self.best_corridor_width = 0

        # Start drones near left edge, spread vertically around middle (y=40)
        for i in range(cp.NUM_DRONES):
            start_y = 30.0 + (i * 5.0)  # y: 30, 35, 40, 45 (near path start at y=41)
            self.drones.append(Drone(i, 3.0, start_y))
            
        self.generate_map()

    def generate_map(self):
        self.trees = []
        self.mines_truth = []
        self.mines_detected = []
        self.elapsed = 0.0
        self.tick = 0
        self.last_alloc_tick = -999
        self.plan_elapsed = 0.0
        self.last_allocations = {}
        self.world_waypoints = []
        self.best_world_waypoints = []
        self.best_score = None
        self.best_valid_world_waypoints = []
        self.best_valid_score = None
        self.best_valid_corridor_width = 0
        self.best_corridor_width = 0
        
        # Mines (copied logic)
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(5, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(1, cp.ARENA_HEIGHT_FT - 1)
            self.mines_truth.append([mx, my])

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
        Plan paths using greedy_bottleneck algorithm.
        Drones fly left to right (along X axis = columns).
        """
        belief_grid = self.get_belief_grid()
        # For planning, treat unknown as traversable but track it in `unknown_mask`.
        unknown_mask = [[cell < 0 for cell in row] for row in belief_grid]
        grid_for_plan = [[0 if cell < 0 else cell for cell in row] for row in belief_grid]

        num_rows = int(cp.ARENA_HEIGHT_FT / 2)   # 40
        num_cols = int(cp.ARENA_WIDTH_FT / 2)    # 150

        # Start: current lead drone position (avoids initial "move left" correction)
        lead = self.drones[0]
        start_row = int(lead.pos[1] / 2)
        start_col = int(lead.pos[0] / 2)
        start = (max(0, min(num_rows - 1, start_row)), max(0, min(num_cols - 1, start_col)))

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
        self.corridor_width = plan.g_width

        # Convert grid path to world coordinates (feet) for drones
        world_waypoints = []
        for (row, col) in plan.path:
            world_x = col * 2 + 1  # center of cell in feet
            world_y = row * 2 + 1
            world_waypoints.append((world_x, world_y))
        self.world_waypoints = world_waypoints

        # Track current path for visualization (updates every replan).
        truth_grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        for mx, my in self.mines_truth:
            col = int(mx / 2)
            row = int(my / 2)
            if 0 <= row < num_rows and 0 <= col < num_cols:
                truth_grid[row][col] = 1
        path_for_score = self._ensure_4_connected(plan.path)
        commands = PathPlan(plan.start, path_for_score, plan.g_width, 0, 0).to_commands()
        report = score_path(truth_grid, commands, a_minutes=min(self.elapsed / 60.0, 7.0))
        self.best_score = report.score
        self.best_world_waypoints = list(world_waypoints)
        self.best_corridor_width = plan.g_width
        if report.fail_reason is None:
            if self.best_valid_score is None or report.score > self.best_valid_score:
                self.best_valid_score = report.score
                self.best_valid_world_waypoints = list(world_waypoints)
                self.best_valid_corridor_width = plan.g_width

        # Print path info (only when path changes)
        if not hasattr(self, '_last_path_len') or self._last_path_len != len(plan.path):
            self._last_path_len = len(plan.path)
            print(f"[Planner] Path: {len(plan.path)} waypoints, G-width: {plan.g_width}, Detected mines: {len(self.mines_detected)}")
            if len(plan.path) <= 10:
                print(f"  Waypoints (ft): {world_waypoints}")
            else:
                print(f"  First 5: {world_waypoints[:5]} ... Last 5: {world_waypoints[-5:]}")

        assignments, self.last_alloc_tick = allocate_receding(
            [DroneState(d.id, (int(d.pos[1] / 2), int(d.pos[0] / 2)), False) for d in self.drones],
            belief_grid,
            plan.path,
            tick=self.tick,
            last_replan_tick=self.last_alloc_tick,
            replan_interval=15,
            min_target_sep=4,
            max_targets=20,
        )
        if assignments:
            new_allocations = {a.drone_id: a.target for a in assignments}
            if new_allocations != self.last_allocations:
                print(f"[Allocator] tick={self.tick} assignments={new_allocations}")
                self.last_allocations = new_allocations
            for drone in self.drones:
                if drone.id in new_allocations:
                    row, col = new_allocations[drone.id]
                    drone.clear_waypoints()
                    drone.add_waypoint(col * 2 + 1, row * 2 + 1)
        else:
            for drone in self.drones:
                if not drone.waypoints:
                    drone.waypoints = list(world_waypoints)

    def update(self, dt):
        self.elapsed += dt
        self.plan_elapsed += dt
        self.tick += 1
        
        # Update Control Parameters from Global Config (if changed dynamically during runtime)
        # Assuming cp.PID_KP etc might change if user edits controlPanel live? 
        # But ordinarily we read once. Let's force update to be safe if user wants real-time tweaking.
        for drone in self.drones:
            drone.set_pid_params(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        
        # Path Planning (receding horizon)
        if self.plan_elapsed >= self.plan_interval:
            self.plan_elapsed = 0.0
            self.plan_paths()
        
        # Physics Update
        for drone in self.drones:
            drone.update_physics(dt)
            
            # Sensing
            for mine in self.mines_truth:
                d = distance(drone.pos, (mine[0], mine[1]))
                if d < cp.DETECTION_RADIUS_FT:
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

        # Drones
        for d in self.drones:
            d.draw(surface, (ox, oy))

        # Current (future) path visualization
        if self.world_waypoints:
            points = []
            for wx, wy in self.world_waypoints:
                sx = ox + (wx * cp.PX_PER_FOOT)
                sy = oy + (wy * cp.PX_PER_FOOT)
                points.append((sx, sy))
            if len(points) > 1:
                width_cells = 1 + 2 * max(0, self.corridor_width)
                width_px = max(1, int(width_cells * 2 * cp.PX_PER_FOOT))
                pygame.draw.lines(surface, cp.Endesga.sebastian_lague_light_purple, False, points, width_px)

        # Historical best path (thicker, different color)
        if self.best_valid_world_waypoints:
            points = []
            for wx, wy in self.best_valid_world_waypoints:
                sx = ox + (wx * cp.PX_PER_FOOT)
                sy = oy + (wy * cp.PX_PER_FOOT)
                points.append((sx, sy))
            if len(points) > 1:
                width_cells = 1 + 2 * max(0, self.best_valid_corridor_width)
                width_px = max(1, int(width_cells * 2 * cp.PX_PER_FOOT))
                pygame.draw.lines(surface, cp.Endesga.network_green, False, points, width_px)

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