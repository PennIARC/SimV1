import pygame
import math
import random
from calcs import distance, normalize_angle
import controlPanel as cp
import numpy as np

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
        self.clearance_map = None
        self.confidence_map = None
        self.safe_truth = []
        self.safe_detected = []
        self.elapsed = 0.0

        # Initialize drones evenly along the left (start) edge using arena height
        if cp.NUM_DRONES == 1:
            start_ys = [cp.ARENA_HEIGHT_FT / 2.0]
        else:
            start_ys = [i * (cp.ARENA_HEIGHT_FT) / (cp.NUM_DRONES - 1) for i in range(cp.NUM_DRONES)]

        for i, y in enumerate(start_ys):
            # place at leftmost column (x=0.0)
            self.drones.append(Drone(i, 0.0, float(y)))
            
        self.generate_map()

    def generate_map(self):
        self.trees = []
        self.mines_truth = []
        self.mines_detected = []
        self.safe_truth = []
        self.safe_detected = []
        self.elapsed = 0.0
        
        # Mines (copied logic)
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(5, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(1, cp.ARENA_HEIGHT_FT - 1)
            self.mines_truth.append([mx, my])

        # Generate safe_truth as all integer grid cells not occupied by a mine (rounded)
        w = int(cp.ARENA_WIDTH_FT) + 1
        h = int(cp.ARENA_HEIGHT_FT) + 1
        mine_cells = set((int(round(mx)), int(round(my))) for mx, my in self.mines_truth)
        for ix in range(w):
            for iy in range(h):
                if (ix, iy) not in mine_cells:
                    self.safe_truth.append([float(ix), float(iy)])

    def stop_all_drones(self):
        """Immediately stop all drones: clear waypoints, zero velocities/accelerations, reset controllers."""
        for d in self.drones:
            try:
                d.clear_waypoints()
                d.vel = [0.0, 0.0]
                d.acc = [0.0, 0.0]
                d.pid_x.reset()
                d.pid_y.reset()
                d.active = False
            except Exception:
                pass

    def plan_paths(self, waypoints=None):
        """
        Assign next-step waypoints for drones.
        If `waypoints` is None or empty, falls back to per-drone random waypoints
        (and preserves existing queued waypoints). If `waypoints` is provided,
        it should be an iterable of (x, y) pairs (one per drone). A None entry
        for a specific drone means "no external waypoint for that drone".
        """
        if waypoints:
            for i, drone in enumerate(self.drones):
                if i < len(waypoints) and waypoints[i] is not None:
                    tx, ty = waypoints[i]
                    # Only replace the current waypoint if it differs significantly
                    if drone.waypoints:
                        cur_tx, cur_ty = drone.waypoints[0]
                        if abs(cur_tx - tx) < 1e-3 and abs(cur_ty - ty) < 1e-3:
                            # same waypoint as already queued: keep existing to avoid resetting PID/vel
                            continue
                    # new/different waypoint: replace queue (this resets PID/vel as intended)
                    drone.clear_waypoints()
                    drone.add_waypoint(float(tx), float(ty))
                else:
                    if not drone.waypoints:
                        tx = random.uniform(5.0, cp.ARENA_WIDTH_FT)
                        ty = random.uniform(5.0, cp.ARENA_HEIGHT_FT)
                        drone.add_waypoint(tx, ty)
        else:
            # preserve previous behavior when no external waypoints provided
            for drone in self.drones:
                if not drone.waypoints:
                    tx = random.uniform(5.0, cp.ARENA_WIDTH_FT)
                    ty = random.uniform(5.0, cp.ARENA_HEIGHT_FT)
                    drone.add_waypoint(tx, ty)

    def update(self, dt, waypoints=None):
        """
        Progress simulation by `dt` seconds.
        If `waypoints` is provided it should be a list/iterable of (x,y)
        pairs (one per-drone) which will be forwarded to `plan_paths`.
        """
        self.elapsed += dt

        # Update Control Parameters from Global Config (if changed dynamically during runtime)
        for drone in self.drones:
            drone.set_pid_params(cp.PID_KP, cp.PID_KI, cp.PID_KD)

        # Path Planning (may accept external per-drone next-step waypoints)
        self.plan_paths(waypoints)

        # Physics Update
        for drone in self.drones:
            drone.update_physics(dt)

            # Sensing
            for mine in self.mines_truth:
                d = distance(drone.pos, (mine[0], mine[1]))
                if d < cp.DETECTION_RADIUS_FT:
                    if mine not in self.mines_detected:
                        self.mines_detected.append(mine)
            
            for safe in self.safe_truth:
                d = distance(drone.pos, (safe[0], safe[1]))
                if d < cp.DETECTION_RADIUS_FT:
                    if safe not in self.safe_detected:
                        self.safe_detected.append(safe)

        # Recompute clearance map after sensing updates
        self.compute_clearance_map()

    def compute_clearance_map(self):
        """
        Compute and store self.clearance_map, confidence_map.

        self.clearance_map with semantics:
          -1.0 = unknown
           0.0 = mine
          >0.0 = distance (ft) to nearest detected mine
        Grid is one cell per foot (0..ARENA_WIDTH_FT, 0..ARENA_HEIGHT_FT).
        """
        w = int(cp.ARENA_WIDTH_FT) + 1
        h = int(cp.ARENA_HEIGHT_FT) + 1

        clearance_map = np.full((h, w), -1.0, dtype=np.float32)
        # confidence_map = np.full((h, w), 1.0, dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        # Mark detected mines
        for mx, my in self.mines_detected:
            ix = int(round(mx))
            iy = int(round(my))
            if 0 <= ix < w and 0 <= iy < h:
                clearance_map[iy, ix] = 0.0
                confidence_map[iy, ix] = 1.0

        # Mark scanned areas
        max_clear = float(w + h)
        for sx, sy in self.safe_detected:
            ix = int(round(sx))
            iy = int(round(sy))
            if 0 <= ix < w and 0 <= iy < h:
                # compute clearance as distance to nearest true mine
                if len(self.mines_detected) > 0:
                    tm = np.array(self.mines_detected)
                    # Euclidean distance to nearest detected mine
                    distances = np.sqrt((tm[:, 0] - ix) ** 2 + (tm[:, 1] - iy) ** 2)
                    clearance_map[iy, ix] = float(distances.min())
                    # print(clearance_map[iy, ix])
                else:
                    # scanned and no detected mines => very large clearance
                    clearance_map[iy, ix] = max_clear
                confidence_map[iy, ix] = 1.0
                               
        self.clearance_map = clearance_map
        self.confidence_map = confidence_map

        return confidence_map, clearance_map

    def get_confidence_and_clearance_maps(self):
        """
        Returns (confidence_map, clearance_map).
        Computes them if not already available.
        """
        if getattr(self, "clearance_map", None) is None or getattr(self, "confidence_map", None) is None:
            return self.compute_clearance_map()
        else:
            return self.confidence_map, self.clearance_map
    
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