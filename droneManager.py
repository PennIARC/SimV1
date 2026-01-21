import pygame
import math
import random
from calcs import distance, normalize_angle
import controlPanel as cp


class Drone:
    def __init__(self, id, start_x, start_y):
        self.id = id
        self.x = float(start_x)
        self.y = float(start_y)
        self.angle = 0.0  # Radians
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.active = True

    def move(self, move_cmd, turn_cmd, dt):
        """
        move_cmd: 0 (stop) or 1 (full speed)
        turn_cmd: -1 (left), 0 (none), 1 (right)
        """
        if not self.active: return

        # Turning
        if turn_cmd != 0:
            self.angle += turn_cmd * cp.TURN_RATE_RAD * dt
            self.angle = normalize_angle(self.angle)

        # Movement
        current_speed = 0.0
        if move_cmd > 0:
            current_speed = cp.MAX_SPEED_FT

        self.vel_x = math.cos(self.angle) * current_speed
        self.vel_y = math.sin(self.angle) * current_speed

        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

        # Arena Bounds Clamping
        self.x = max(0, min(cp.ARENA_WIDTH_FT, self.x))
        self.y = max(0, min(cp.ARENA_HEIGHT_FT, self.y))

    def draw(self, surface, arena_offset):
        # Calculate screen coordinates based on the arena's top-left offset
        sx = (self.x * cp.PX_PER_FOOT) + arena_offset[0]
        sy = (self.y * cp.PX_PER_FOOT) + arena_offset[1]

        # Draw Detection Radius (Visual Feedback)
        # Using a transparent surface for the circle to avoid messing up underlying pixels too much
        # But for simplicity in this low-res engine, direct circle drawing with alpha color works if surface supports it
        # However, to be safe with standard pygame primitives:
        pygame.draw.circle(surface, cp.Endesga.sebastian_lague_light_purple,
                           (int(sx), int(sy)), int(cp.DETECTION_RADIUS_FT * cp.PX_PER_FOOT), 1)

        # Draw Drone (Arrowhead shape)
        size = cp.VISUAL_DRONE_SIZE * cp.PX_PER_FOOT

        # Points calculation
        tip = (sx + math.cos(self.angle) * size, sy + math.sin(self.angle) * size)
        left_wing = (sx + math.cos(self.angle + 2.6) * size, sy + math.sin(self.angle + 2.6) * size)
        right_wing = (sx + math.cos(self.angle - 2.6) * size, sy + math.sin(self.angle - 2.6) * size)
        back_indent = (sx - math.cos(self.angle) * (size * 0.5), sy - math.sin(self.angle) * (size * 0.5))

        points = [tip, left_wing, back_indent, right_wing]
        pygame.draw.polygon(surface, cp.Endesga.white, points)


class DroneHandler:
    def __init__(self):
        self.drones = []
        self.mines_truth = []  # All mines [x, y]
        self.mines_detected = []  # Mines seen by drones [x, y]
        self.trees = []  # Obstacles [x, y, radius]
        self.safe_path = []  # List of (x,y) tuples
        self.elapsed = 0.0

        # Initialize Drones stacked at start
        for i in range(cp.NUM_DRONES):
            self.drones.append(Drone(i, 5.0, 10.0 + (i * 10.0)))

        self.generate_map()

    def generate_map(self):
        self.trees = []
        self.mines_truth = []
        self.mines_detected = []
        self.elapsed = 0.0


        # ignoring trees for now

        # # 1. Random obstacles (Trees)
        # for _ in range(15):
        #     tx = random.uniform(20, cp.ARENA_WIDTH_FT - 20)
        #     ty = random.uniform(5, cp.ARENA_HEIGHT_FT - 5)
        #     tr = random.uniform(2.0, 6.0)  # Radius in ft
        #     self.trees.append([tx, ty, tr])

        # 2. Random Mines
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(5, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(1, cp.ARENA_HEIGHT_FT - 1)

            # Don't spawn inside trees
            valid = True
            for t in self.trees:
                if distance((mx, my), (t[0], t[1])) < (t[2] + 2.0):
                    valid = False
                    break

            if valid:
                self.mines_truth.append([mx, my])

    def update(self, dt):
        self.elapsed += dt

        for drone in self.drones:
            # Temporary "Bounce" AI
            move = 1
            turn = 0
            if drone.y < 5:
                turn = 1
            elif drone.y > cp.ARENA_HEIGHT_FT - 5:
                turn = -1
            elif drone.x > cp.ARENA_WIDTH_FT - 5:
                turn = 1
            elif drone.x < 2:
                turn = 0

            drone.move(move, turn, dt)

            # Sensing
            for mine in self.mines_truth:
                d = distance((drone.x, drone.y), (mine[0], mine[1]))
                if d < cp.DETECTION_RADIUS_FT:
                    if mine not in self.mines_detected:
                        self.mines_detected.append(mine)

    def draw(self, surface, offset=(0, 0)):
        ox, oy = offset

        # Draw Arena Background
        rect = pygame.Rect(ox, oy, cp.ARENA_WIDTH_FT * cp.PX_PER_FOOT, cp.ARENA_HEIGHT_FT * cp.PX_PER_FOOT)
        pygame.draw.rect(surface, cp.BACKGROUND_COLOR, rect)
        pygame.draw.rect(surface, cp.Endesga.grey_blue, rect, 2)  # Border

        # Draw Grid Lines
        for x_ft in range(0, int(cp.ARENA_WIDTH_FT) + 1, cp.GRID_LINE_SPACING):
            lx = ox + (x_ft * cp.PX_PER_FOOT)
            pygame.draw.line(surface, cp.Endesga.my_blue, (lx, rect.top), (lx, rect.bottom), 1)

        # Draw Trees
        for t in self.trees:
            cx = ox + (t[0] * cp.PX_PER_FOOT)
            cy = oy + (t[1] * cp.PX_PER_FOOT)
            rad = t[2] * cp.PX_PER_FOOT
            pygame.draw.circle(surface, cp.Endesga.dark_green, (int(cx), int(cy)), int(rad))
            pygame.draw.circle(surface, cp.Endesga.network_green, (int(cx), int(cy)), int(rad * 0.7))

        # Draw Mines (Truth - Faint)
        for m in self.mines_truth:
            if m not in self.mines_detected:
                mx = ox + (m[0] * cp.PX_PER_FOOT)
                my = oy + (m[1] * cp.PX_PER_FOOT)
                pygame.draw.circle(surface, (40, 45, 55), (int(mx), int(my)), 1)

        # Draw Mines (Detected - Bright)
        for m in self.mines_detected:
            mx = ox + (m[0] * cp.PX_PER_FOOT)
            my = oy + (m[1] * cp.PX_PER_FOOT)
            pygame.draw.circle(surface, cp.Endesga.network_red, (int(mx), int(my)), 2)

        # Draw Drones
        for d in self.drones:
            d.draw(surface, (ox, oy))