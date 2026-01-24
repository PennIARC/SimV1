#####
# Pointer-free greedy bottleneck planner
# Mirrors the behavior of greedy_path_planning.py but avoids using
# `State` objects or pointer-style backreferences on heap entries.
#####

import heapq
import math
from typing import List, Tuple, Optional, Dict

# Minimum required clearance (ft) for a cell to be considered traversable
CLEARANCE_THRESHOLD = 1.0


class GreedyBottleneckPlanner:
    """
    Greedy bottleneck-preserving path planner without pointer-style State objects.
    API matches the original planner: constructor, `plan(...)`,
    and `suggest_exploration_targets(...)`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        start_cells: List[Tuple[int, int]],
        goal_cells: List[Tuple[int, int]],
    ):
        self.height = height
        self.width = width
        self.start_cells = start_cells
        self.goal_cells = set(goal_cells)

        # parent grid: for each cell store previous cell as (x,y) or None
        self.parent: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

        # persistent best path and its bottleneck
        self.persistent_best: Optional[List[Tuple[int, int]]] = None
        self.persistent_bottleneck: float = -math.inf

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield nx, ny

    def heuristic_distance_to_goal(self, x: int, y: int) -> float:
        return min(abs(x - gx) + abs(y - gy) for gx, gy in self.goal_cells)

    def reconstruct_path_from_grid(self, x: int, y: int) -> List[Tuple[int, int]]:
        path = []
        bx, by = x, y
        while True:
            path.append((bx, by))
            p = self.parent[by][bx]
            if p is None:
                break
            bx, by = p
        return path[::-1]

    def _better_partial(self, ax: int, ay: int, aw: float, b: Optional[Tuple[int, int, float]]) -> bool:
        """Prefer partial candidates that are closer to goal, then larger bottleneck."""
        if b is None:
            return True
        bx, by, bw = b
        ha = self.heuristic_distance_to_goal(ax, ay)
        hb = self.heuristic_distance_to_goal(bx, by)
        if ha != hb:
            return ha < hb
        if aw != bw:
            return aw > bw
        return True

    def plan(self, confidence_map: List[List[bool]], clearance_map: List[List[float]]) -> Optional[Dict]:
        best_bottleneck = [[-math.inf for _ in range(self.width)] for _ in range(self.height)]
        self.parent = [[None for _ in range(self.width)] for _ in range(self.height)]
        open_set = []  # heap of (-bottleneck, heuristic, x, y, bottleneck)

        # Seed from start cells
        for sx, sy in self.start_cells:
            if confidence_map[sy][sx] <= 0:
                continue
            if clearance_map[sy][sx] <= CLEARANCE_THRESHOLD:
                continue
            best_bottleneck[sy][sx] = clearance_map[sy][sx]
            self.parent[sy][sx] = None
            heapq.heappush(open_set, (-best_bottleneck[sy][sx], self.heuristic_distance_to_goal(sx, sy), sx, sy, best_bottleneck[sy][sx]))

        # Fallback seeding
        if not open_set:
            for sx, sy in self.start_cells:
                found = False
                for r in range(1, 7):
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            nx, ny = sx + dx, sy + dy
                            if not self.in_bounds(nx, ny):
                                continue
                            if confidence_map[ny][nx] <= 0:
                                continue
                            if clearance_map[ny][nx] <= CLEARANCE_THRESHOLD:
                                continue
                            if best_bottleneck[ny][nx] > -math.inf:
                                continue
                            best_bottleneck[ny][nx] = clearance_map[ny][nx]
                            self.parent[ny][nx] = None
                            heapq.heappush(open_set, (-best_bottleneck[ny][nx], self.heuristic_distance_to_goal(nx, ny), nx, ny, best_bottleneck[ny][nx]))
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break

        if not open_set:
            seed_cols = min(6, self.width)
            for y in range(self.height):
                for x in range(seed_cols):
                    if confidence_map[y][x] <= 0:
                        continue
                    if clearance_map[y][x] <= CLEARANCE_THRESHOLD:
                        continue
                    if best_bottleneck[y][x] > -math.inf:
                        continue
                    best_bottleneck[y][x] = clearance_map[y][x]
                    self.parent[y][x] = None
                    heapq.heappush(open_set, (-best_bottleneck[y][x], self.heuristic_distance_to_goal(x, y), x, y, best_bottleneck[y][x]))

        best_partial: Optional[Tuple[int, int, float]] = None

        while open_set:
            _, _, cx, cy, cw = heapq.heappop(open_set)

            if self._better_partial(cx, cy, cw, best_partial):
                best_partial = (cx, cy, cw)

            if (cx, cy) in self.goal_cells:
                path = self.reconstruct_path_from_grid(cx, cy)
                return {"path": path, "bottleneck": int(cw), "length": len(path), "reached": True}

            for nx, ny in self.neighbors(cx, cy):
                if confidence_map[ny][nx] <= 0:
                    continue
                if clearance_map[ny][nx] <= CLEARANCE_THRESHOLD:
                    continue

                new_w = min(cw, clearance_map[ny][nx])
                if new_w <= best_bottleneck[ny][nx]:
                    continue

                best_bottleneck[ny][nx] = new_w
                self.parent[ny][nx] = (cx, cy)
                heapq.heappush(open_set, (-new_w, self.heuristic_distance_to_goal(nx, ny), nx, ny, new_w))

        if best_partial is None:
            return None

        bx, by, bw = best_partial
        path = self.reconstruct_path_from_grid(bx, by)
        self._update_persistent(path, bw, clearance_map)

        return {"path": path, "bottleneck": bw, "length": len(path), "reached": False, "persistent_path": self.persistent_best}

    def _update_persistent(self, path: List[Tuple[int, int]], bottleneck: float, clearance_map: List[List[float]]):
        if not path:
            return

        last_safe = -1
        for i, (x, y) in enumerate(path):
            if clearance_map[y][x] > CLEARANCE_THRESHOLD:
                last_safe = i
            else:
                break

        if last_safe < 0:
            return

        safe_prefix = path[: last_safe + 1]

        end_x, end_y = safe_prefix[-1]
        end_dist = self.heuristic_distance_to_goal(end_x, end_y)

        if self.persistent_best is None:
            self.persistent_best = safe_prefix
            self.persistent_bottleneck = bottleneck
            return

        cur_end_x, cur_end_y = self.persistent_best[-1]
        cur_end_dist = self.heuristic_distance_to_goal(cur_end_x, cur_end_y)

        # Primary: closer to goal (smaller distance).
        # Secondary: larger bottleneck. Tertiary: longer safe prefix.
        if end_dist < cur_end_dist:
            self.persistent_best = safe_prefix
            self.persistent_bottleneck = bottleneck
            return

        if end_dist == cur_end_dist:
            if bottleneck > self.persistent_bottleneck:
                self.persistent_best = safe_prefix
                self.persistent_bottleneck = bottleneck
                return

            if bottleneck == self.persistent_bottleneck and len(safe_prefix) > len(self.persistent_best):
                self.persistent_best = safe_prefix
                self.persistent_bottleneck = bottleneck

    def suggest_exploration_targets(
        self,
        path: List[Tuple[int, int]],
        confidence_map: List[List[bool]],
        num_drones: int = 4,
        radius: int = 5,
    ) -> List[Tuple[int, int]]:
        active_path = path
        if self.persistent_best is not None:
            if path is None or len(self.persistent_best) > len(path):
                active_path = self.persistent_best
        elif path is None or len(path) == 0:
            if self.persistent_best is None:
                return [None] * num_drones
            active_path = self.persistent_best

        candidates = []

        best_idx = 0
        best_dist = math.inf
        for i, (px, py) in enumerate(active_path):
            d = self.heuristic_distance_to_goal(px, py)
            if d < best_dist:
                best_dist = d
                best_idx = i

        cx, cy = active_path[best_idx]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if not self.in_bounds(nx, ny):
                    continue
                if confidence_map[ny][nx]:
                    continue
                candidates.append((nx, ny))

        def score(cell):
            x, y = cell
            return abs(x - cx) + abs(y - cy)

        candidates = sorted(set(candidates), key=score)

        if candidates:
            target = candidates[0]
        else:
            mid_idx = max(1, min(len(active_path) - 1, best_idx))
            target = active_path[mid_idx]

        waypoints = [None] * num_drones
        waypoints[0] = target
        return waypoints

    def fixed_targets(self, num_drones: int) -> List[Tuple[int, int]]:
        """Return `num_drones` targets evenly spaced along the height,
        placed at the rightmost column so drones fly left->right.
        """
        if num_drones <= 0:
            return []

        if num_drones == 1:
            ys = [self.height // 2]
        else:
            ys = [int(round(i * (self.height - 1) / (num_drones - 1))) for i in range(num_drones)]

        tx = self.width - 1
        return [(tx, y) for y in ys]
