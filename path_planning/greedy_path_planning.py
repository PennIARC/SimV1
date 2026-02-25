#####
# Pointer-free greedy bottleneck planner
# Mirrors the behavior of greedy_path_planning.py but avoids using
# `State` objects or pointer-style backreferences on heap entries.
#####

import heapq
import math
from typing import List, Tuple, Optional, Dict

# Minimum required clearance (grid units) for a cell to be considered traversable.
# 0.5 allows any non-mine cell (competition rules: can walk adjacent to mines).
CLEARANCE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


class GreedyBottleneckPlanner:
    """
    Greedy bottleneck-preserving path planner.
    - Uses a best-first search to maximize the minimum clearance (bottleneck)
      along the path from start to goal.
    - plan(confidence_map, clearance_map) returns the best currently safe path.
    - Maintains persistent_best (a safe prefix) for cases where goal is not reachable.
    - scan_front_x is used by the exploration module (monotonic scan front).
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

        # parent[y][x] = (px, py) backpointer for path reconstruction
        self.parent: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

        # persistent best safe prefix across replans
        self.persistent_best: Optional[List[Tuple[int, int]]] = None
        self.persistent_bottleneck: float = -math.inf
        


    # Grid helpers
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
        path: List[Tuple[int, int]] = []
        cx, cy = x, y
        while True:
            path.append((cx, cy))
            p = self.parent[cy][cx]
            if p is None:
                break
            cx, cy = p
        path.reverse()
        return path

    # Selection helpers
    def _is_traversable(
        self,
        x: int,
        y: int,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> bool:
        return (
            confidence_map[y][x] > CONFIDENCE_THRESHOLD
            and clearance_map[y][x] > CLEARANCE_THRESHOLD
        )

    def _better_partial(
        self,
        ax: int, ay: int, aw: float,
        b: Optional[Tuple[int, int, float]],
    ) -> bool:
        """
        Decide whether (ax, ay, aw) is a better "current best" frontier than b.

        Safety-first online planning:
        - Primary: closer to goal (progress)
        - Secondary: larger bottleneck (safer)
        """
        if b is None:
            return True
        bx, by, bw = b

        # goal distance (smaller is better)
        ha = self.heuristic_distance_to_goal(ax, ay)
        hb = self.heuristic_distance_to_goal(bx, by)
        if ha != hb:
            return ha < hb
        # tie-break by bottleneck (larger is better)
        if aw != bw:
            return aw > bw
        
        return True

    def _seed_positions(
        self,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> List[Tuple[int, int]]:
        """
        Return a list of seed positions to initialize the search.

        Priority:
        1) Any traversable start_cells
        2) Traversable cells within small radius around each start_cell
        3) Traversable cells in the first few columns (near start boundary)
        """
        seeds: List[Tuple[int, int]] = []

        # (1) direct start cells
        for sx, sy in self.start_cells:
            if self.in_bounds(sx, sy) and self._is_traversable(sx, sy, confidence_map, clearance_map):
                seeds.append((sx, sy))

        if seeds:
            return seeds

        # (2) local neighborhood around starts (deterministic expanding squares)
        for sx, sy in self.start_cells:
            if not self.in_bounds(sx, sy):
                continue
            for r in range(1, 7):
                found = False
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        x, y = sx + dx, sy + dy
                        if not self.in_bounds(x, y):
                            continue
                        if self._is_traversable(x, y, confidence_map, clearance_map):
                            seeds.append((x, y))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if seeds:
            return seeds

        # (3) first few columns
        seed_cols = min(6, self.width)
        for y in range(self.height):
            for x in range(seed_cols):
                if self._is_traversable(x, y, confidence_map, clearance_map):
                    seeds.append((x, y))

        return seeds

    # Persistent best update
    def _update_persistent(
        self,
        path: List[Tuple[int, int]],
        bottleneck: float,
        clearance_map: List[List[float]],
    ) -> None:
        """
        Keep a persistent safe prefix that is:
        - primarily: closer to goal,
        - secondarily: higher bottleneck,
        - third: longer safe prefix.
        """
        if not path:
            return

        # find longest safe prefix
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

        cur_x, cur_y = self.persistent_best[-1]
        cur_dist = self.heuristic_distance_to_goal(cur_x, cur_y)

        # compare (end_dist, -bottleneck, -len(prefix))
        better = (
            (end_dist < cur_dist)
            or (end_dist == cur_dist and bottleneck > self.persistent_bottleneck)
            or (end_dist == cur_dist and bottleneck == self.persistent_bottleneck and len(safe_prefix) > len(self.persistent_best))
        )
        if better:
            self.persistent_best = safe_prefix
            self.persistent_bottleneck = bottleneck
    
    
    def plan(
        self,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> Optional[Dict]:
        """
        Greedy widest-path style planning over a 4-connected grid.
        Returns either a goal-reaching path or the best current frontier path.

        Output dict contains:
          - path: List[(x, y)]
          - bottleneck: float
          - length: int
          - reached: bool
          - persistent_path: Optional[List[(x,y)]]  (only when reached=False)
        """
        best_bottleneck = [[-math.inf] * self.width for _ in range(self.height)]
        self.parent = [[None] * self.width for _ in range(self.height)]
        # Heap: (-bottleneck, heuristic, x, y, bottleneck)
        # Primary: widest corridor (safest path). Secondary: closer to goal.
        open_set: List[Tuple[float, int, int, int, float]] = []

        # Initialize from seeds
        seeds = self._seed_positions(confidence_map, clearance_map)
        if not seeds:
            return None

        for x, y in seeds:
            w0 = clearance_map[y][x]
            if w0 <= best_bottleneck[y][x]:
                continue
            best_bottleneck[y][x] = w0
            self.parent[y][x] = None
            heapq.heappush(open_set, (-w0, self.heuristic_distance_to_goal(x, y), x, y, w0))

        best_partial: Optional[Tuple[int, int, float]] = None

        # Main loop
        while open_set:
            _, _, cx, cy, cw = heapq.heappop(open_set)

            if self._better_partial(cx, cy, cw, best_partial):
                best_partial = (cx, cy, cw)

            if (cx, cy) in self.goal_cells:
                path = self.reconstruct_path_from_grid(cx, cy)
                return {
                    "path": path,
                    "bottleneck": cw,
                    "length": len(path),
                    "reached": True,
                }

            for nx, ny in self.neighbors(cx, cy):
                if not self._is_traversable(nx, ny, confidence_map, clearance_map):
                    continue

                new_w = min(cw, clearance_map[ny][nx] * confidence_map[ny][nx])
                if new_w <= best_bottleneck[ny][nx]:
                    continue

                best_bottleneck[ny][nx] = new_w
                self.parent[ny][nx] = (cx, cy)
                heapq.heappush(open_set, (-new_w, self.heuristic_distance_to_goal(nx, ny), nx, ny, new_w))

        # No goal, return best partial frontier path
        if best_partial is None:
            return None

        bx, by, bw = best_partial
        path = self.reconstruct_path_from_grid(bx, by)
        self._update_persistent(path, bw, clearance_map)

        return {
            "path": path,
            "bottleneck": bw,
            "length": len(path),
            "reached": False,
            "persistent_path": self.persistent_best,
        }





    