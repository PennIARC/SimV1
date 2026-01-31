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
        
        # exploration scan front (monotonic)
        self.scan_front_x: int = -1

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
        open_set: List[Tuple[float, int, int, int, float]] = []  # (-w, h, x, y, w)

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

                new_w = min(cw, clearance_map[ny][nx])
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
        

    def suggest_exploration_targets(
        self,
        path: List[Tuple[int, int]],
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
        mines_detected_high: List[Tuple[int, int]],
        sensing_radius_high: int = 10,
        sensing_radius_low: int = 3,
    ) -> List[Tuple[int, int]]:
        """
        Return 4 deterministic waypoints: [UAV0, UAV1, UAV2, UAV3]

        UAV0/UAV1 (high-alt): scan upper/lower halves, advance a persistent scanning front 
            along width toward goal.
        UAV2/UAV3 (low-alt): choose informative inspection targets using:
            low mine density (from high-alt detections) + close to goal + close to path + high clearance.
        """

        H = len(clearance_map)
        W = len(clearance_map[0]) if H > 0 else 0
        if H == 0 or W == 0:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]

        mine_density_radius = max(1, sensing_radius_high // 2)

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < W and 0 <= y < H

        def safe(x: int, y: int) -> bool:
            return in_bounds(x, y) and clearance_map[y][x] > CLEARANCE_THRESHOLD

        def offsets_1d(max_off: int):
            # 0, +1, -1, +2, -2, ...
            out = [0]
            for d in range(1, max_off + 1):
                out.extend([d, -d])
            return out

        def mine_density(x: int, y: int, r: int) -> int:
            r2 = r * r
            return sum(
                1
                for mx, my in mines_detected_high
                if (mx - x) * (mx - x) + (my - y) * (my - y) <= r2
            )

        # Anchor selection
        if path:
            anchor_x, anchor_y = min(path, key=lambda p: self.heuristic_distance_to_goal(p[0], p[1]))
        elif getattr(self, "persistent_best", None):
            anchor_x, anchor_y = self.persistent_best[-1]
        else:
            anchor_x, anchor_y = (W // 2, H // 2)

        # Goal-side direction & persistent scanning front
        if getattr(self, "goal_cells", None):
            goal_x = sum(gx for gx, _ in self.goal_cells) / max(1, len(self.goal_cells))
        else:
            goal_x = W - 1

        if goal_x > anchor_x:
            desired_x = min(W - 1, anchor_x + 1)
        elif goal_x < anchor_x:
            desired_x = max(0, anchor_x - 1)
        else:
            desired_x = anchor_x

        if self.scan_front_x < 0:
            self.scan_front_x = desired_x
        else:
            self.scan_front_x = max(self.scan_front_x, desired_x)

        scan_x = int(min(max(0, self.scan_front_x), W - 1))
        mid = H // 2

        # High-alt pick: choose a safe cell near band center around scan_x
        def pick_scan_point(x0: int, y0: int, y1: int) -> Tuple[int, int]:
            band_center = (y0 + y1) // 2
            # search near scan_x first, then nearby columns; near band center first
            for dx in offsets_1d(min(6, W - 1)):
                x = x0 + dx
                if not (0 <= x < W):
                    continue
                for dy in offsets_1d(max(1, (y1 - y0) // 2 + 2)):
                    y = band_center + dy
                    if y < y0 or y > y1:
                        continue
                    if safe(x, y):
                        return (x, y)

            # fallback: scan wider columns, any safe cell in band
            for dx in offsets_1d(W - 1):
                x = min(W - 1, max(0, x0 + dx))
                for y in range(y0, y1 + 1):
                    if safe(x, y):
                        return (x, y)

            # last resort: return an in-bounds placeholder (may be unsafe in extreme cases)
            return (x0, band_center)

        u0 = pick_scan_point(scan_x, 0, max(0, mid - 1))
        u1 = pick_scan_point(scan_x, mid, H - 1)

        # Reference point for deterministic fallback fills
        if mines_detected_high:
            ref_x = int(round(sum(mx for mx, _ in mines_detected_high) / len(mines_detected_high)))
            ref_y = int(round(sum(my for _, my in mines_detected_high) / len(mines_detected_high)))
        else:
            ref_x, ref_y = (W // 2, H // 2)

        def dist_to_path(x: int, y: int) -> int:
            if not path:
                return abs(x - anchor_x) + abs(y - anchor_y)
            return min(abs(x - px) + abs(y - py) for px, py in path)

        # Low-alt candidates
        def build_candidates(require_confidence: bool) -> List[Tuple[Tuple, Tuple[int, int]]]:
            out = []
            for y in range(H):
                for x in range(W):
                    if require_confidence and confidence_map[y][x] < 0.5:
                        continue
                    if clearance_map[y][x] <= 0:
                        continue

                    key = (
                        mine_density(x, y, mine_density_radius), # lower is better
                        self.heuristic_distance_to_goal(x, y),   # closer to goal
                        dist_to_path(x, y),                      # closer to path
                        -clearance_map[y][x],                    # higher clearance
                        x, y                                     # deterministic tie-break
                    )
                    out.append((key, (x, y)))
            out.sort(key=lambda t: t[0])
            return out

        candidates = build_candidates(require_confidence=True)
        if not candidates:
            candidates = build_candidates(require_confidence=False)

        # Pick UAV2/UAV3
        min_spacing = int(max(1, round(sensing_radius_low * 2 - 1)))

        def fill_nearby(used: set) -> Tuple[int, int]:
            # deterministic expanding ring around ref
            for d in range(0, max(W, H)):
                for ox in offsets_1d(d):
                    for oy in offsets_1d(d):
                        x, y = ref_x + ox, ref_y + oy
                        if not in_bounds(x, y):
                            continue
                        if clearance_map[y][x] <= 0:
                            continue
                        if (x, y) in used:
                            continue
                        used.add((x, y))
                        return (x, y)

            # hard fallback
            if safe(anchor_x, anchor_y) and (anchor_x, anchor_y) not in used:
                used.add((anchor_x, anchor_y))
                return (anchor_x, anchor_y)
            c = (W // 2, H // 2)
            used.add(c)
            return c

        used = set()
        u2 = u3 = None

        for _, (x, y) in candidates:
            if u2 is None:
                u2 = (x, y)
                used.add(u2)
                continue
            if u3 is None and abs(y - u2[1]) >= min_spacing:
                u3 = (x, y)
                used.add(u3)
                break

        if u2 is None:
            u2 = fill_nearby(used)

        if u3 is None:
            for _, (x, y) in candidates:
                if (x, y) != u2 and abs(y - u2[1]) >= min_spacing:
                    u3 = (x, y)
                    used.add(u3)
                    break
        if u3 is None:
            u3 = fill_nearby(used)

        if u2 == u3:
            u3 = fill_nearby({u2})

        # clamp + ensure ints
        waypoints = [u0, u1, u2, u3]
        return [(int(min(max(0, x), W - 1)), int(min(max(0, y), H - 1))) for x, y in waypoints]


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
