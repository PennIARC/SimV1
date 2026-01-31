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
            if confidence_map[sy][sx] <= CONFIDENCE_THRESHOLD:
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
                            if confidence_map[ny][nx] <= CONFIDENCE_THRESHOLD:
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
                    if confidence_map[y][x] <= CONFIDENCE_THRESHOLD:
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
                if confidence_map[ny][nx] <= CONFIDENCE_THRESHOLD:
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
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
        mines_detected_high: List[Tuple[int, int]],
        high_advance_steps: int = 3,
        sensing_radius_high: int = 3,
        sensing_radius_low: int = 1,
    ) -> List[Tuple[int, int]]:
        """
        Suggest 4 deterministic exploration targets for the four UAVs.

        Order: [UAV0, UAV1, UAV2, UAV3]
        - UAV0/UAV1: high-alt scanners (upper / lower halves). Advance one column
          toward the goal-side each timestep.
        - UAV2/UAV3: low-alt inspectors around a path anchor chosen near the
          goal that prefer low mine-density and high clearance.

        Deterministic, grid-aligned, and avoids assigning duplicate points.
        """
        NUM_DRONES = 4

        # Basic map geometry
        H = len(clearance_map)
        W = len(clearance_map[0]) if H > 0 else 0

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < W and 0 <= y < H

        def is_safe(x: int, y: int) -> bool:
            return in_bounds(x, y) and (clearance_map[y][x] > 0)

        # Deterministic 1D offsets: 0, +1, -1, +2, -2, ...
        def offsets_1d(max_off: int):
            out = [0]
            for d in range(1, max_off + 1):
                out.append(d)
                out.append(-d)
            return out

        # Estimate mine density (count within Euclidean radius)
        def estimate_mine_density(x: int, y: int, radius: int) -> int:
            r2 = radius * radius
            cnt = 0
            for mx, my in mines_detected_high:
                dx = mx - x
                dy = my - y
                if dx * dx + dy * dy <= r2:
                    cnt += 1
            return cnt

        # Handle empty/invalid map
        if H == 0 or W == 0:
            return [(0, 0)] * NUM_DRONES

        # Determine anchor on path: prefer point closest to goal (via heuristic)
        if path:
            best_idx = 0
            best_dist = math.inf
            for i, (px, py) in enumerate(path):
                d = self.heuristic_distance_to_goal(px, py)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            anchor_x, anchor_y = path[best_idx]
        else:
            # fallback to persistent best if present, else map center
            if self.persistent_best:
                pb = self.persistent_best
                anchor_x, anchor_y = pb[-1]
            else:
                anchor_x, anchor_y = (W // 2, H // 2)

        # Determine goal-side direction using goal_cells: move one column toward
        # the average goal x (deterministic). If no goal info, advance right.
        if self.goal_cells:
            goal_x = sum(gx for gx, _ in self.goal_cells) / max(1, len(self.goal_cells))
        else:
            goal_x = W - 1

        # Advance toward goal-side by configurable number of columns per timestep
        if goal_x > anchor_x:
            next_x = min(W - 1, anchor_x + max(1, high_advance_steps))
        elif goal_x < anchor_x:
            next_x = max(0, anchor_x - max(1, high_advance_steps))
        else:
            next_x = anchor_x

        # High-alt UAVs: UAV0 scans upper half, UAV1 scans lower half.
        mid = H // 2

        def find_safe_in_column(col_x: int, y_min: int, y_max: int):
            # Prefer rows near band center, deterministic offsets
            band_center = (y_min + y_max) // 2
            vert_offs = offsets_1d(max(1, (y_max - y_min) // 2 + 2))
            horiz_offs = offsets_1d(min(6, W - 1))
            # try exact column first then nearby columns deterministically
            for dx in horiz_offs:
                cx = col_x + dx
                if cx < 0 or cx >= W:
                    continue
                for oy in vert_offs:
                    cy = band_center + oy
                    if cy < y_min or cy > y_max:
                        continue
                    if is_safe(cx, cy):
                        return (cx, cy)
            return None

        u0 = find_safe_in_column(next_x, 0, max(0, mid - 1))
        u1 = find_safe_in_column(next_x, mid, H - 1)
        
        # u0 = (next_x, (mid - 1)//2)
        # u1 = (next_x, mid + (H - 1 - mid)//2)

        # If either not found, expand search deterministically across columns
        if u0 is None:
            for dx in offsets_1d(W - 1):
                cx = min(W - 1, max(0, next_x + dx))
                for y in range(0, max(0, mid)):
                    if is_safe(cx, y):
                        u0 = (cx, y)
                        break
                if u0:
                    break

        if u1 is None:
            for dx in offsets_1d(W - 1):
                cx = min(W - 1, max(0, next_x + dx))
                for y in range(mid, H):
                    if is_safe(cx, y):
                        u1 = (cx, y)
                        break
                if u1:
                    break

        # As last-resort placeholders (in-bounds) if still None
        if u0 is None:
            u0 = (min(W - 1, next_x), max(0, (0 + max(0, mid - 1)) // 2))
        if u1 is None:
            u1 = (min(W - 1, next_x), min(H - 1, mid + max(0, (H - 1 - mid) // 2)))

        # choose a deterministic reference point for tie-breaking that is NOT the planner path
        if mines_detected_high:
            avg_x = sum(mx for mx, _ in mines_detected_high) / len(mines_detected_high)
            avg_y = sum(my for _, my in mines_detected_high) / len(mines_detected_high)
            ref_x, ref_y = int(round(avg_x)), int(round(avg_y))
        else:
            ref_x, ref_y = W // 2, H // 2

        # Low-alt UAVs: select targets based ONLY on high-alt detections
        # Prefer cells that have been seen by high-alt (confidence >= 0.5),
        # with low estimated mine density (w.r.t. high-alt detections) and high clearance.
        candidates = []
        # Precompute distance-to-path for each candidate (if path available)
        def dist_to_path(px, py):
            if not path:
                return abs(px - anchor_x) + abs(py - anchor_y)
            best = math.inf
            for (axp, ayp) in path:
                d = abs(px - axp) + abs(py - ayp)
                if d < best:
                    best = d
            return best

        for cy in range(H):
            for cx in range(W):
                # only consider cells observed by high-alt (confidence >= 0.5)
                if confidence_map[cy][cx] < 0.5:
                    continue
                if clearance_map[cy][cx] <= 0:
                    continue
                if (cx, cy) == u0 or (cx, cy) == u1:
                    continue
                density = estimate_mine_density(cx, cy, sensing_radius_low)
                # distance to goal (use planner heuristic)
                dist_goal = self.heuristic_distance_to_goal(cx, cy)
                # distance to current greedy path (Manhattan)
                dist_path = dist_to_path(cx, cy)
                # prefer lower density, then closer to goal, then closer to path, then higher clearance
                clearance = clearance_map[cy][cx]
                key = (density, dist_goal, dist_path, -clearance, cx, cy)
                candidates.append((key, (cx, cy)))

        # If no candidates from high-alt observations, fall back to any scanned cells
        if not candidates:
            for cy in range(H):
                for cx in range(W):
                    if clearance_map[cy][cx] <= 0:
                        continue
                    if (cx, cy) == u0 or (cx, cy) == u1:
                        continue
                    density = estimate_mine_density(cx, cy, sensing_radius_low)
                    clearance = clearance_map[cy][cx]
                    manh = abs(cx - ref_x) + abs(cy - ref_y)
                    key = (density, -clearance, manh, cx, cy)
                    candidates.append((key, (cx, cy)))
                    dist_goal = self.heuristic_distance_to_goal(cx, cy)
                    dist_path = dist_to_path(cx, cy)
                    clearance = clearance_map[cy][cx]
                    key = (density, dist_goal, dist_path, -clearance, cx, cy)
                    candidates.append((key, (cx, cy)))

        candidates.sort(key=lambda t: t[0])

        # ref_x, ref_y already computed above

        # Enforce same-x between low-alt UAVs and minimum y-spacing.
        min_spacing = int(max(1, round(sensing_radius_low * 2 - 1)))

        def find_vertical_at_column(x_col, avoid_set, prefer_y=None, require_spacing=None):
            base_y = prefer_y if prefer_y is not None else anchor_y
            for oy in offsets_1d(max(H, (require_spacing or min_spacing) + 2)):
                ty = base_y + oy
                if not in_bounds(x_col, ty):
                    continue
                if clearance_map[ty][x_col] <= 0:
                    continue
                if (x_col, ty) in avoid_set:
                    continue
                # check spacing to points in avoid_set that are low-alt (only y spacing)
                ok = True
                for ax, ay in avoid_set:
                    if abs(ty - ay) < (require_spacing or min_spacing):
                        ok = False
                        break
                if not ok:
                    continue
                return (x_col, ty)
            return None

        u2 = None
        u3 = None
        used = {u0, u1}
        # First pass: pick u2 as first candidate, try to pick u3 with same x and sufficient spacing
        for _, (cx, cy) in candidates:
            if u2 is None:
                u2 = (cx, cy)
                used.add(u2)
                continue
            # try to pick u3 with same x and spacing
            if u3 is None and cx == u2[0] and abs(cy - u2[1]) >= min_spacing:
                u3 = (cx, cy)
                used.add(u3)
                break

        # If u2 found but no same-x partner yet, search vertically in that column
        if u2 is not None and u3 is None:
            partner = find_vertical_at_column(u2[0], {u0, u1, u2}, prefer_y=u2[1], require_spacing=min_spacing)
            if partner is not None:
                u3 = partner
                used.add(u3)


        # Fill missing low-alt UAVs deterministically near the high-alt observed region
        def fill_nearby(used_set):
            for d in range(0, max(W, H)):
                for ox in offsets_1d(d):
                    for oy in offsets_1d(d):
                        x = ref_x + ox
                        y = ref_y + oy
                        if not in_bounds(x, y):
                            continue
                        if clearance_map[y][x] <= 0:
                            continue
                        if (x, y) in used_set:
                            continue
                        used_set.add((x, y))
                        return (x, y)
            # last resort: anchor or map-center
            if is_safe(anchor_x, anchor_y) and (anchor_x, anchor_y) not in used_set:
                used_set.add((anchor_x, anchor_y))
                return (anchor_x, anchor_y)
            cx = W // 2
            cy = H // 2
            used_set.add((cx, cy))
            return (cx, cy)

        if u2 is None:
            u2 = fill_nearby(used)
        if u3 is None:
            u3 = fill_nearby(used)

        # Ensure all waypoints distinct; if collision, fix deterministically
        waypoints = [u0, u1, u2, u3]
        seen = set()
        for i, wp in enumerate(waypoints):
            if wp in seen:
                # replace with a nearby free cell
                replacement = fill_nearby(seen)
                waypoints[i] = replacement
                seen.add(replacement)
            else:
                seen.add(wp)

        # Final clamp to int and in-bounds
        final = []
        for x, y in waypoints:
            ix = int(min(max(0, x), W - 1))
            iy = int(min(max(0, y), H - 1))
            final.append((ix, iy))

        return final

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
