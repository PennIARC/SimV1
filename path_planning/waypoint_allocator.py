"""
Swarm Task Allocator — assigns waypoints for 4 drones.

HIGH-ALT (UAV0/UAV1): persistent scan front, fly in straight lines advancing toward the goal column.
LOW-ALT (UAV2/UAV3): each follows its paired high-alt drone.
    Priority: lowest mine density -> widest corridor -> push right -> follow leader.

Completely decoupled from human path planning.
"""

from typing import List, Tuple

CLEARANCE_THRESHOLD = 0.5

class WaypointAllocator:

    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width
        self.scan_front_x: float = 0.0 

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(
        self,
        confidence_map,
        clearance_map,  
        mines_detected_high: List[Tuple[int, int]],
        drone_positions_grid: List[Tuple[int, int]], # drone real-time positions in grid coordinates
        sensing_radius_high: int,
        sensing_radius_low: int,
    ) -> List[Tuple[int, int]]:
        
        H, W = self.H, self.W
        if H == 0 or W == 0:
            return [(0, 0)] * 4

        mid = H // 2
        mine_density_radius = max(1, sensing_radius_high // 2)

        # --- Helpers ---
        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < W and 0 <= y < H

        def safe(x: int, y: int) -> bool:
            return in_bounds(x, y) and clearance_map[y][x] != 0.0

        def offsets_1d(max_off: int):
            out = [0]
            for d in range(1, max_off + 1):
                out.extend([d, -d])
            return out

        def mine_density(x: int, y: int, r: int) -> int:
            r_ft = r * 2.0
            r2 = r_ft * r_ft
            px = x * 2.0 + 1.0
            py = y * 2.0 + 1.0
            return sum(
                1
                for mx, my in mines_detected_high
                if (mx - px) * (mx - px) + (my - py) * (my - py) <= r2
            )

        # =============================================================
        # 1. HIGH-ALT drones (UAV0/UAV1) advance their scan front
        # =============================================================
        u0_gx = drone_positions_grid[0][0]
        u1_gx = drone_positions_grid[1][0]
        avg_high_x = (u0_gx + u1_gx) / 2.0

        if avg_high_x >= self.scan_front_x - 0.1:
            advance_step = 2
            self.scan_front_x = min(W - 1, self.scan_front_x + advance_step)

        scan_x = int(min(self.scan_front_x, W - 1))

        # =============================================================
        # 2. (HIGH-ALT) choose waypoints
        # =============================================================
        def pick_scan_point(x0: int, y0: int, y1: int) -> Tuple[int, int]:
            band_center = (y0 + y1) // 2
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
            for dx in offsets_1d(W - 1):
                x = min(W - 1, max(0, x0 + dx))
                for y in range(y0, y1 + 1):
                    if safe(x, y):
                        return (x, y)
            return (x0, band_center)

        u0 = pick_scan_point(scan_x, 0, max(0, mid - 1))
        u1 = pick_scan_point(scan_x, mid, H - 1)

        # =============================================================
        # 3. LOW-ALT drones (UAV2/UAV3) choose waypoints
        # =============================================================
        def build_candidates(
            require_confidence: bool,
            y_min: int = 0,
            y_max: int = H - 1,
            hx: int = 0,  
            hy: int = 0,  
        ) -> List[Tuple[tuple, Tuple[int, int]]]:
            out = []
            for y in range(y_min, y_max + 1):
                for x in range(W):
                    if require_confidence and confidence_map[y][x] < 0.5:
                        continue
                    
                    if clearance_map[y][x] == 0.0:  #only skip confirmed mine
                        continue
                    
                    key = (
                        confidence_map[y][x],
                        mine_density(x, y, mine_density_radius),
                        -clearance_map[y][x],
                        -x,
                        abs(x - hx) + abs(y - hy),
                        
                    )
                    out.append((key, (x, y)))
            
            out.sort(key=lambda t: t[0])
            return out

        candidates_upper = build_candidates(True, 0, max(0, mid - 1), u0[0], u0[1])
        if not candidates_upper:
            candidates_upper = build_candidates(False, 0, max(0, mid - 1), u0[0], u0[1])

        candidates_lower = build_candidates(True, mid, H - 1, u1[0], u1[1])
        if not candidates_lower:
            candidates_lower = build_candidates(False, mid, H - 1, u1[0], u1[1])

        # =============================================================
        # 4. Fallback & Collision Avoidance
        # =============================================================
        if mines_detected_high:
            ref_x = int(round(sum(mx for mx, _ in mines_detected_high) / len(mines_detected_high)))
            ref_y = int(round(sum(my for _, my in mines_detected_high) / len(mines_detected_high)))
        else:
            ref_x, ref_y = (W // 2, H // 2)

        def fill_nearby(used: set) -> Tuple[int, int]:
            for d in range(0, max(W, H)):
                for ox in offsets_1d(d):
                    for oy in offsets_1d(d):
                        x, y = ref_x + ox, ref_y + oy
                        if not in_bounds(x, y):
                            continue
                        if clearance_map[y][x] == 0.0: 
                            continue
                        if (x, y) in used:
                            continue
                        used.add((x, y))
                        return (x, y)
            c = (scan_x, H // 2)
            used.add(c)
            return c

        used = set()
        
        u2 = candidates_upper[0][1] if candidates_upper else fill_nearby(used)
        used.add(u2)
        u3 = candidates_lower[0][1] if candidates_lower else fill_nearby(used)
        used.add(u3)

        # simple collision avoidance mechanism: if u2 and u3 are too close, force u3 to move away
        if abs(u2[0] - u3[0]) + abs(u2[1] - u3[1]) < 3:
            used = {u2, u3}
            u3 = fill_nearby(used)

        waypoints = [u0, u1, u2, u3]
        return [(int(min(max(0, x), W - 1)), int(min(max(0, y), H - 1)))
                for x, y in waypoints]

    # ------------------------------------------------------------------
    # Fixed Targets for initial setup or alternative modes
    # ------------------------------------------------------------------
    def fixed_targets(self, num_drones: int) -> List[Tuple[int, int]]:
        if num_drones <= 0:
            return []
        if num_drones == 1:
            ys = [self.H // 2]
        else:
            ys = [int(round(i * (self.H - 1) / (num_drones - 1)))
                  for i in range(num_drones)]
        tx = self.W - 1
        return [(tx, y) for y in ys]