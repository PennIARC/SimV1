#####
# A* path planner on discrete 2ft grid.
#
# Inherits infrastructure from GreedyBottleneckPlanner;
# overrides plan() with standard A* (g + h) search.
#
# Key differences from greedy:
#   - Strict clearance >= 1.0 (2ft safety buffer from mines)
#   - True cost function g(n) with corridor-centering penalty
#   - Confidence < 0.5 penalty to prefer explored areas
#####

import heapq
import math
from typing import List, Tuple, Optional, Dict

from path_planning.greedy_path_planning import (
    GreedyBottleneckPlanner,
    CLEARANCE_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)

# Strict safety buffer: 1.0 grid unit = 2ft radius from any mine
ASTAR_CLEARANCE_MIN = 1.0

# Cost penalties
PENALTY_LOW_CLEARANCE = 5.0   # extra cost per step when clearance is between 1.0 and 2.0
PENALTY_LOW_CONFIDENCE = 20.0  # extra cost per step when confidence < 0.5 (unexplored)


class AStarPlanner(GreedyBottleneckPlanner):
    """
    A* path planner for the 2ft competition grid.

    Finds the lowest-cost path from any start cell to the goal column (x=149).
    Cost function penalizes narrow corridors and unexplored regions,
    pushing the path toward wide, well-surveyed corridors.

    Inherits persistent_best, _update_persistent, _seed_positions,
    reconstruct_path_from_grid, and grid helpers from GreedyBottleneckPlanner.
    """

    def __init__(
        self,
        height: int,
        width: int,
        start_cells: List[Tuple[int, int]],
        goal_cells: List[Tuple[int, int]],
    ):
        super().__init__(height, width, start_cells, goal_cells)

    def _is_traversable(
        self,
        x: int,
        y: int,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> bool:
        """Stricter than greedy: require clearance >= 1.0 (2ft safety buffer)."""
        return (
            confidence_map[y][x] > CONFIDENCE_THRESHOLD
            and clearance_map[y][x] >= ASTAR_CLEARANCE_MIN
        )

    def _step_cost(
        self,
        x: int,
        y: int,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> float:
        """
        Cost of stepping into cell (x, y).

        Base cost is 1.0 per step. Additional penalties:
        - Low clearance (1.0 <= c < 2.0): penalizes edges of corridors
        - Low confidence (< 0.5): penalizes unexplored territory
        """
        cost = 1.0

        clr = clearance_map[y][x]
        if clr < 2.0:
            # Linear penalty: 0 at clearance=2.0, full penalty at clearance=1.0
            cost += PENALTY_LOW_CLEARANCE * (2.0 - clr)

        conf = confidence_map[y][x]
        if conf < 0.5:
            cost += PENALTY_LOW_CONFIDENCE * (0.5 - conf)

        return cost

    def plan(
        self,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> Optional[Dict]:
        """
        A* search on 4-connected grid.

        Priority queue ordered by f = g + h where:
            g = cumulative step cost from start
            h = Manhattan distance to nearest goal cell

        Returns dict matching greedy/RRT format:
            path, bottleneck, length, reached, [persistent_path]
        """
        INF = float("inf")
        g_cost = [[INF] * self.width for _ in range(self.height)]
        self.parent = [[None] * self.width for _ in range(self.height)]

        # Track bottleneck (min clearance along path) for scoring compatibility
        bottleneck_at = [[0.0] * self.width for _ in range(self.height)]

        # Heap: (f, g, h, x, y)
        open_set: List[Tuple[float, float, float, int, int]] = []

        seeds = self._seed_positions(confidence_map, clearance_map)
        if not seeds:
            return None

        for x, y in seeds:
            g = 0.0
            h = self.heuristic_distance_to_goal(x, y)
            if g < g_cost[y][x]:
                g_cost[y][x] = g
                bottleneck_at[y][x] = clearance_map[y][x]
                self.parent[y][x] = None
                heapq.heappush(open_set, (g + h, g, h, x, y))

        best_partial: Optional[Tuple[int, int, float, float]] = None  # (x, y, bottleneck, g)

        while open_set:
            f, g, _, cx, cy = heapq.heappop(open_set)

            if g > g_cost[cy][cx]:
                continue

            cw = bottleneck_at[cy][cx]
            h_val = self.heuristic_distance_to_goal(cx, cy)

            # Track best frontier node (closest to goal, then lowest cost)
            if best_partial is None:
                best_partial = (cx, cy, cw, g)
            else:
                bx, by, bbn, bg = best_partial
                bh = self.heuristic_distance_to_goal(bx, by)
                if h_val < bh or (h_val == bh and g < bg):
                    best_partial = (cx, cy, cw, g)

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

                new_g = g + self._step_cost(nx, ny, confidence_map, clearance_map)
                if new_g >= g_cost[ny][nx]:
                    continue

                g_cost[ny][nx] = new_g
                bottleneck_at[ny][nx] = min(cw, clearance_map[ny][nx])
                self.parent[ny][nx] = (cx, cy)
                new_h = self.heuristic_distance_to_goal(nx, ny)
                heapq.heappush(open_set, (new_g + new_h, new_g, new_h, nx, ny))

        if best_partial is None:
            return None

        bx, by, bw, _ = best_partial
        path = self.reconstruct_path_from_grid(bx, by)
        self._update_persistent(path, bw, clearance_map)

        return {
            "path": path,
            "bottleneck": bw,
            "length": len(path),
            "reached": False,
            "persistent_path": self.persistent_best,
        }
