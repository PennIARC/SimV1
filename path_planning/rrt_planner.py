#####
# RRT (Rapidly-exploring Random Tree) path planner on discrete 2ft grid.
#
# Inherits exploration/waypoint logic from GreedyBottleneckPlanner;
# only overrides plan() with RRT-based search.
#
#####

import random
import math
from typing import List, Tuple, Optional, Dict

from path_planning.greedy_path_planning import (
    GreedyBottleneckPlanner,
    CLEARANCE_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)


class RRTPlanner(GreedyBottleneckPlanner):
    """
    RRT path planner for the 2ft competition grid.

    using the same exploration/waypoint logic as GreedyBottleneckPlanner (suggest_exploration_targets,
    fixed_targets, _update_persistent etc.), only replace plan() with RRT search.）

    Params:
        max_iter:   maximum number of iterations for each plan() call (larger → more likely to find path, but slower)
        step_size:  maximum number of steps to take in each extension (affects tree growth speed vs accuracy)
        goal_bias:  probability of directly sampling the goal column (0~1, higher → more greedy)
    """

    def __init__(
        self,
        height: int,
        width: int,
        start_cells: List[Tuple[int, int]],
        goal_cells: List[Tuple[int, int]],
        max_iter: int = 3000,
        step_size: int = 5,
        goal_bias: float = 0.15,
    ):
        # call parent class initialization: set height/width/start_cells/goal_cells/persistent_best etc.
        super().__init__(height, width, start_cells, goal_cells)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias

    # ------------------------------------------------------------------
    # Core RRT: override plan()
    # ------------------------------------------------------------------
    def plan(
        self,
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> Optional[Dict]:
        """
        RRT planning on 4-connected discrete grid.

        called once per frame, generate a RRT based on the current confidence/clearance map
        """

        # ------ Step 1: find valid start ------
        # reuse the seed finding logic from the parent class (will find feasible cells near the start column)
        seeds = self._seed_positions(confidence_map, clearance_map)
        if not seeds:
            return None

        # Pick the seed closest to goal as starting point
        start = min(seeds, key=lambda s: self.heuristic_distance_to_goal(s[0], s[1]))

        # ------ Tree data structures ------
        # tree_parent: node → parent (None for root)
        # tree_bottleneck: node → min clearance from root to this node
        # node_set: O(1) membership check
        tree_parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        tree_bottleneck: Dict[Tuple[int, int], float] = {
            start: clearance_map[start[1]][start[0]]
        }
        tree_nodes: List[Tuple[int, int]] = [start]
        node_set = {start}

        # Track best frontier node: (node, goal_dist, bottleneck)
        best_partial: Optional[Tuple[Tuple[int, int], float, float]] = None

        # Precompute goal_cells list for random choice
        goal_list = list(self.goal_cells)

        # ------ Step 2: RRT main loop ------
        for _ in range(self.max_iter):

            # (a) Sample a random target cell
            if random.random() < self.goal_bias:
                # Bias toward goal: pick a random cell on the goal column
                sample = random.choice(goal_list)
            else:
                # Uniform random over grid
                sx = random.randint(0, self.width - 1)
                sy = random.randint(0, self.height - 1)
                sample = (sx, sy)

            # (b) Nearest neighbor in tree (Manhattan distance, brute force)
            nearest = self._nearest(tree_nodes, sample)

            # (c) Extend tree from nearest toward sample
            new_node = self._extend(
                nearest, sample, confidence_map, clearance_map
            )

            # Skip if extension failed or node already in tree
            if new_node is None or new_node in node_set:
                continue

            # (d) Add new_node to tree
            tree_nodes.append(new_node)
            node_set.add(new_node)
            tree_parent[new_node] = nearest
            tree_bottleneck[new_node] = min(
                tree_bottleneck[nearest],
                clearance_map[new_node[1]][new_node[0]],
            )

            # Update best partial (closest to goal, then highest bottleneck)
            dist = self.heuristic_distance_to_goal(new_node[0], new_node[1])
            bn = tree_bottleneck[new_node]
            if best_partial is None:
                best_partial = (new_node, dist, bn)
            else:
                _, bd, bb = best_partial
                if dist < bd or (dist == bd and bn > bb):
                    best_partial = (new_node, dist, bn)

            # (e) Check if we reached a goal cell
            if new_node in self.goal_cells:
                path = self._reconstruct_rrt(new_node, tree_parent)
                return {
                    "path": path,
                    "bottleneck": tree_bottleneck[new_node],
                    "length": len(path),
                    "reached": True,
                }

        # ------ Step 3: no goal → return best partial ------
        if best_partial is None:
            return None

        node, dist, bn = best_partial
        path = self._reconstruct_rrt(node, tree_parent)
        self._update_persistent(path, bn, clearance_map)

        return {
            "path": path,
            "bottleneck": bn,
            "length": len(path),
            "reached": False,
            "persistent_path": self.persistent_best,
        }

    # ------------------------------------------------------------------
    # RRT helper: find nearest node in tree
    # ------------------------------------------------------------------
    def _nearest(
        self,
        tree_nodes: List[Tuple[int, int]],
        sample: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Brute-force nearest neighbor search (Manhattan distance).
        the grid has 150×40=6000 cells, the tree can have at most ~3000 nodes, brute force
        """
        sx, sy = sample
        return min(tree_nodes, key=lambda n: abs(n[0] - sx) + abs(n[1] - sy))

    # ------------------------------------------------------------------
    # RRT helper: extend tree from 'from_node' toward 'to_sample'
    # ------------------------------------------------------------------
    def _is_traversable(self, x, y, confidence_map, clearance_map):
        # 1. Safety Buffer (The Inflation Layer)
        # Must stay away from confirmed mines (0.0)
        # Using 1.0 ensures we satisfy the 1-foot safety radius.
        if clearance_map[y][x] < 1.0:
            return False

        # 2. Confidence Check
        # Instead of a fixed threshold (0.5), we check if the cell has ANY exploration data.
        # -1.0 usually means "Unexplored" in your droneManager logic.
        # 0.5 means High-alt scanned; 1.0 means Low-alt confirmed.
        if confidence_map[y][x] <= 0: # This includes -1.0 (unexplored)
            return False

        return True

    def _extend(
        self,
        from_node: Tuple[int, int],
        to_sample: Tuple[int, int],
        confidence_map: List[List[float]],
        clearance_map: List[List[float]],
    ) -> Optional[Tuple[int, int]]:
        """



        

        from from_node, take greedy steps toward to_sample:
        - take one step in the direction that reduces Manhattan distance to to_sample the most
        - stop if out-of-bounds or non-traversable cell is hit
        - return the last valid cell reached, or None if no progress
        """
        cx, cy = from_node
        tx, ty = to_sample
        last_valid: Optional[Tuple[int, int]] = None

        for _ in range(self.step_size):
            dx = tx - cx
            dy = ty - cy

            # Already at target
            if dx == 0 and dy == 0:
                break

            # Choose the axis with larger remaining distance
            if abs(dx) >= abs(dy):
                nx = cx + (1 if dx > 0 else -1)
                ny = cy
            else:
                nx = cx
                ny = cy + (1 if dy > 0 else -1)

            # Bounds check
            if not self.in_bounds(nx, ny):
                break

            # Traversability check (same rule as greedy: confidence + clearance)
            if not self._is_traversable(nx, ny, confidence_map, clearance_map):
                break

            cx, cy = nx, ny
            last_valid = (cx, cy)

        return last_valid

    # ------------------------------------------------------------------
    # RRT helper: fill intermediate cells between two tree nodes
    # ------------------------------------------------------------------
    def _fill_cells(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Walk cell-by-cell from a toward b (same greedy-axis logic as _extend).
        Returns intermediate cells EXCLUDING a, INCLUDING b.
        """
        cx, cy = a
        tx, ty = b
        cells: List[Tuple[int, int]] = []
        max_steps = abs(tx - cx) + abs(ty - cy)
        for _ in range(max_steps):
            if (cx, cy) == (tx, ty):
                break
            dx = tx - cx
            dy = ty - cy
            if abs(dx) >= abs(dy):
                cx += 1 if dx > 0 else -1
            else:
                cy += 1 if dy > 0 else -1
            cells.append((cx, cy))
        return cells

    # ------------------------------------------------------------------
    # RRT helper: reconstruct path from tree parent pointers
    # ------------------------------------------------------------------
    def _reconstruct_rrt(
        self,
        node: Tuple[int, int],
        parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        """
        Follow parent pointers back to root, then reverse.
        Fill ALL intermediate cells so the path is cell-by-cell
        (same format as greedy: consecutive entries are adjacent).
        """
        raw: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = node
        while current is not None:
            raw.append(current)
            current = parent_map.get(current)
        raw.reverse()

        if len(raw) <= 1:
            return raw

        # full_path = [raw[0]]
        # for i in range(1, len(raw)):
        #     full_path.extend(self._fill_cells(raw[i - 1], raw[i]))
        # return full_path
        
        # 1. Expand the sparse RRT nodes into a continuous step-by-step path
        full_path = [raw[0]]
        for i in range(1, len(raw)):
            full_path.extend(self._fill_cells(raw[i - 1], raw[i]))
        
        # 2. Crucial: Remove consecutive duplicate nodes
        # This prevents the scoring pipeline from failing due to "zero-length steps"
        unique_path = []
        for p in full_path:
            if not unique_path or p != unique_path[-1]:
                unique_path.append(p)
                
        return unique_path # Only ONE return at the end