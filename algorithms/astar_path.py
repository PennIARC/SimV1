"""
A* Path Algorithm for IARC 2026 Simulation
Uses A* search for optimal path finding with scoring-based heuristic.
"""

import math
import heapq
from typing import List, Tuple, Dict, Set
from algorithms.base import PathAlgorithm


class AStarPathAlgorithm(PathAlgorithm):
    """
    A* pathfinding variant that:
    1. Initial phase same as simple spacing
    2. On return, uses A* with a heuristic based on the IARC scoring function
    3. Considers mine locations in the heuristic for coverage optimization
    """
    
    def __init__(self, config: dict, arena_width: float, arena_height: float):
        super().__init__(config, arena_width, arena_height)
        self.grid_resolution = self.algorithm_config.get("grid_resolution_ft", 5.0)
        self.heuristic_weight = self.algorithm_config.get("heuristic_weight", 1.2)
        
        # Precompute grid dimensions
        self.grid_width = int(arena_width / self.grid_resolution) + 1
        self.grid_height = int(arena_height / self.grid_resolution) + 1
    
    def calculate_initial_positions(self, num_drones: int) -> List[Tuple[float, float]]:
        """
        Space drones evenly across the arena height, starting from the left.
        Same as SimpleSpacingAlgorithm for the initial phase.
        """
        positions = []
        
        if num_drones == 0:
            return positions
        
        coverage_per_drone = 2 * self.detection_radius
        overlap_factor = self.algorithm_config.get("overlap_factor", 0.0)
        effective_spacing = coverage_per_drone * (1 - overlap_factor)
        
        total_height_needed = num_drones * coverage_per_drone - (num_drones - 1) * coverage_per_drone * overlap_factor
        start_y = (self.arena_height - total_height_needed) / 2 + self.detection_radius
        
        if total_height_needed > self.arena_height:
            available_height = self.arena_height - 2 * self.detection_radius
            if num_drones > 1:
                effective_spacing = available_height / (num_drones - 1)
            start_y = self.detection_radius
        
        for i in range(num_drones):
            x = self.start_x
            y = start_y + i * effective_spacing
            y = max(self.detection_radius, min(self.arena_height - self.detection_radius, y))
            positions.append((x, y))
        
        return positions
    
    def plan_scan_path(self, drone_positions: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Plan a straight-line scan across the arena.
        Same as SimpleSpacingAlgorithm for the scanning phase.
        """
        self.phase = "scanning"
        waypoints_per_drone = []
        
        target_x = self.arena_width - self.start_x
        
        for pos in drone_positions:
            current_x, current_y = pos
            waypoints = [(target_x, current_y)]
            waypoints_per_drone.append(waypoints)
        
        return waypoints_per_drone
    
    def plan_return_path(
        self, 
        drone_positions: List[Tuple[float, float]], 
        detected_mines: List[Tuple[float, float]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Plan an optimal return path using A* search.
        
        Uses a heuristic that considers:
        - Distance to goal (path length)
        - Mine coverage density
        - Avoiding redundant areas
        """
        self.phase = "returning"
        waypoints_per_drone = []
        
        # Build a density map of detected mines
        mine_density = self._build_mine_density_map(detected_mines)
        
        for i, pos in enumerate(drone_positions):
            current_x, current_y = pos
            goal = (self.start_x, current_y)
            
            # Run A* to find optimal path
            path = self._astar(
                start=(current_x, current_y),
                goal=goal,
                mine_density=mine_density,
                other_drone_paths=[wp for j, wp in enumerate(waypoints_per_drone) if j != i]
            )
            
            # Convert path to waypoints (skip current position)
            waypoints = path[1:] if len(path) > 1 else [goal]
            waypoints_per_drone.append(waypoints)
        
        return waypoints_per_drone
    
    def _build_mine_density_map(self, detected_mines: List[Tuple[float, float]]) -> Dict[Tuple[int, int], float]:
        """
        Build a grid-based density map of detected mines.
        
        Returns a dict mapping grid cells to mine density.
        """
        density = {}
        
        for mine_x, mine_y in detected_mines:
            grid_x = int(mine_x / self.grid_resolution)
            grid_y = int(mine_y / self.grid_resolution)
            
            # Clamp to grid bounds
            grid_x = max(0, min(self.grid_width - 1, grid_x))
            grid_y = max(0, min(self.grid_height - 1, grid_y))
            
            key = (grid_x, grid_y)
            density[key] = density.get(key, 0) + 1
        
        return density
    
    def _astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        mine_density: Dict[Tuple[int, int], float],
        other_drone_paths: List[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """
        A* search for optimal path from start to goal.
        
        Uses a modified heuristic that considers:
        - Distance to goal
        - Preference for areas with low mine density (potential gaps)
        """
        # Convert to grid coordinates
        start_grid = (int(start[0] / self.grid_resolution), int(start[1] / self.grid_resolution))
        goal_grid = (int(goal[0] / self.grid_resolution), int(goal[1] / self.grid_resolution))
        
        # Priority queue: (f_score, counter, grid_pos)
        counter = 0
        open_set = [(0, counter, start_grid)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0}
        f_score: Dict[Tuple[int, int], float] = {start_grid: self._heuristic(start_grid, goal_grid, mine_density)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # Explore neighbors (8-directional movement)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.grid_width and 0 <= neighbor[1] < self.grid_height):
                    continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost
                move_cost = math.sqrt(dx**2 + dy**2) * self.grid_resolution
                
                # Add penalty for dense areas (already covered)
                density_penalty = mine_density.get(neighbor, 0) * 0.5
                
                tentative_g = g_score[current] + move_cost + density_penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_grid, mine_density)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        # No path found, return direct path
        return [start, goal]
    
    def _heuristic(
        self, 
        pos: Tuple[int, int], 
        goal: Tuple[int, int],
        mine_density: Dict[Tuple[int, int], float]
    ) -> float:
        """
        A* heuristic function.
        
        Combines:
        - Euclidean distance to goal (admissible)
        - Bonus for low-density areas (potential gaps to fill)
        """
        # Euclidean distance in feet
        dx = (pos[0] - goal[0]) * self.grid_resolution
        dy = (pos[1] - goal[1]) * self.grid_resolution
        distance = math.sqrt(dx**2 + dy**2)
        
        # Low density bonus (areas with fewer mines might need coverage)
        density = mine_density.get(pos, 0)
        density_factor = 1.0 / (1.0 + density)  # Higher when density is low
        
        # Combine (distance is primary, density is secondary)
        return distance - density_factor * 2
    
    def _reconstruct_path(
        self, 
        came_from: Dict[Tuple[int, int], Tuple[int, int]], 
        current: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Reconstruct the path from A* search results.
        
        Converts grid coordinates back to world coordinates.
        """
        path_grid = [current]
        
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        
        path_grid.reverse()
        
        # Convert to world coordinates (center of each grid cell)
        path_world = []
        for gx, gy in path_grid:
            wx = (gx + 0.5) * self.grid_resolution
            wy = (gy + 0.5) * self.grid_resolution
            # Clamp to arena bounds
            wx = max(self.start_x, min(self.arena_width - self.start_x, wx))
            wy = max(self.detection_radius, min(self.arena_height - self.detection_radius, wy))
            path_world.append((wx, wy))
        
        # Simplify path (remove redundant waypoints)
        return self._simplify_path(path_world)
    
    def _simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Simplify the path by removing collinear points.
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            next_pt = path[i + 1]
            
            # Check if curr is collinear with prev and next
            # Using cross product
            cross = (curr[0] - prev[0]) * (next_pt[1] - prev[1]) - (curr[1] - prev[1]) * (next_pt[0] - prev[0])
            
            if abs(cross) > 0.1:  # Not collinear
                simplified.append(curr)
        
        simplified.append(path[-1])
        return simplified
    
    def check_scan_complete(self, drone_positions: List[Tuple[float, float]]) -> bool:
        """Check if all drones have reached the far side of the arena."""
        target_x = self.arena_width - self.start_x
        tolerance = self.path_config.get("waypoint_tolerance_ft", 1.0)
        
        for pos in drone_positions:
            if pos[0] < target_x - tolerance:
                return False
        return True
    
    def check_return_complete(self, drone_positions: List[Tuple[float, float]]) -> bool:
        """Check if all drones have returned to the starting side."""
        tolerance = self.path_config.get("waypoint_tolerance_ft", 1.0)
        
        for pos in drone_positions:
            if pos[0] > self.start_x + tolerance:
                return False
        return True
