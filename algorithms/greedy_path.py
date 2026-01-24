"""
Greedy Path Algorithm for IARC 2026 Simulation
Optimizes the return path using a greedy approach.
"""

import math
from typing import List, Tuple
from algorithms.base import PathAlgorithm


class GreedyPathAlgorithm(PathAlgorithm):
    """
    Greedy path optimizer that:
    1. Initial phase same as simple spacing
    2. On return, greedily optimizes for best score by:
       - Minimizing path length
       - Maximizing mine detection coverage
       - Minimizing time
    """
    
    def __init__(self, config: dict, arena_width: float, arena_height: float):
        super().__init__(config, arena_width, arena_height)
        self.lookahead_steps = self.algorithm_config.get("lookahead_steps", 5)
        self.mine_weight = self.algorithm_config.get("mine_weight", 1.0)
        self.length_weight = self.algorithm_config.get("length_weight", 1.0)
    
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
        Plan an optimized return path using greedy approach.
        
        The greedy algorithm tries to:
        1. Cover any missed areas (based on detected mine distribution)
        2. Minimize total path length
        3. Avoid redundant coverage
        """
        self.phase = "returning"
        waypoints_per_drone = []
        
        # Analyze mine distribution to find potential gaps
        mine_y_positions = [m[1] for m in detected_mines] if detected_mines else []
        
        for i, pos in enumerate(drone_positions):
            current_x, current_y = pos
            waypoints = []
            
            # Calculate optimal return path
            if mine_y_positions:
                # Find areas with low mine density (potential gaps)
                optimal_y = self._find_optimal_y(current_y, mine_y_positions, drone_positions, i)
                
                # Create a curved return path through optimal Y
                mid_x = self.arena_width / 2
                waypoints.append((mid_x, optimal_y))
            
            # Final waypoint: return to starting X
            waypoints.append((self.start_x, current_y))
            waypoints_per_drone.append(waypoints)
        
        return waypoints_per_drone
    
    def _find_optimal_y(
        self, 
        current_y: float, 
        mine_y_positions: List[float], 
        all_drone_positions: List[Tuple[float, float]],
        drone_index: int
    ) -> float:
        """
        Find the optimal Y position for the return path.
        
        Uses a greedy approach to balance:
        - Staying close to current Y (minimize path length)
        - Covering areas with fewer detected mines
        - Not overlapping with other drones
        """
        best_y = current_y
        best_score = float('-inf')
        
        # Try different Y positions
        step = 5.0  # feet
        min_y = self.detection_radius
        max_y = self.arena_height - self.detection_radius
        
        for test_y in self._frange(min_y, max_y, step):
            score = self._evaluate_y_position(
                test_y, current_y, mine_y_positions, 
                all_drone_positions, drone_index
            )
            if score > best_score:
                best_score = score
                best_y = test_y
        
        return best_y
    
    def _evaluate_y_position(
        self,
        test_y: float,
        current_y: float,
        mine_y_positions: List[float],
        all_drone_positions: List[Tuple[float, float]],
        drone_index: int
    ) -> float:
        """
        Evaluate how good a Y position is for the return path.
        
        Returns a score where higher is better.
        """
        score = 0.0
        
        # Penalty for distance from current Y (path length)
        distance_penalty = abs(test_y - current_y) * self.length_weight
        score -= distance_penalty
        
        # Bonus for covering areas with fewer mines (potential missed areas)
        mines_in_range = sum(
            1 for my in mine_y_positions 
            if abs(my - test_y) < self.detection_radius
        )
        # Fewer mines = potentially missed area = higher priority
        coverage_bonus = (1.0 / (1.0 + mines_in_range)) * self.mine_weight * 10
        score += coverage_bonus
        
        # Penalty for overlapping with other drones
        for j, other_pos in enumerate(all_drone_positions):
            if j != drone_index:
                other_y = other_pos[1]
                if abs(test_y - other_y) < self.detection_radius * 2:
                    overlap = self.detection_radius * 2 - abs(test_y - other_y)
                    score -= overlap * 2
        
        return score
    
    def _frange(self, start: float, stop: float, step: float):
        """Float range generator."""
        current = start
        while current <= stop:
            yield current
            current += step
    
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
