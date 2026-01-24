"""
Simple Spacing Algorithm for IARC 2026 Simulation
Basic algorithm that spaces drones evenly and moves them in straight lines.
"""

from typing import List, Tuple
from algorithms.base import PathAlgorithm


class SimpleSpacingAlgorithm(PathAlgorithm):
    """
    Dead simple algorithm that:
    1. Spaces drones vertically based on detection radius
    2. Moves all drones straight across the arena (through the middle)
    3. At the far end, returns in straight lines back
    
    This is the baseline algorithm for comparison with more sophisticated approaches.
    """
    
    def __init__(self, config: dict, arena_width: float, arena_height: float):
        super().__init__(config, arena_width, arena_height)
        self.overlap_factor = self.algorithm_config.get("overlap_factor", 0.0)
    
    def calculate_initial_positions(self, num_drones: int) -> List[Tuple[float, float]]:
        """
        Space drones evenly across the arena height, starting from the left.
        
        Drones are positioned so their detection circles cover the maximum
        vertical area without gaps (unless overlap_factor is set).
        """
        positions = []
        
        if num_drones == 0:
            return positions
        
        # Calculate spacing between drone centers
        # Each drone covers 2 * detection_radius vertically
        coverage_per_drone = 2 * self.detection_radius
        
        # Apply overlap factor (0 = touching, positive = overlap)
        effective_spacing = coverage_per_drone * (1 - self.overlap_factor)
        
        # Calculate total height needed
        total_height_needed = num_drones * coverage_per_drone - (num_drones - 1) * coverage_per_drone * self.overlap_factor
        
        # Center the formation vertically in the arena
        start_y = (self.arena_height - total_height_needed) / 2 + self.detection_radius
        
        # If drones don't fit, compress them
        if total_height_needed > self.arena_height:
            # Compress to fit within arena
            available_height = self.arena_height - 2 * self.detection_radius
            if num_drones > 1:
                effective_spacing = available_height / (num_drones - 1)
            start_y = self.detection_radius
        
        for i in range(num_drones):
            x = self.start_x
            y = start_y + i * effective_spacing
            # Clamp to arena bounds
            y = max(self.detection_radius, min(self.arena_height - self.detection_radius, y))
            positions.append((x, y))
        
        return positions
    
    def plan_scan_path(self, drone_positions: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Plan a simple straight-line scan across the arena.
        
        All drones move from their current position to the far right side
        of the arena, maintaining their Y position.
        """
        self.phase = "scanning"
        waypoints_per_drone = []
        
        # Target X is near the right edge of the arena
        target_x = self.arena_width - self.start_x
        
        for pos in drone_positions:
            current_x, current_y = pos
            # Simple straight line to the right
            waypoints = [(target_x, current_y)]
            waypoints_per_drone.append(waypoints)
        
        return waypoints_per_drone
    
    def plan_return_path(
        self, 
        drone_positions: List[Tuple[float, float]], 
        detected_mines: List[Tuple[float, float]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Plan a simple straight-line return to the starting side.
        
        For the simple algorithm, we just go straight back.
        More advanced algorithms would optimize this path.
        """
        self.phase = "returning"
        waypoints_per_drone = []
        
        for pos in drone_positions:
            current_x, current_y = pos
            # Return to starting X position
            waypoints = [(self.start_x, current_y)]
            waypoints_per_drone.append(waypoints)
        
        return waypoints_per_drone
    
    def check_scan_complete(self, drone_positions: List[Tuple[float, float]]) -> bool:
        """
        Check if all drones have reached the far side of the arena.
        """
        target_x = self.arena_width - self.start_x
        tolerance = self.path_config.get("waypoint_tolerance_ft", 1.0)
        
        for pos in drone_positions:
            if pos[0] < target_x - tolerance:
                return False
        
        return True
    
    def check_return_complete(self, drone_positions: List[Tuple[float, float]]) -> bool:
        """
        Check if all drones have returned to the starting side.
        """
        tolerance = self.path_config.get("waypoint_tolerance_ft", 1.0)
        
        for pos in drone_positions:
            if pos[0] > self.start_x + tolerance:
                return False
        
        return True
