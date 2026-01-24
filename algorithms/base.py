"""
Base Algorithm Class for IARC 2026 Simulation
Defines the interface that all path-finding algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class PathAlgorithm(ABC):
    """
    Abstract base class for drone path-finding algorithms.
    
    All algorithms receive the same state and must produce waypoints
    for each drone to follow.
    """
    
    def __init__(self, config: dict, arena_width: float, arena_height: float):
        """
        Initialize the algorithm with configuration and arena dimensions.
        
        Args:
            config: Full configuration dictionary from config.yaml
            arena_width: Arena width in feet
            arena_height: Arena height in feet
        """
        self.config = config
        self.arena_width = arena_width
        self.arena_height = arena_height
        
        # Extract common configuration
        self.drone_config = config.get("drone", {})
        self.path_config = config.get("path", {})
        self.detection_radius = self.drone_config.get("detection_radius_ft", 5.0)
        self.spacing_margin = self.drone_config.get("spacing_margin_ft", 1.0)
        self.start_x = self.drone_config.get("start_x_ft", 5.0)
        
        # Algorithm-specific config
        algorithm_name = config.get("algorithm", "simple_spacing")
        algorithms_config = config.get("algorithms", {})
        self.algorithm_config = algorithms_config.get(algorithm_name, {})
        
        # State tracking
        self.phase = "initial"  # initial, scanning, returning, complete
        self.total_path_length = 0.0
    
    @abstractmethod
    def calculate_initial_positions(self, num_drones: int) -> List[Tuple[float, float]]:
        """
        Calculate starting positions for all drones with proper spacing.
        
        Args:
            num_drones: Number of drones in the swarm
        
        Returns:
            List of (x, y) positions for each drone
        """
        pass
    
    @abstractmethod
    def plan_scan_path(self, drone_positions: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Plan the scanning path through the arena.
        
        Args:
            drone_positions: Current (x, y) positions of all drones
        
        Returns:
            List of waypoint lists, one for each drone.
            Each inner list contains (x, y) waypoints.
        """
        pass
    
    @abstractmethod
    def plan_return_path(
        self, 
        drone_positions: List[Tuple[float, float]], 
        detected_mines: List[Tuple[float, float]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Plan the return path after scanning, optimizing for score.
        
        Args:
            drone_positions: Current (x, y) positions of all drones
            detected_mines: List of (x, y) positions of detected mines
        
        Returns:
            List of waypoint lists, one for each drone.
        """
        pass
    
    def get_phase(self) -> str:
        """Get the current algorithm phase."""
        return self.phase
    
    def set_phase(self, phase: str):
        """Set the algorithm phase."""
        self.phase = phase
    
    def add_path_length(self, distance: float):
        """Add to the total path length tracking."""
        self.total_path_length += distance
    
    def get_total_path_length(self) -> float:
        """Get the accumulated path length."""
        return self.total_path_length
    
    def calculate_coverage_width(self, num_drones: int) -> float:
        """
        Calculate the total width covered by all drones.
        
        Args:
            num_drones: Number of drones
        
        Returns:
            Total coverage width in feet
        """
        # Each drone covers 2 * detection_radius
        # With spacing_margin between coverage areas
        single_coverage = 2 * self.detection_radius
        overlap_factor = self.algorithm_config.get("overlap_factor", 0.0)
        
        if num_drones == 1:
            return single_coverage
        
        # Total = first drone + (n-1) * (coverage - overlap)
        overlap = single_coverage * overlap_factor
        return single_coverage + (num_drones - 1) * (single_coverage - overlap)
