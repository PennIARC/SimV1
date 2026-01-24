"""
IARC 2026 Scoring System
Implements the official IARC scoring formula.

Score = 150000 * W / [(1+B) * L * (1 + 7*A + 100*N)]

Where:
- W = path width in feet = 2 × (1 + 2G), with G being green zone width in squares
- L = path length in feet = 2 × (sum of all movement steps)
- B = number of missed mines in the green zone
- A = minutes to complete the scan
- N = ounces over the one-pound limit
"""

from typing import List, Tuple
import math


def calculate_path_width(green_zone_squares: int) -> float:
    """
    Calculate the path width (W) component of the score.
    
    Args:
        green_zone_squares: G value - green zone width in squares
    
    Returns:
        W = 2 × (1 + 2G) in feet
    """
    return 2.0 * (1.0 + 2.0 * green_zone_squares)


def calculate_path_length(total_movement_ft: float) -> float:
    """
    Calculate the path length (L) component of the score.
    
    Args:
        total_movement_ft: Sum of all movement steps in feet
    
    Returns:
        L = 2 × (sum of all movement steps) in feet
    """
    return 2.0 * total_movement_ft


def calculate_score(
    W: float, 
    L: float, 
    B: int, 
    A: float, 
    N: float
) -> float:
    """
    Calculate the IARC competition score.
    
    Args:
        W: Path width in feet
        L: Path length in feet
        B: Number of missed mines in the green zone
        A: Minutes to complete the scan
        N: Ounces over the one-pound limit
    
    Returns:
        Score value (higher is better)
    """
    if L <= 0:
        return 0.0
    
    numerator = 150000.0 * W
    denominator = (1.0 + B) * L * (1.0 + 7.0 * A + 100.0 * N)
    
    if denominator <= 0:
        return 0.0
    
    return numerator / denominator


class ScoreTracker:
    """
    Tracks all metrics needed for IARC scoring during simulation.
    
    Accumulates movement distances, elapsed time, and mine counts.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the score tracker.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        scoring_config = config.get("scoring", {})
        
        self.green_zone_squares = scoring_config.get("green_zone_squares", 1)
        self.weight_penalty_oz = scoring_config.get("weight_penalty_oz", 0)
        
        # Metrics to track
        self.total_movement_ft = 0.0  # Sum of all drone movements
        self.elapsed_seconds = 0.0    # Total time elapsed
        self.missed_mines = 0          # Mines in green zone not detected
        
        # Previous positions for distance calculation
        self.prev_positions: List[Tuple[float, float]] = []
        
        # Explicit safe path metrics (overrides accumulation for L if set)
        self.safe_path_width_ft = 0.0
        self.safe_path_length_ft = 0.0
        self.using_safe_path = False
        
    def set_safe_path_metrics(self, width_ft: float, length_ft: float):
        """Set the calculated safe path metrics."""
        self.safe_path_width_ft = width_ft
        self.safe_path_length_ft = length_ft
        self.using_safe_path = True
    
    def reset(self):
        """Reset all tracked metrics."""
        self.total_movement_ft = 0.0
        self.elapsed_seconds = 0.0
        self.missed_mines = 0
        self.prev_positions = []
        self.using_safe_path = False
        self.safe_path_width_ft = 0.0
        self.safe_path_length_ft = 0.0
    
    def update(
        self, 
        drone_positions: List[Tuple[float, float]], 
        dt: float,
        mines_truth: List[Tuple[float, float]] = None,
        mines_detected: List[Tuple[float, float]] = None
    ):
        """
        Update metrics with current simulation state.
        
        Args:
            drone_positions: Current (x, y) positions of all drones
            dt: Time delta since last update
            mines_truth: All mines (for calculating missed)
            mines_detected: Mines that have been detected
        """
        # Track elapsed time
        self.elapsed_seconds += dt
        
        # Track movement distance
        if self.prev_positions:
            for i, pos in enumerate(drone_positions):
                if i < len(self.prev_positions):
                    prev = self.prev_positions[i]
                    distance = math.sqrt(
                        (pos[0] - prev[0])**2 + (pos[1] - prev[1])**2
                    )
                    self.total_movement_ft += distance
        
        # Update previous positions
        self.prev_positions = list(drone_positions)
        
        # Calculate missed mines (if data provided)
        if mines_truth is not None and mines_detected is not None:
            self.missed_mines = len(mines_truth) - len(mines_detected)
    
    def get_W(self) -> float:
        """Get the path width component."""
        if self.using_safe_path:
            # If width is already calculating feet, use it. 
            # IARC says W = 2 * (1 + 2G). 
            # If safe_path_width_ft IS calculating the bottleneck width in feet, we can use it directly?
            # Or is safe_path_width_ft = G?
            # The SafePathFinder returns 'clearance_ft' (radius).
            # So width = 2 * clearance.
            # But the formula is W = 2 * (1 + 2G).
            # This implies G is unitless "squares"? 
            # Let's assume the SafePathFinder provides the effective 'W' directly or we map it.
            # Let's assume calculate_path_width expects abstract G.
            # But if we have actual physical width, we should use it.
            # The user provided the formula image.
            # "W = path width in feet".
            # If we know the physical bottleneck width in feet (let's say 10ft), W = 10?
            # The formula W = 2 * (1 + 2G) calculates W from G.
            # If we measure W directly, we just return W.
            # So if safe_path_width_ft is the bottleneck diameter, return it.
            return self.safe_path_width_ft * 2.0 # clearance is radius, so diameter
            
        return calculate_path_width(self.green_zone_squares)
    
    def get_L(self) -> float:
        """Get the path length component."""
        if self.using_safe_path:
            return self.safe_path_length_ft
            
        return calculate_path_length(self.total_movement_ft)
    
    def get_B(self) -> int:
        """Get the missed mines count."""
        return self.missed_mines
    
    def get_A(self) -> float:
        """Get the elapsed time in minutes."""
        return self.elapsed_seconds / 60.0
    
    def get_N(self) -> float:
        """Get the weight penalty in ounces."""
        return self.weight_penalty_oz
    
    def get_score(self) -> float:
        """
        Calculate the current IARC score.
        
        Returns:
            Current score based on accumulated metrics
        """
        return calculate_score(
            W=self.get_W(),
            L=self.get_L(),
            B=self.get_B(),
            A=self.get_A(),
            N=self.get_N()
        )
    
    def get_metrics_dict(self) -> dict:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary with all scoring components and final score
        """
        return {
            "W": self.get_W(),
            "L": self.get_L(),
            "B": self.get_B(),
            "A": self.get_A(),
            "N": self.get_N(),
            "score": self.get_score(),
            "total_movement_ft": self.total_movement_ft,
            "elapsed_seconds": self.elapsed_seconds,
        }
    
    def format_score_string(self) -> str:
        """
        Format the score and metrics as a display string.
        
        Returns:
            Formatted string for UI display
        """
        metrics = self.get_metrics_dict()
        return (
            f"Score: {metrics['score']:.2f} | "
            f"Path: {metrics['total_movement_ft']:.1f}ft | "
            f"Time: {metrics['A']:.1f}min | "
            f"Missed: {metrics['B']}"
        )


if __name__ == "__main__":
    # Test the scoring system
    print("=== IARC Scoring System Test ===\n")
    
    # Test the formula components
    G = 1  # Green zone width in squares
    W = calculate_path_width(G)
    print(f"Green Zone Squares (G): {G}")
    print(f"Path Width (W): {W} ft")
    
    movement = 300.0  # Total movement in feet
    L = calculate_path_length(movement)
    print(f"\nTotal Movement: {movement} ft")
    print(f"Path Length (L): {L} ft")
    
    # Test complete score calculation
    B = 5   # Missed mines
    A = 2.0 # Minutes
    N = 0.0 # Weight penalty
    
    score = calculate_score(W, L, B, A, N)
    print(f"\nMissed Mines (B): {B}")
    print(f"Time (A): {A} minutes")
    print(f"Weight Penalty (N): {N} oz")
    print(f"\nFinal Score: {score:.4f}")
    
    # Test ScoreTracker
    print("\n=== ScoreTracker Test ===")
    config = {
        "scoring": {
            "green_zone_squares": 1,
            "weight_penalty_oz": 0,
        }
    }
    tracker = ScoreTracker(config)
    
    # Simulate some movement
    positions = [(0, 0), (0, 10), (0, 20), (0, 30)]
    tracker.update(positions, 0.0)
    
    positions = [(10, 0), (10, 10), (10, 20), (10, 30)]
    tracker.update(positions, 1.0)
    
    print(f"Metrics: {tracker.get_metrics_dict()}")
    print(f"Display: {tracker.format_score_string()}")
