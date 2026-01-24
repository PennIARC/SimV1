"""
Algorithm Package for IARC 2026 Simulation
Provides modular path-finding algorithms for drone swarms.
"""

from algorithms.base import PathAlgorithm
from algorithms.simple_spacing import SimpleSpacingAlgorithm
from algorithms.greedy_path import GreedyPathAlgorithm
from algorithms.astar_path import AStarPathAlgorithm


# Registry of available algorithms
ALGORITHMS = {
    "simple_spacing": SimpleSpacingAlgorithm,
    "greedy_path": GreedyPathAlgorithm,
    "astar_path": AStarPathAlgorithm,
}


def get_algorithm(name: str) -> type:
    """
    Get an algorithm class by name.
    
    Args:
        name: Algorithm name (simple_spacing, greedy_path, astar_path)
    
    Returns:
        Algorithm class (not instance)
    
    Raises:
        ValueError: If algorithm name is not found
    """
    if name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    
    return ALGORITHMS[name]


def list_algorithms() -> list:
    """
    Get list of available algorithm names.
    
    Returns:
        List of algorithm name strings
    """
    return list(ALGORITHMS.keys())
