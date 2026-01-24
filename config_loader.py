"""
Configuration Loader for IARC 2026 Simulation
Loads and validates YAML configuration file.
"""

import yaml
import os

CONFIG_FILE = "config.yaml"

# Default configuration values
DEFAULT_CONFIG = {
    "algorithm": "simple_spacing",
    "drone": {
        "count": 4,
        "detection_radius_ft": 5.0,
        "spacing_margin_ft": 1.0,
        "start_x_ft": 5.0,
    },
    "path": {
        "waypoint_tolerance_ft": 1.0,
        "return_after_scan": True,
        "scan_direction": "forward",
    },
    "arena": {
        "width_ft": 300.0,
        "height_ft": 80.0,
    },
    "scoring": {
        "green_zone_squares": 1,
        "weight_penalty_oz": 0,
    },
    "algorithms": {
        "simple_spacing": {
            "overlap_factor": 0.0,
        },
        "greedy_path": {
            "lookahead_steps": 5,
            "mine_weight": 1.0,
            "length_weight": 1.0,
        },
        "astar_path": {
            "grid_resolution_ft": 5.0,
            "heuristic_weight": 1.2,
        },
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Values in override take precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    Falls back to defaults if file doesn't exist or has missing values.
    
    Args:
        config_path: Optional path to config file. Defaults to CONFIG_FILE in current directory.
    
    Returns:
        Dictionary containing all configuration values.
    """
    if config_path is None:
        # Look for config in same directory as this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, CONFIG_FILE)
    
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                config = deep_merge(DEFAULT_CONFIG, file_config)
                print(f"[Config] Loaded configuration from {config_path}")
            else:
                print(f"[Config] Empty config file, using defaults")
        except yaml.YAMLError as e:
            print(f"[Config] Error parsing config file: {e}")
            print("[Config] Using default configuration")
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
            print("[Config] Using default configuration")
    else:
        print(f"[Config] Config file not found at {config_path}, using defaults")
    
    return config


def get_algorithm_config(config: dict) -> dict:
    """
    Get the configuration for the currently selected algorithm.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        Dictionary containing algorithm-specific configuration
    """
    algorithm_name = config.get("algorithm", "simple_spacing")
    algorithms_config = config.get("algorithms", {})
    return algorithms_config.get(algorithm_name, {})


# Singleton pattern for config caching
_cached_config = None
_config_mtime = None


def get_config(force_reload: bool = False) -> dict:
    """
    Get configuration with caching. Reloads if file has been modified.
    
    Args:
        force_reload: Force reload even if cached
    
    Returns:
        Configuration dictionary
    """
    global _cached_config, _config_mtime
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, CONFIG_FILE)
    
    try:
        current_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else None
    except OSError:
        current_mtime = None
    
    if force_reload or _cached_config is None or current_mtime != _config_mtime:
        _cached_config = load_config(config_path)
        _config_mtime = current_mtime
    
    return _cached_config


if __name__ == "__main__":
    # Test the config loader
    config = load_config()
    print("\n=== Loaded Configuration ===")
    print(yaml.dump(config, default_flow_style=False))
    
    print(f"\nSelected algorithm: {config['algorithm']}")
    print(f"Algorithm config: {get_algorithm_config(config)}")
