"""
Safe Path Finder for IARC 2026 Simulation
Calculates the optimal safe path through the detected minefield.

Goal: Maximize path width (clearance) and minimize path length.
Approach: Grid-based distance transform and bottleneck pathfinding.
"""

import math
import numpy as np
from typing import List, Tuple, Dict

class SafePathFinder:
    def __init__(self, arena_width: float, arena_height: float, resolution: float = 1.0):
        """
        Initialize the path finder.
        
        Args:
            arena_width: Width in feet
            arena_height: Height in feet
            resolution: Grid resolution in feet (lower = more precise but slower)
        """
        self.width = arena_width
        self.height = arena_height
        self.res = resolution
        
        # Grid dimensions
        self.cols = int(arena_width / resolution)
        self.rows = int(arena_height / resolution)
        
        # Coordinate grids
        self.y_dom, self.x_dom = np.indices((self.rows, self.cols))
        # Convert to world coords (center of cells)
        self.x_world = (self.x_dom * resolution) + (resolution / 2)
        self.y_world = (self.y_dom * resolution) + (resolution / 2)
        
        # State
        self.current_mines = []
        self.current_path = []  # List of (x, y) tuples
        self.path_clearance = 0.0 # Bottleneck radius in feet
        self.path_length = 0.0
        
    def update(self, detected_mines: List[Tuple[float, float]]):
        """
        Update the safe path based on new mine detections.
        Constrains the search to the bounding box of detected mines.
        
        Args:
            detected_mines: List of (x, y) mine coordinates
        """
        if not detected_mines:
            self.current_path = []
            self.path_clearance = 0.0
            self.path_length = 0.0
            return

        mines_arr = np.array(detected_mines)
        
        # 1. Determine Bounding Box of Known Area
        min_x, min_y = np.min(mines_arr, axis=0)
        max_x, max_y = np.max(mines_arr, axis=0)
        
        # Add a small buffer to the bbox? The user said "smallest rectangle that fits...".
        # If we treat the bbox edges as "walls", clearance checks will handle it.
        # But if we strictly clip to min/max, the path must be INSIDE.
        # This implies we can't go around the outermost mines on the outside?
        # "create a path through that" implies going *through* the field.
        # We will strictly limit the grid to this ROI.
        
        # Convert bbox to grid indices
        # Ensure we are within arena bounds
        c_min = max(0, int(min_x / self.res))
        c_max = min(self.cols - 1, int(max_x / self.res))
        r_min = max(0, int(min_y / self.res))
        r_max = min(self.rows - 1, int(max_y / self.res))
        
        # If bbox is too small (e.g. 1 mine), no path
        if c_max <= c_min or r_max <= r_min:
            self.current_path = []
            self.path_length = 0.0
            self.path_clearance = 0.0
            return
            
        # Define the Region of Interest (ROI) dimensions
        roi_rows = r_max - r_min + 1
        roi_cols = c_max - c_min + 1
        
        # Grid subsets for the ROI
        # Correctly slice the coordinate grids
        roi_y_world = self.y_world[r_min:r_max+1, c_min:c_max+1]
        roi_x_world = self.x_world[r_min:r_max+1, c_min:c_max+1]
        
        # 2. Calculate Clearance Map for ROI
        # Flatten for broadcasting
        roi_points = np.stack([roi_x_world.ravel(), roi_y_world.ravel()], axis=1)
        
        # Distance to mines
        # We can optimize by only considering mines IN or NEAR the bbox, 
        # but with N=400 it's fast enough to check all.
        dists = np.sqrt(np.sum((roi_points[:, np.newaxis, :] - mines_arr[np.newaxis, :, :])**2, axis=2))
        min_dists = np.min(dists, axis=1)
        
        # Distance to ROI Boundaries (The "Known Area" limits)
        # Treat the bbox edges as obstacles/walls
        # User: "smallest rectangle that fits... path through that" implies stay inside.
        dist_to_top = max_y - roi_points[:, 1]
        dist_to_bottom = roi_points[:, 1] - min_y
        dist_to_left = roi_points[:, 0] - min_x
        dist_to_right = max_x - roi_points[:, 0]
        
        # We generally want to flow Left->Right, so we don't necessarily penalize x-distance 
        # as a hard "clearance" for the *path definition* (bottleneck is width), 
        # but we must stay inside.
        # The bottleneck is usually defined by Y-clearance (vertical constriction).
        # We'll treat Top/Bottom as clearance limits.
        
        min_dists = np.minimum(min_dists, dist_to_top)
        min_dists = np.minimum(min_dists, dist_to_bottom)
        
        self.clearance_map_roi = min_dists.reshape((roi_rows, roi_cols))
        
        # 3. Find Maximum Bottleneck Capacity in ROI
        unique_vals = np.sort(np.unique(self.clearance_map_roi))
        
        low = 0
        high = len(unique_vals) - 1
        best_idx = 0
        
        while low <= high:
            mid = (low + high) // 2
            threshold = unique_vals[mid]
            
            # Use ROI-specific connectivity check
            if self._check_connectivity_roi(self.clearance_map_roi, threshold):
                best_idx = mid
                low = mid + 1
            else:
                high = mid - 1
        
        self.path_clearance = unique_vals[best_idx]
        
        # 4. Find Shortest Path
        roi_path = self._find_shortest_path_roi(self.clearance_map_roi, self.path_clearance)
        
        # Convert local ROI indices to World Coords
        # _find_shortest_path_roi returns list of (r, c) local indices
        self.current_path = []
        for r, c in roi_path:
            # ROI r,c -> Global grid r,c -> World x,y
            global_r = r + r_min
            global_c = c + c_min
            wx = self.x_world[global_r, global_c]
            wy = self.y_world[global_r, global_c]
            self.current_path.append((wx, wy))
            
        self.path_length = self._calculate_path_length(self.current_path)

    def _check_connectivity_roi(self, clearance_map, threshold):
        """BFS on ROI map."""
        rows, cols = clearance_map.shape
        mask = clearance_map >= threshold
        
        queue = []
        visited = np.zeros_like(mask, dtype=bool)
        
        # Start at first column of ROI
        for r in range(rows):
            if mask[r, 0]:
                queue.append((r, 0))
                visited[r, 0] = True
                
        idx = 0
        while idx < len(queue):
            r, c = queue[idx]
            idx += 1
            if c == cols - 1:
                return True
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc] and mask[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return False

    def _find_shortest_path_roi(self, clearance_map, threshold):
        """Find path on ROI map."""
        rows, cols = clearance_map.shape
        mask = clearance_map >= threshold
        
        queue = []
        visited = np.zeros_like(mask, dtype=bool)
        parent = {}
        
        for r in range(rows):
            if mask[r, 0]:
                queue.append((r, 0))
                visited[r, 0] = True
                parent[(r, 0)] = None
        
        end_node = None
        idx = 0
        while idx < len(queue):
            r, c = queue[idx]
            idx += 1
            if c == cols - 1:
                end_node = (r, c)
                break
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc] and mask[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
                        parent[(nr, nc)] = (r, c)
        
        if end_node is None:
            return []
            
        # Reconstruct path (local indices)
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr) # (r, c)
            curr = parent[curr]
            
        return path[::-1]

    def _calculate_path_length(self, path):
        if not path:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            length += math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
        return length

    def get_path_metrics(self):
        """Return metrics for the scoring function."""
        # Bottleneck radius = clearance.
        # Green Zone Width (squares).
        # IARC rules: W = 2 * (1 + 2 * G)
        # G is "green zone width in squares".
        # This implies our clearance maps to "squares".
        # If clearance is radius efficiently, diameter = 2*r.
        # Let's assume "Green Zone Width in Squares" correlates to our clearance diameter.
        # But we can just pass the raw clearance and let the scorer handle it,
        # or convert it here.
        # Let's return raw metrics.
        return {
            "clearance_ft": self.path_clearance,
            "length_ft": self.path_length,
            "path_points": self.current_path
        }
