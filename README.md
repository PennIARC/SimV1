# SimV1

2D top-down simulation for testing multi-UAV exploration and path planning.

---

## GreedyBottleneckPlanner

### Overview
`GreedyBottleneckPlanner` finds grid paths that prioritize progress toward the goal and use the path bottleneck (minimum clearance) to break ties. It also provides deterministic exploration targets for a 4-UAV two-altitude scouting strategy.

### API
- Constructor:
  - `GreedyBottleneckPlanner(height: int, width: int, start_cells: List[Tuple[int,int]], goal_cells: List[Tuple[int,int]])`
- Planning:
  - `plan(confidence_map: List[List[float]], clearance_map: List[List[float]]) -> Optional[Dict]`
    - Returns `None` or a dict containing keys like `"path"`, `"bottleneck"`, `"length"`, `"reached"` and when not reached `"persistent_path"`.
- Exploration targets:
  - `suggest_exploration_targets(path, confidence_map, clearance_map, mines_detected_high, sensing_radius_high=3, sensing_radius_low=1) -> List[Tuple[int,int]]`
    - Returns 4 deterministic waypoints `[UAV0, UAV1, UAV2, UAV3]`.

### Key constants & planner state
- `CLEARANCE_THRESHOLD` (default `1.0`): minimum clearance for a cell to be considered traversable.
- `CONFIDENCE_THRESHOLD` (default `0.5`): minimum confidence for a cell to be treated as observed.
- Persistent state:
  - `self.persistent_best`: stored best safe prefix when the planner cannot reach the goal.
  - `self.persistent_bottleneck`: the bottleneck value for `persistent_best`.
  - `self.scan_front_x`: persistent (monotonic) scanning front used by high-alt UAV logic to avoid retreating scan columns.

### `plan(...)`
- Inputs:
  - `confidence_map[y][x]` and `clearance_map[y][x]`.
  - Planner considers a cell usable only if `confidence_map[y][x] > CONFIDENCE_THRESHOLD` and `clearance_map[y][x] > CLEARANCE_THRESHOLD`.
- Objective:
  - Prefer progress toward the goal (smaller heuristic distance). When multiple candidate frontiers are comparable w.r.t. the heuristic, prefer the path with a larger bottleneck (safer route).
- Heap ordering:
  - Open set items are considered with primary ordering by heuristic distance-to-goal and secondary ordering by larger bottleneck. Conceptually: `(heuristic_distance_to_goal, -bottleneck, x, y, bottleneck)` so the planner biases toward cells closer to the goal and uses bottleneck as a deterministic tiebreaker.
- Seeding:
  - Seeds from `start_cells` if they are observable & safe; has a local radius fallback around start cells; final fallback seeds a set of columns if still empty.
- Expansion:
  - For each popped cell, it tries 4-neighbors; updates the neighbor's best bottleneck (min of current bottleneck and neighbor clearance) and re-inserts if improved.
- Termination:
  - If the search expands a cell that is in `goal_cells`, it reconstructs and returns that path with `"reached": True`. That returned path is the best-reaching path given the traversability constraints at that time.
- Best-partial:
  - If the search exhausts without reaching a goal, the algorithm keeps a running `best_partial` (candidate cell selected by heuristic then bottleneck) and reconstructs/reports a partial path. It also calls `_update_persistent` to possibly update `self.persistent_best`.
- Return dict:
  - On success: `{"path": path, "bottleneck": int(cw), "length": len(path), "reached": True}`
  - On failure: `{"path": path, "bottleneck": bw, "length": len(path), "reached": False, "persistent_path": self.persistent_best}`

### `_update_persistent(...)`: rules for storing a persistent best
- Only considers the safe prefix: contiguous prefix from path start where `clearance > CLEARANCE_THRESHOLD`.
- If there's no existing `persistent_best`, store safe prefix.
- Otherwise compare:
  1. Prefer the one whose safe-prefix endpoint is closer to goal (smaller heuristic distance).
  2. If equal distance, prefer larger bottleneck.
  3. If equal bottleneck, prefer longer safe prefix.
- Purpose: provide a useful fallback / visible “best so far” option when direct goal reachability is not found.

### `suggest_exploration_targets(...)`: determine the UAV exploration strategy
- Purpose: return 4 deterministic grid waypoints for 4 UAVs:
  - UAV0, UAV1: high-alt scanners (upper and lower halves of arena).
  - UAV2, UAV3: low-alt inspectors (confirming/inspecting cells reported by high-alt).
- Inputs of interest:
  - `path`: current planner path (may be empty).
  - `confidence_map`: used to identify cells seen by high-alt (>=0.5) vs unknown.
  - `clearance_map`: used to avoid unsafe cells.
  - `mines_detected_high`: list of mines the high-alt sensors have reported (used for density estimation).
  - `sensing_radius_high/low`: radii for density / spacing calculations.

#### Behavior details
- Anchor selection:
  - If `path` exists, choose the path point closest to the goal (by heuristic) as `anchor`.
  - Else fall back to `self.persistent_best` end or map center.
- High-alt logic (UAV0 / UAV1):
  - Compute a desired next column (`desired_next`) one column toward the average goal x from the anchor.
  - Use the persistent `self.scan_front_x` and set `self.scan_front_x = max(self.scan_front_x, desired_next)` so the high-alt scanning column only advances forward (no retreat/backtracking).
  - For each high-alt UAV pick a safe cell in that column confined to its half of the map (upper / lower). Search order is deterministic using `offsets_1d` for vertical and limited horizontal adjustments, and falls back to scanning nearby columns deterministically.
  - If no safe cell found, placeholders inside bounds are used.
- Low-alt logic (UAV2 / UAV3):
  - Build a candidate list from cells observed by high-alt (`confidence_map[y][x] >= 0.5`) and with positive clearance.
  - Each candidate is scored by a deterministic tuple:
    - `(density, dist_goal, dist_path, -clearance, cx, cy)` where:
      - `density` = count of high-alt-detected mines within `sensing_radius_low` (lower is better).
      - `dist_goal` = planner heuristic distance-to-goal (closer is better).
      - `dist_path` = Manhattan distance to the planner path (closer is better).
      - `-clearance` prefers higher clearance.
  - If no high-alt observed cells exist, the same key is used on all scanned (clearance>0) cells as a fallback.
  - Selection:
    - Pick the best candidate (`u2`), then pick next best candidate that satisfies a minimum y-spacing from `u2` (`min_spacing = max(1, round(sensing_radius_low*2 - 1))`) to avoid low-alt overlap.
    - If no suitable partner found, search vertically near `u2` with spacing enforcement; final fallback uses `fill_nearby`: a deterministic search radiating from a reference point (average detected high-alt mine location or map center).
  - Low-alt waypoints are ensured distinct from each other, but overlap with high-alt waypoints is allowed.
- Determinism:
  - Uses deterministic offset order `offsets_1d` (0, +1, -1, +2, -2, ...) for both horizontal and vertical searches.
  - Sorting keys include the cell coordinates last to provide deterministic tie breaks.

### Edge cases & safeguards
- Empty / invalid maps: returns `[(0,0), (0,0), (0,0), (0,0)]`.
- All candidate selection and fallbacks are deterministic, no randomness is used.
- All returned waypoints are clamped to integer grid coordinates and in-bounds before returning.

### Return format
- `suggest_exploration_targets` returns a `List[Tuple[int,int]]` of length 4 corresponding to UAV0..UAV3.

### Design rationale
- Bottleneck objective (maximin) biases planner away from narrow, risky corridors (safer operational behavior).
- Persistent safe-prefix storage (`persistent_best`) gives the system useful partial plans when full reachability is not yet observed.
- Two-altitude division:
  - High-alt scouts provide broad coverage (large detection radius) and produce a `confidence_map`.
  - Low-alt inspectors use high-alt observations to prioritize confirming doubtful cells (lower detection radius, higher fidelity).
- Determinism is important for reproducibility in experiments and analysis; using deterministic offsets + tie-break keys avoids nondeterministic behavior in demos.
- `self.scan_front_x` prevents high-alt units from retreating scan columns, producing steady forward progress.

