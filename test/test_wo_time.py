import random
from typing import List, Tuple
import math
from path_planning.greedy_path_planning import GreedyBottleneckPlannerNoPointers as GreedyBottleneckPlanner

def print_map(path: List[Tuple[int, int]]):
    path_set = set(path)

    print("\nMap legend:")
    print("S = start, G = goal, X = mine, * = path, . = free\n")

    for y in reversed(range(HEIGHT)):
        row = ""
        for x in range(WIDTH):
            if (x, y) in mine_cells:
                row += " X "
            elif (x, y) in path_set:
                row += " * "
            elif x == 0:
                row += " S "
            elif x == WIDTH - 1:
                row += " G "
            else:
                row += " . "
        print(row)
    print()


WIDTH = 30
HEIGHT = 10
NUM_MINES = 40
MINE_RADIUS = 0  # foot safety radius
random.seed(42)

# The entire map is known.
known_map = [[True for _ in range(WIDTH)] for _ in range(HEIGHT)]
clearance_map = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
all_cells = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)]
mine_cells = set(random.sample(all_cells, NUM_MINES))

for y in range(HEIGHT):
    for x in range(WIDTH):
        if (x, y) in mine_cells:
            clearance_map[y][x] = 0.0
            continue

        min_dist = math.inf
        for mx, my in mine_cells:
            d = math.hypot(x - mx, y - my)
            min_dist = min(min_dist, d)

        clearance_map[y][x] = max(0.0, min_dist - MINE_RADIUS)


start_cells = [(0, y) for y in range(HEIGHT)]
goal_cells = [(WIDTH - 1, y) for y in range(HEIGHT)]

planner = GreedyBottleneckPlanner(
    height=HEIGHT,
    width=WIDTH,
    start_cells=start_cells,
    goal_cells=goal_cells
)

result = planner.plan(known_map, clearance_map)

if result is None:
    print("No safe path found.")
else:
    print("Planner result:")
    print(f"  Bottleneck: {result['bottleneck']}")
    print(f"  Path length: {result['length']}")
    print(f"  Path: {result['path']}")

    print_map(result["path"])
