"""
IARC Mission 10 Scoring System
Implements the official IARC scoring formula.

Score = 150000 * W / [(1+B) * L * (1 + 7*A + 100*N)]

Where:
    W = path width in feet = 2 × (1 + 2G), G = green zone width in grid squares
    L = path length in feet = cell_size × number_of_steps
    B = number of missed mines IN THE GREEN ZONE (not on blue path, but within G squares)
    A = minutes to complete the scan
    N = ounces over the one-pound drone weight limit

Key rules:
    - Mine on blue path → FATAL → score = 0
    - B only counts mines in the green buffer, NOT all undetected mines
    - L is the path Seriah walks (blue path length), NOT total drone flight distance
    - Optimal G is found by trying G=0..MAX_G (wider path ↑W but may ↑B)
"""

from typing import List, Tuple, Dict, Set

# Maximum green zone width to try (in grid squares)
MAX_GREEN_ZONE = 20


def mines_ft_to_grid(
    mines_list: List,
    cell_size_ft: float,
    grid_w: int,
    grid_h: int,
) -> Set[Tuple[int, int]]:
    """
    Convert mine positions from feet to 2ft competition grid cells.

    Args:
        mines_list:   list of [mx_ft, my_ft] in feet.
        cell_size_ft: competition cell size (2.0).
        grid_w:       grid width  in cells (150).
        grid_h:       grid height in cells (40).

    Returns:
        Set of (gx, gy) grid cells containing a mine.
    """
    cell = int(cell_size_ft)
    mine_grid: Set[Tuple[int, int]] = set()
    for m in mines_list:
        mx_ft, my_ft = m[0], m[1]
        gx = min(max(0, int(mx_ft) // cell), grid_w - 1)
        gy = min(max(0, int(my_ft) // cell), grid_h - 1)
        mine_grid.add((gx, gy))
    return mine_grid


def compute_green_zone(
    blue_cells: Set[Tuple[int, int]],
    G: int,
    grid_h: int,
) -> Set[Tuple[int, int]]:
    """
    Expand the blue path by G cells in the y-direction (field width)
    to form the green zone buffer.  Cells already on the blue path
    are excluded.

    Args:
        blue_cells: set of (x, y) cells on the blue path.
        G:          green zone half-width in grid squares.
        grid_h:     grid height in cells (bounds check).

    Returns:
        Set of (x, y) cells that form the green zone (excluding blue).
    """
    if G <= 0:
        return set()
    green: Set[Tuple[int, int]] = set()
    for px, py in blue_cells:
        for dy in range(-G, G + 1):
            ny = py + dy
            if 0 <= ny < grid_h:
                cell = (px, ny)
                if cell not in blue_cells:
                    green.add(cell)
    return green


def compute_score(
    path: List[Tuple[int, int]],
    mines_list: List,
    A_minutes: float,
    grid_w: int,
    grid_h: int,
    N_oz: int = 0,
    cell_size_ft: float = 2.0,
) -> Tuple[float, int, Dict]:
    """
    Compute the IARC Mission 10 competition score for a given path.
    Tries G = 0 .. MAX_GREEN_ZONE-1 and returns the best (score, G, info).

    Args:
        path:         list of (grid_x, grid_y) on the 2ft competition grid.
        mines_list:   list of [mx_ft, my_ft] in FEET (truth or detected).
        A_minutes:    scan time in minutes.
        grid_w:       planner grid width  (150).
        grid_h:       planner grid height (40).
        N_oz:         ounces over 1-lb drone weight limit.
        cell_size_ft: competition cell size (2.0).

    Returns:
        (best_score, best_G, info_dict)
        info_dict keys: G, W, L, B, A, N, steps   (or "fatal" if mine on blue path)
    """
    if not path or len(path) < 2:
        return 0.0, 0, {}

    blue_cells = set(path)
    mine_grid = mines_ft_to_grid(mines_list, cell_size_ft, grid_w, grid_h)

    # Fatal check: mine on blue path → score = 0
    fatal = blue_cells & mine_grid
    if fatal:
        return 0.0, 0, {"fatal": True, "fatal_cell": next(iter(fatal))}

    # L = path length in feet.  Steps = len(path) - 1, each step = cell_size_ft.
    steps = len(path) - 1
    L = cell_size_ft * steps
    if L <= 0:
        return 0.0, 0, {}

    time_factor = 1.0 + 7.0 * A_minutes + 100.0 * N_oz

    best_score = 0.0
    best_G = 0
    best_info: Dict = {}

    for G in range(0, MAX_GREEN_ZONE):
        W = cell_size_ft * (1 + 2 * G)  # width in feet

        # Count missed mines (B) in green zone
        if G > 0:
            green_cells = compute_green_zone(blue_cells, G, grid_h)
            B = len(mine_grid & green_cells)
        else:
            B = 0

        denom = (1 + B) * L * time_factor
        if denom <= 0:
            continue
        score = 150000.0 * W / denom

        if score > best_score:
            best_score = score
            best_G = G
            best_info = {
                "G": G, "W": W, "L": L, "B": B,
                "A": A_minutes, "N": N_oz, "steps": steps,
            }

    return best_score, best_G, best_info


def format_score(score: float, G: int, info: Dict, algorithm: str = "", seed=None) -> str:
    """
    Format score + metrics into a readable multi-line string for terminal output.
    """
    header = "=== COMPETITION SCORE"
    if algorithm:
        header += f" [{algorithm}]"
    if seed is not None:
        header += f" seed={seed}"
    header += " ==="

    if info.get("fatal"):
        return f"{header}\n  FATAL: mine on blue path at {info.get('fatal_cell')} → Score = 0"

    lines = [
        header,
        f"  Score: {score:.2f}  |  G={G}  W={info.get('W', 0)}ft"
        f"  L={info.get('L', 0)}ft  B={info.get('B', 0)}"
        f"  A={info.get('A', 0):.2f}min  steps={info.get('steps', 0)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== scoring.py self-test ===\n")

    # Straight path at y=20 across full 150-cell field
    test_path = [(x, 20) for x in range(150)]

    # Mine near path but not on it (y=22, G>=2 will catch it)
    mines = [[50.0, 44.0]]  # grid (25, 22)
    s, g, info = compute_score(test_path, mines, A_minutes=3.0, grid_w=150, grid_h=40)
    print(format_score(s, g, info, algorithm="test"))

    # Mine ON path → fatal
    mines_fatal = [[40.0, 40.0]]  # grid (20, 20) — on path
    s2, g2, info2 = compute_score(test_path, mines_fatal, A_minutes=3.0, grid_w=150, grid_h=40)
    print(format_score(s2, g2, info2, algorithm="fatal-test"))

    # No mines
    s3, g3, info3 = compute_score(test_path, [], A_minutes=3.0, grid_w=150, grid_h=40)
    print(format_score(s3, g3, info3, algorithm="clean"))
