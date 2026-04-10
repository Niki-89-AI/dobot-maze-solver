#!/usr/bin/env python3
"""
solve_maze.py

Reads a maze JSON (like the one you posted), allows choosing start/end,
solves the maze under the rule:
  - Traversable cells have value 1 only.
  - Start and End cells may be 0.
  - Movement is 4-connected (up/down/left/right).

Outputs:
  - A PNG with the path drawn on the original image.
  - A JSON with the path pixel points (including start/end pixel coordinates).

Usage examples:
  python solve_maze.py maze.json
  python solve_maze.py maze.json --start green --end red
  python solve_maze.py maze.json --start 9,1 --end 1,9
  python solve_maze.py maze.json --out-image path_overlay.png --out-json path_points.json
"""

import json
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    value: int
    center_px: Tuple[int, int]


def load_maze(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

###########TO DO
def cells_to_grid(cells_json: List[Dict[str, Any]], rows: int, cols: int) -> List[List[Cell]]:
    """
    Convert list of cell dictionaries into 2D grid of Cell objects.
    """
    #empty rows x cols grid
    grid: List[List[Optional[Cell]]] = [[None for _ in range(cols)] for _ in range(rows)]

    #fills the grid
    for c in cells_json:
        r = int(c["row"])
        col = int(c["col"])
        value = int(c["value"])
        cx, cy = c["center_px"]
        grid[r][col] = Cell(row=r, col=col, value=value, center_px=(int(cx), int(cy)))

    # no cell is missing?
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                raise ValueError(f"Missing cell at row={r}, col={c}")

    return grid  # type: ignore


###########TO DO
def nearest_cell_by_pixel(grid: List[List[Cell]], x: float, y: float) -> Tuple[int, int]:
    """
    Return (row, col) of the cell whose center is nearest to (x, y).
    """
    best_rc = (0, 0)
    best_d2 = float("inf")

    #BFS needs grid cell coordinates
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            cx, cy = grid[row][col].center_px
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_rc = (row, col)

    return best_rc



def parse_start_end(
    grid, data, start_arg, end_arg
):
    circles = data.get("circles", [])
    green = next((c for c in circles if c.get("color") == "green"), None)
    red   = next((c for c in circles if c.get("color") == "red"),   None)
    if green is None or red is None:
        raise ValueError("JSON must include green and red circles.")

    green_px = (float(green["center"][0]), float(green["center"][1]))
    red_px   = (float(red["center"][0]),   float(red["center"][1]))

    # Defaults: cells nearest to the green/red dots
    default_start = nearest_cell_by_pixel(grid, *green_px)
    default_end   = nearest_cell_by_pixel(grid, *red_px)

    def parse_point(arg, default_rc):
        if not arg:
            return default_rc
        a = arg.strip().lower()
        if a == "green": return default_start
        if a == "red":   return default_end
        if "," in a:
            r_str, c_str = a.split(",", 1)
            return (int(r_str), int(c_str))
        raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

    start_rc = parse_point(start_arg, default_start)
    end_rc   = parse_point(end_arg,   default_end)

    # Choose which circle pixel to use as the start/end anchor:
    # 1) If user explicitly said 'green' or 'red', honor that.
    # 2) If user gave coordinates, anchor to whichever circle is closer to that cell center.
    def choose_anchor(which, rc):
        if which and which.strip().lower() in ("green", "red"):
            return green_px if which.strip().lower() == "green" else red_px
        cx, cy = grid[rc[0]][rc[1]].center_px
        dg = (cx - green_px[0])**2 + (cy - green_px[1])**2
        dr = (cx - red_px[0])**2   + (cy - red_px[1])**2
        return green_px if dg <= dr else red_px

    start_dot = choose_anchor(start_arg, start_rc)
    end_dot   = choose_anchor(end_arg,   end_rc)

    # Return anchors in the same order as chosen start/end
    return start_rc, end_rc, (int(start_dot[0]), int(start_dot[1])), (int(end_dot[0]), int(end_dot[1]))

###########TO DO
def bfs_path(
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Find shortest path using BFS.
    Traversable cells have value 1.
    Start and end cells may be 0.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    sr, sc = start_rc
    er, ec = end_rc

    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= er < rows and 0 <= ec < cols):
        return None

    #Initialize BFS
    q = deque() # queue of cells to explore
    q.append((sr, sc))

    visited = set() #prevents revisiting
    visited.add((sr, sc))

    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sr, sc): None} #stores where each cell came from so path can be reconstructed later

    #possible moves 
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    #BFS loop
    while q:
        r, c = q.popleft()
        # check if goal is reached
        if (r, c) == (er, ec): # if current cell is the end cell, reconstruct the path
            path = []
            cur: Optional[Tuple[int, int]] = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        # compute neighboring cell
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols): # inbounds?
                continue
            if (nr, nc) in visited:
                continue

            # Allowed if path cell, or if it is the end cell
            if grid[nr][nc].value != 1 and (nr, nc) != (er, ec): # start and end may be 0
                continue

            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))

    return None


def draw_path_on_image(
    image_path: str,
    out_image_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
    line_width: int = 5,
) -> None:
    """
    Draws the polyline from the green dot to the first cell center,
    through the path cells, and finally to the red dot.
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Build pixel polyline
    poly: List[Tuple[int, int]] = []

    # Start at the green circle center
    poly.append(start_circle_px)

    # Then the centers of each cell on the path (in order)
    for (r, c) in cell_path:
        poly.append(grid[r][c].center_px)

    # Finish at the red circle center
    poly.append(end_circle_px)

    # Draw the path
    draw.line(poly, width=line_width, fill=(255, 0, 0, 255))  # solid red path
    # Mark endpoints for clarity
    r_rad = max(6, line_width * 2)
    g_rad = max(6, line_width * 2)
    gx, gy = start_circle_px
    rx, ry = end_circle_px
    draw.ellipse((gx - g_rad, gy - g_rad, gx + g_rad, gy + g_rad), outline=(0, 255, 0, 255), width=3)
    draw.ellipse((rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad), outline=(255, 0, 0, 255), width=3)

    img.save(out_image_path)


def write_path_json(
    out_json_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
) -> None:
    """
    Writes a JSON containing:
      - start/end pixel coordinates (green/red dot centers)
      - start/end cells
      - path as cells (row,col,value,center_px)
      - path as pixels (list of [x,y] including the dots at beginning/end)
      - length (number of moves)
    """
    # path cells expanded
    path_cells_expanded = [
        {
            "row": r,
            "col": c,
            "value": grid[r][c].value,
            "center_px": [grid[r][c].center_px[0], grid[r][c].center_px[1]],
        }
        for (r, c) in cell_path
    ]

    # pixel polyline that matches drawn line (start dot -> all cell centers -> end dot)
    pixel_polyline: List[List[int]] = []
    pixel_polyline.append([start_circle_px[0], start_circle_px[1]])
    for (r, c) in cell_path:
        x, y = grid[r][c].center_px
        pixel_polyline.append([x, y])
    pixel_polyline.append([end_circle_px[0], end_circle_px[1]])

    out = {
        "start_cell": {"row": start_rc[0], "col": start_rc[1]},
        "end_cell": {"row": end_rc[0], "col": end_rc[1]},
        "start_circle_px": [start_circle_px[0], start_circle_px[1]],
        "end_circle_px": [end_circle_px[0], end_circle_px[1]],
        "path_cells": path_cells_expanded,
        "path_pixels": pixel_polyline,
        "moves": max(0, len(cell_path) - 1),  # moves between cells (excludes dot-to-cell hops)
        "notes": "Path respects rule: 1-only traversal; start/end cells may be 0.",
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Solve a grid maze from JSON and draw the path.")
    ap.add_argument("--json_path", default= "part_2_maze_solution/result.json" , help="Input maze JSON path.")
    ap.add_argument("--start", help="Start: 'green' (default), 'red', or 'row,col'", default='green')
    ap.add_argument("--end", help="End: 'red' (default), 'green', or 'row,col'", default='red')
    ap.add_argument("--out-image", help="Output image with path (PNG).",
                    default="part_2_maze_solution/solution_overlay.png")
    ap.add_argument("--out-json", help="Output JSON with path pixel points.",
                    default="part_2_maze_solution/solution_path_points.json")
    ap.add_argument("--line-width", type=int, default=5, help="Path line width on image.")
    args = ap.parse_args()

    data = load_maze(args.json_path)
    rows = int(data["grid_rows"])
    cols = int(data["grid_cols"])
    grid = cells_to_grid(data["cells"], rows, cols)

    start_rc, end_rc, start_circle_px, end_circle_px = parse_start_end(
        grid, data, args.start, args.end
    )

    path = bfs_path(grid, start_rc, end_rc)
    if path is None:
        raise SystemExit("No feasible path found under the given rules.")

    # Draw overlay on the image
    image_path = data["input"]
    if not Path(image_path).exists():
        raise SystemExit(f"Input image not found at: {image_path}")

    draw_path_on_image(
        image_path=image_path,
        out_image_path=args.out_image,
        cell_path=path,
        grid=grid,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
        line_width=args.line_width,
    )

    # Write path JSON
    write_path_json(
        out_json_path=args.out_json,
        cell_path=path,
        grid=grid,
        start_rc=start_rc,
        end_rc=end_rc,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
    )

    print(f"Done.\n - Path image: {args.out_image}\n - Path JSON: {args.out_json}")


if __name__ == "__main__":
    main()
