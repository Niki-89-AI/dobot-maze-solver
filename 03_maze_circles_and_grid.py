# #!/usr/bin/env python3

# This script processes the warped maze image to detect start/end circles, 
# identify walls, divide the maze into grid cells with values,
# and generate a structured JSON representation for path planning.

#Input: maze_warp.png


# Workflow:
#   1) Detect circles via HoughCircles and classify color (red/green/unknown).
#      Saves a circles overlay.
#   2) Build grid over the maze, detect black walls, and compute per-cell occupancy:
#         value = 1 -> no wall in the cell
#         value = 0 -> wall present in the cell (>= threshold % of wall pixels)
#      Saves:
#         - grid overlay (lines only)
#         - annotated grid overlay (0/1 at cell centers)
#         - walls mask (255 = wall)
#   3) Write a single JSON with:
#         - circles: [ {center:[x,y], radius:r, color:"red/green/unknown"} ]
#         - grid_size_px, grid_rows, grid_cols, threshold_percent
#         - paths to overlays and mask
#         - cells: [ {row, col, value, center_px:[x,y]} ]

# Usage:
#   python maze_circles_and_grid.py maze.png \
#     --grid 30 --threshold 0.0 \
#     --circles-overlay-out circles_overlay.png \
#     --grid-overlay-out grid_overlay.png \
#     --grid-overlay-annot-out grid_overlay_annot.png \
#     --walls-mask-out walls_mask.png \
#     --json-out result.json \
#     --adaptive 0 --blur 5 --open 0 --close 0 \
#     --font-scale 0.4 --thickness 1
# """

#!/usr/bin/env python3
import argparse
import sys
import json
import numpy as np
import cv2

# ---------------- Circle color detection ----------------

###########TO DO
#convert image to HSV and detect red/green circles to identify start and end points

def detect_color(frame, center, radius):
    """
    Detect dominant color of a circular region.
    Return:
        "green" for green circle
        "red" for red circle
        None if neither is dominant
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cx, cy = int(center[0]), int(center[1])
    r = max(1, int(radius))

    # Circular mask for the ROI
    circ_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(circ_mask, (cx, cy), r, 255, -1)

    # HSV ranges (detect green and red)
    lower_red1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 80, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_count = cv2.countNonZero(cv2.bitwise_and(red_mask, red_mask, mask=circ_mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=circ_mask))

    if green_count > red_count and green_count > 20:
        return "green"
    if red_count > green_count and red_count > 20:
        return "red"
    return None

###########TO DO

#convert image to grayscale, reduce noise, detect circles
def detect_circles_and_overlay(img_bgr, out_path=None):
    """
    Detect red/green circles and return list of circle dicts.
    """
    overlay = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=45,
        minRadius=10,
        maxRadius=40
    )

    circles_info = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for (x, y, r) in circles:
            color_name = detect_color(img_bgr, (x, y), r)
            if color_name is None:
                continue

            circles_info.append({
                "center": [int(x), int(y)],
                "radius": int(r),
                "color": color_name
            })

            draw_color = (0, 255, 0) if color_name == "green" else (0, 0, 255)
            cv2.circle(overlay, (x, y), r, draw_color, 2)
            cv2.circle(overlay, (x, y), 3, draw_color, -1)
            cv2.putText(
                overlay,
                color_name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                draw_color,
                2,
                cv2.LINE_AA
            )

    if out_path is not None:
        cv2.imwrite(out_path, overlay)

    return circles_info

# ---------------- Grid + walls ----------------

###########TO DO
# convert image into binary (255/0)
def binarize_walls(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    """
    Walls should be white (255), free space black (0).
    """
    if adaptive:
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10
        )
    else:
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    return mask

def morph(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    m = mask.copy()
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_open+1, 2*k_open+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_close+1, 2*k_close+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m

#draws grid lines on top of the image
def draw_grid_lines(img: np.ndarray, grid: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = (255, 255, 255)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w-1, y), color, 1)
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h-1), color, 1)
    return out

#each cell gets labeled
def draw_grid_with_values(img: np.ndarray, grid: int, values_mat: np.ndarray,
                          font_scale: float, thickness: int) -> np.ndarray:
    out = draw_grid_lines(img, grid)
    h, w = out.shape[:2]
    gh, gw = values_mat.shape
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            text = str(int(values_mat[gy, gx]))
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = "part_2_maze_solution/maze_warp.png")
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--adaptive", type=int, default=0)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--open", type=int, default=0)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=10.0)

    ap.add_argument("--circles-overlay-out", default="part_2_maze_solution/circles_overlay.png")
    ap.add_argument("--grid-overlay-out", default="part_2_maze_solution/grid_overlay.png")
    ap.add_argument("--grid-overlay-annot-out", default="part_2_maze_solution/grid_overlay_annot.png")
    ap.add_argument("--walls-mask-out", default="part_2_maze_solution/walls_mask.png")
    ap.add_argument("--json-out", default="part_2_maze_solution/result.json")

    ap.add_argument("--font-scale", type=float, default=0.4)
    ap.add_argument("--thickness", type=int, default=1)

    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    grid = max(1, args.grid)

    # 1. Detect circles
    circles_info = detect_circles_and_overlay(img, args.circles_overlay_out)

    # 2a. Grid overlay
    grid_overlay = draw_grid_lines(img, grid)
    cv2.imwrite(args.grid_overlay_out, grid_overlay)

    # 2b. Walls mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur > 0:
        k = max(1, args.blur | 1)
        gray = cv2.medianBlur(gray, k)
    walls_mask = binarize_walls(gray, adaptive=bool(args.adaptive))
    walls_mask = morph(walls_mask, k_open=max(0, args.open), k_close=max(0, args.close))

    # RASE walls where circles are
    for c in circles_info:
        cx, cy = int(c["center"][0]), int(c["center"][1])
        r = int(c["radius"] * 1.2)
        cv2.circle(walls_mask, (cx, cy), r, 0, -1)  # set to 0 (non-wall)

    cv2.imwrite(args.walls_mask_out, walls_mask)

    # 2c. Build grid cells
    h, w = walls_mask.shape[:2]
    gh = (h + grid - 1) // grid
    gw = (w + grid - 1) // grid
    values_mat = np.zeros((gh, gw), dtype=np.uint8)
    cells = []
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            blk = walls_mask[y0:y1, x0:x1]
            wall_pct = (blk > 0).mean() * 100.0 if blk.size > 0 else 0.0
            value = 0 if wall_pct >= args.threshold else 1
            values_mat[gy, gx] = value
            cells.append({
                "row": int(gy),
                "col": int(gx),
                "value": int(value),
                "center_px": [int(cx), int(cy)]
            })

    # 2d. Annotated grid
    grid_overlay_annot = draw_grid_with_values(img, grid, values_mat, args.font_scale, args.thickness)
    cv2.imwrite(args.grid_overlay_annot_out, grid_overlay_annot)

    # 3. Write JSON
    meta = {
        "input": args.input,
        "circles_overlay_path": args.circles_overlay_out,
        "grid_size_px": grid,
        "grid_rows": int(gh),
        "grid_cols": int(gw),
        "threshold_percent": args.threshold,
        "grid_overlay_path": args.grid_overlay_out,
        "grid_overlay_annot_path": args.grid_overlay_annot_out,
        "walls_mask_path": args.walls_mask_out,
        "circles": circles_info,
        "cells": cells
    }
    with open(args.json_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Circles overlay saved to: {args.circles_overlay_out}")
    print(f"Grid overlay saved to: {args.grid_overlay_out}")
    print(f"Walls mask saved to: {args.walls_mask_out}")
    print(f"Annotated grid overlay saved to: {args.grid_overlay_annot_out}")
    print(f"JSON saved to: {args.json_out}")
    print("Legend: cell value 1 = path, 0 = wall")

if __name__ == "__main__":
    main()
