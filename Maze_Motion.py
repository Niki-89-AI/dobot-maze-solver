#pixel point in the camera image → converted to robot coordinates with M → robot physically moves there.

import numpy as np
import cv2  
import time
import pydobot
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities_2 import move_to_home, move_to_specific_position, get_current_pose
import json
import os

def load_path_pixels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data["unwarped_path_pixels"]

#calibration matrix (affine)
M = np.array([
    [-1.79726959e-02, -4.27012106e-01, 4.43102425e+02],
    [-4.13501903e-01,  7.86829912e-03, 1.38241276e+02]
], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
    [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
    [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-40)                           
    time.sleep(1)

def main():
    device = pydobot.Dobot(port="COM5")
    device.speed(50, 50)
    move_to_home(device)
    time.sleep(2)

    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "..", "part_2_maze_solution", "solution_path_points_unwarped.json")

    pixel_coords = load_path_pixels(json_path)

    for (u, v) in pixel_coords:
        move_robot_point(device ,M, u, v) 

    device.close() 
    
if __name__ == "__main__":
    main()
