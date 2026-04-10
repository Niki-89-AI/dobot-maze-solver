# dobot-maze-solver
Vision-based maze solving robot using Dobot Magician Lite (Computer Vision + BFS + Robot Control)

🧩 Dobot Maze Solver Robot

This project implements a vision-based autonomous maze solving system using the Dobot Magician Lite robot.
It integrates:
Computer Vision (image processing & detection)
Path Planning using BFS
Coordinate Transformation (pixel → robot)
Robotic Motion Execution

🚀 Features

Detects maze structure from camera input
Converts maze into a grid representation
Computes shortest path using BFS
Maps pixel coordinates to robot workspace
Executes path using robotic arm

🛠️ Tech Stack

Python
OpenCV
NumPy
Dobot API

⚙️ Pipeline

Capture maze image from camera
Apply perspective transformation
Detect walls and free paths
Generate binary grid
Solve maze using BFS
Convert to robot coordinates
Execute trajectory

📷 Demo

👉 Watch the robot solve the maze
https://www.youtube.com/shorts/gi8C1Oyw_Mk

⚠️ Challenges & Solutions

Color detection tuning (HSV adjustments)
Robot reachability constraints
Calibration accuracy improvements
📄 Report
Full project report available in /report
