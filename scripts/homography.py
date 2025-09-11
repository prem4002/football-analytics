import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define source and target points ---
source_points = np.array([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
], dtype=np.float32)

target_points = np.array([
    [10, 0],
    [90, 20],
    [80, 100],
    [20, 80]
], dtype=np.float32)

# --- Step 2: Compute homography ---
H, _ = cv2.findHomography(source_points, target_points)

# --- Step 3: Define some test points inside the square ---
points = np.array([[50, 50], [75, 25]], dtype=np.float32).reshape(-1, 1, 2)
transformed_points = cv2.perspectiveTransform(points, H)

# --- Step 4: Plot source vs target shapes ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Left: Source shape (square + test points)
axs[0].scatter(source_points[:, 0], source_points[:, 1], c="red", label="Source corners")
axs[0].scatter(points[:, 0, 0], points[:, 0, 1], c="green", marker="x", label="Test points")
axs[0].plot(np.append(source_points[:,0], source_points[0,0]),
            np.append(source_points[:,1], source_points[0,1]), 'r--')  # connect square edges
axs[0].set_title("Source (Square)")
axs[0].invert_yaxis()  # Flip y to match image coordinates
axs[0].legend()
axs[0].set_aspect('equal')

# Right: Target shape (trapezoid + transformed points)
axs[1].scatter(target_points[:, 0], target_points[:, 1], c="blue", label="Target corners")
axs[1].scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1],
               c="yellow", marker="x", label="Transformed test points")
axs[1].plot(np.append(target_points[:,0], target_points[0,0]),
            np.append(target_points[:,1], target_points[0,1]), 'b--')  # connect trapezoid edges
axs[1].set_title("Target (Trapezoid)")
axs[1].invert_yaxis()
axs[1].legend()
axs[1].set_aspect('equal')

plt.show()