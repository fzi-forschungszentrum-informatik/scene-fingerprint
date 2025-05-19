import numpy as np
import math


def generate_straight_path(x_start, y_start, psi_rad, vx, vy, resolution=2.0):
    """
    Generate a straight line in 2D coordinates with homogeneous spacing.

    Parameters:
    x_start (float): Starting x-coordinate.
    y_start (float): Starting y-coordinate.
    psi_rad (float): Angle of the line in radians.
    velocity (float): Defines the length of the line.
    resolution (float): Maximum allowed distance between points.

    Returns:
    list: A list of (x, y) tuples representing the points on the line.
    """

    # Path length corresponds to 10s waytime
    length = np.sqrt(vx**2 + vy**2) * 10.0
    num_points = max(2, int(length / resolution) + 1)
    if math.isnan(psi_rad):
        angle = np.arctan2(vy, vx)
    else:
        angle = psi_rad

    x_points = np.linspace(0, length, num_points) * np.cos(angle) + x_start
    y_points = np.linspace(0, length, num_points) * np.sin(angle) + y_start

    return (x_points.tolist(), y_points.tolist())
