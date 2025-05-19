import numpy as np
import matplotlib.pyplot as plt


def draw_triangle(x, y, yaw, width=1.8, length=5, color="orange"):
    triangle = np.array([
        [length/2, 0],
        [-length/2, width/2],
        [-length/2, -width/2]
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_triangle = (R @ triangle.T).T + np.array([x, y])
    plt.fill(rotated_triangle[:, 0], rotated_triangle[:,
             1], color, alpha=0.7, linewidth=0.5, edgecolor='white')


def draw_circle(x, y, radius=4, color="blue"):
    theta = np.linspace(0, 2*np.pi, 15)
    x_plt = radius * np.cos(theta) + x
    y_plt = radius * np.sin(theta) + y
    plt.plot(x_plt, y_plt, color=color, linewidth=0.7,)
