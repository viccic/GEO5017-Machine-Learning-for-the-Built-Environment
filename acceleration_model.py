# Loading data

import numpy as np


# Load data into np array
x_coords = np.array([2.00, 1.08, -0.83, -1.97, -1.31, 0.57])
y_coords = np.array([0.00, 1.68, 1.82, 0.28, -1.51, -1.91])
z_coords = np.array([1.00, 2.38, 2.49, 2.15, 2.59, 4.32])

positions = np.vstack([x_coords, y_coords, z_coords]).transpose()
times = np.array([1.00, 2.00, 3.00, 4.00, 5.00, 6.00])

# Define error function
def error_function(positions, times, a0, a1, a2):
    e = 0
    for p, t in zip(positions, times):
        e += (p - (a0 + a1*t +a2*t**2))**2
    return e


def gradient_function(positions, times, a0, a1, a2):

