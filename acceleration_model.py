# Loading data

import numpy as np
from Trajectory import plot_trajectory


# Load data into np array
x_coords = np.array([2.00, 1.08, -0.83, -1.97, -1.31, 0.57])
y_coords = np.array([0.00, 1.68, 1.82, 0.28, -1.51, -1.91])
z_coords = np.array([1.00, 2.38, 2.49, 2.15, 2.59, 4.32])

coordinates = np.vstack([x_coords, y_coords, z_coords]).transpose()
time_record = np.array([1.00, 2.00, 3.00, 4.00, 5.00, 6.00])


# Define error function
def error_function(positions, times, params):
    e = 0
    for p, t in zip(positions, times):
        e += (p - (params[0] + params[1]*t +params[2]*t**2))**2
    return e


#Define gradient function: derivative of error function
def gradient_function(positions, times, params):
    gradient_vector = []
    # Calculate components
    for param in params:
        gradient = 0
        for p, t in zip(positions, times):
            gradient += np.sum(2 * (p - (params[0] + params[1] * t + params[2] * t ** 2)) * t**np.where(params == param)[0][0])
        gradient_vector.append(gradient)
    return np.array(gradient_vector)

#nbcjqwibi
def gradient_descent(positions, times, function, gradient, learn_rate, max_iter, tol=0.001):
    """
    Performs gradient descent to minimize a given function.

    Parameters:
    start (float): The starting point for the algorithm.
    function (callable): The function to minimize.
    gradient (callable): The gradient of the function.
    learn_rate (float): The learning rate (step size).
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for stopping the algorithm.

    Returns:
    float: The point at which the function is minimized.
    """
    params = np.random.randint(1, 10, size=3) # Initialize the starting point
    for it in range(max_iter):
        diff = learn_rate * gradient(positions, times, params)  # Calculate the step size
        if np.any(np.abs(diff) < tol):  # Check if the step size is smaller than the tolerance
            break  # If yes, stop the algorithm
        #print("iteration =", it, "\t\tparams =", "{:.5f}".format(params), "\t\tf(x) =", "{:.3f}".format(function(x)))
        params = params + diff  # Update the current point
    return params

plot_trajectory()
print(gradient_descent(coordinates, time_record, error_function, gradient_function, 0.0001, 100))


