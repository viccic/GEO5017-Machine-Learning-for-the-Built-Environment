import numpy as np
from Trajectory import plot_trajectory
from constant_velocity_case import simple_linear_regression

# Importing data

# defining positions in 3 axis
x = np.array([2.0, 1.08, -0.83, -1.97, -1.31, 0.57])
y = np.array([0.0, 1.68, 1.82, 0.28, -1.51, -1.91])
z = np.array([1.0, 2.38, 2.49, 2.15, 2.59, 4.32])

# defining time array
t = np.array([1,2,3,4,5,6])

# Start of 2.1

# plotting the trajectory
plot_trajectory(x,y,z)

# End of 2.1

# Start of 2.2.a

# implementing simple linear regression for x-dimension
# slope_x, intercept_x = simple_linear_regression(t,x)
# print("slope for x: ", slope_x)
# print("intercept for x: ", intercept_x)
#
# # implementing simple linear regression for y-dimension
# slope_y, intercept_y = simple_linear_regression(t,y)
# print("slope for y: ", slope_y)
# print("intercept for y: ", intercept_y)
#
# # implementing simple linear regression for z-dimension
# slope_z,intercept_z = simple_linear_regression(t,z)
# print("slope for z: ", slope_z)
# print("intercept for z: ", intercept_z)

# Learning rate and number of iterations
learning_rate = 0.01
num_iterations = 100

slope_x, slope_y, slope_z = 1,1,1
intercept_x, intercept_y, intercept_z = 0,0,0

# implementing gradient decent
for it in range(num_iterations):

    # FOR X-DIMENSION

    # Compute gradient for intercept (∂L/∂intercept)
    derivative_sum_of_squares_intercept_x = -2 * np.sum(x - (intercept_x + slope_x * t))

    # Compute gradient for slope (∂L/∂slope)
    derivative_sum_of_squares_slope_x = -2 * np.sum(t * (x - (intercept_x + slope_x * t)))

    # Compute updates
    diff_intercept_x = learning_rate * derivative_sum_of_squares_intercept_x
    diff_slope_x = learning_rate * derivative_sum_of_squares_slope_x

    # FOR Y-DIMENSION

    # Compute gradient for intercept (∂L/∂intercept)
    derivative_sum_of_squares_intercept_y = -2 * np.sum(y - (intercept_y + slope_y * t))

    # Compute gradient for slope (∂L/∂slope)
    derivative_sum_of_squares_slope_y = -2 * np.sum(t * (y - (intercept_y + slope_y * t)))

    # Compute updates
    diff_intercept_y = learning_rate * derivative_sum_of_squares_intercept_y
    diff_slope_y = learning_rate * derivative_sum_of_squares_slope_y

    # FOR Z-DIMENSION

    # Compute gradient for intercept (∂L/∂intercept)
    derivative_sum_of_squares_intercept_z = -2 * np.sum(z - (intercept_z + slope_z * t))

    # Compute gradient for slope (∂L/∂slope)
    derivative_sum_of_squares_slope_z = -2 * np.sum(t * (z - (intercept_z + slope_z * t)))

    # Compute updates
    diff_intercept_z = learning_rate * derivative_sum_of_squares_intercept_z
    diff_slope_z = learning_rate * derivative_sum_of_squares_slope_z

    # Convergence check
    if np.abs(diff_intercept_x) < 0.001 and np.abs(diff_slope_x) < 0.001 and np.abs(diff_intercept_y) < 0.001 and np.abs(diff_slope_y) < 0.001 and np.abs(diff_intercept_z) < 0.001 and np.abs(diff_slope_z) < 0.001:
        print("Converged!")
        break

    # Update parameters
    intercept_x -= diff_intercept_x
    slope_x -= diff_slope_x

    intercept_y -= diff_intercept_y
    slope_y -= diff_slope_y

    intercept_z -= diff_intercept_z
    slope_z -= diff_slope_z

    print("iteration", it)

print("\nFinal values:")
print("Intercept for x:", intercept_x)
print("Slope for x:", slope_x)

print("Intercept for y:", intercept_y)
print("Slope for y:", slope_y)

print("Intercept for z:", intercept_z)
print("Slope for z:", slope_z)

x_new, y_new, z_new = np.zeros(6), np.zeros(6), np.zeros(6)
for i in range(len(t)):
    x_new[i] = intercept_x + slope_x * t[i]
    y_new[i] = intercept_y + slope_y * t[i]
    z_new[i] = intercept_z + slope_z * t[i]

plot_trajectory(x_new,y_new,z_new)

# End of 2.2.a