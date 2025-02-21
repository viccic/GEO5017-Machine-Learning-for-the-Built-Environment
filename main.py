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
beta_x, alpha_x = simple_linear_regression(t,x)
print("beta for x: ", beta_x)
print("alpha for x: ", alpha_x)

# implementing simple linear regression for y-dimension
beta_y, alpha_y = simple_linear_regression(t,y)
print("beta for y: ", beta_y)
print("alpha for y: ", alpha_y)

# implementing simple linear regression for z-dimension
beta_z, alpha_z = simple_linear_regression(t,z)
print("beta for z: ", beta_z)
print("alpha for z: ", alpha_z)

# Learning rate
learning_rate = 0.01

# implementing gradient decent for x-dimension

for it in range(100):

    # with respect to intercept
    derivative_sum_of_squares_intercept = -2 * (
                (x[0] - (alpha_x + beta_x * t[0])) + (x[1] - (alpha_x + beta_x * t[1])) + (
                    x[2] - (alpha_x + beta_x * t[2]))
                + (x[3] - (alpha_x + beta_x * t[3])) + (x[4] - (alpha_x + beta_x * t[4])) + (
                            x[5] - (alpha_x + beta_x * t[5])))

    diff_intercept = learning_rate * derivative_sum_of_squares_intercept
    print('diff_intercept: ', diff_intercept)

    # with respect to slope
    derivative_sum_of_squares_slope = -2 * (
                t[0] * (x[0] - (alpha_x + beta_x * t[0])) + t[1] * (x[1] - (alpha_x + beta_x * t[1])) + t[2] * (
                    x[2] - (alpha_x + beta_x * t[2]))
                + t[3] * (x[3] - (alpha_x + beta_x * t[3])) + t[4] * (x[4] - (alpha_x + beta_x * t[4])) + t[5] * (
                            x[5] - (alpha_x + beta_x * t[5])))
    diff_slope = learning_rate * derivative_sum_of_squares_slope
    print('diff_slope: ', diff_slope)

    if np.abs(diff_intercept) < 0.001 or np.abs(diff_slope) < 0.001:  # Check if the step size is smaller than the tolerance
        break  # If yes, stop the algorithm
    # print("iteration =", it, "\t\tx =", "{:.5f}".format(x), "\t\tf(x) =", "{:.3f}".format(function(x)))
    alpha_x = alpha_x - diff_intercept  # Update the current intercept
    beta_x = beta_x - diff_slope # Update the current
    print("iteration: ", it)

print("final alpha for x: ", alpha_x)
print("final beta for x: ", beta_x)


# End of 2.2.a