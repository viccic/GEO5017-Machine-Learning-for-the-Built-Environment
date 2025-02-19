import numpy as np
from scipy.differentiate import derivative

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

# implementing gradient decent for x-dimension

# with respect to intercept

derivative_sum_of_squares = -2 * ((x[0] - (alpha_x + beta_x * t[0])) + (x[1] - (alpha_x + beta_x * t[1])) + (x[2] - (alpha_x + beta_x * t[2]))
+ (x[3] - (alpha_x + beta_x * t[3])) + (x[4] - (alpha_x + beta_x * t[4])) + (x[5] - (alpha_x + beta_x * t[5])))



# End of 2.2.a