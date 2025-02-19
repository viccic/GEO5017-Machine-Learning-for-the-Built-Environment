import numpy as np
from Trajectory import plot_trajectory
from constant_velocity_case import simple_linear_regression

# defining positions in 3 axis
x = np.array([2.0, 1.08, -0.83, -1.97, -1.31, 0.57])
y = np.array([0.0, 1.68, 1.82, 0.28, -1.51, -1.91])
z = np.array([1.0, 2.38, 2.49, 2.15, 2.59, 4.32])

# defining time array
t = np.array([1,2,3,4,5,6])

# plotting the trajectory
plot_trajectory(x,y,z)

# implementing simple linear regression for x-dimension
beta, alpha = simple_linear_regression(x,t)
print("beta for x: ", beta)
print("alpha for x: ", alpha)

# implementing simple linear regression for y-dimension
beta, alpha = simple_linear_regression(y,t)
print("beta for y: ", beta)
print("alpha for y: ", alpha)

# implementing simple linear regression for z-dimension
beta, alpha = simple_linear_regression(z,t)
print("beta for z: ", beta)
print("alpha for z: ", alpha)

