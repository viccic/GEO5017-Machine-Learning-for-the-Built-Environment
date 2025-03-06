import numpy as np
from trajectory import plot_trajectory
from constant_velocity_case import constant_velocity_def
from constant_acceleration_case import constant_acceleration_def
from visualize_initial_plus_estimated_positions import visualize

# Importing data

# defining positions in 3 axis
x = np.array([2.0, 1.08, -0.83, -1.97, -1.31, 0.57])
y = np.array([0.0, 1.68, 1.82, 0.28, -1.51, -1.91])
z = np.array([1.0, 2.38, 2.49, 2.15, 2.59, 4.32])

# defining time array
t = np.array([1,2,3,4,5,6])

# Start of 2.1

# plotting the trajectory
fig_1 = plot_trajectory(x,y,z)
fig_1.savefig('./output/Trajectory_plot.png')

# End of 2.1

# Start of 2.2.a

# Set learning rate and number of iterations
learning_rate = 0.01
num_iterations = 300
tolerance = 0.0001
constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, SSE_x, SSE_y, SSE_z = constant_velocity_def(x, y, z, t, learning_rate, num_iterations,tolerance)

print("\nFinal values for constant speed case:")
print(f"Parameters for x --> α: {constant_coefficient_x:.3f}, β: {velocity_x:.3f}")
print(f"Parameters for y --> α: {constant_coefficient_y:.3f}, β: {velocity_y:.3f}")
print(f"Parameters for z --> α: {constant_coefficient_z:.3f}, β: {velocity_z:.3f}")
print(f"SSE_x : {SSE_x:.3f}")
print(f"SSE_y : {SSE_y:.3f}")
print(f"SSE_z : {SSE_z:.3f}")

# Plot the initial and the estimated model for constant velocity
x_new, y_new, z_new = np.zeros(6), np.zeros(6), np.zeros(6)

for i in range(6):
    x_new[i] = constant_coefficient_x + velocity_x * t[i]
    y_new[i] = constant_coefficient_y + velocity_y * t[i]
    z_new[i] = constant_coefficient_z + velocity_z * t[i]

fig_2 = visualize(x,y,z,x_new,y_new,z_new)
fig_2.savefig('./output/Initial_trajectory_and_Constant_velocity_model.png')

# End of 2.2.a

# Start of 2.2.b

# Set learning rate and number of iterations
learning_rate = 0.0004
num_iterations = 22000
tolerance = 0.0001
constant_coefficient_x, velocity_x, acceleration_x, constant_coefficient_y, velocity_y, acceleration_y, constant_coefficient_z, velocity_z, acceleration_z, SSE_x, SSE_y, SSE_z = constant_acceleration_def(x, y, z, t, learning_rate, num_iterations, tolerance)

print("\nFinal values for constant acceleration case:")
print(f"Parameters for x --> α0: {constant_coefficient_x:.3f}, α1: {velocity_x:.3f}, α2: {acceleration_x:.3f}")
print(f"Parameters for y --> α0: {constant_coefficient_y:.3f}, α1: {velocity_y:.3f}, α2: {acceleration_y:.3f}")
print(f"Parameters for z --> α0: {constant_coefficient_z:.3f}, α1: {velocity_z:.3f}, α2: {acceleration_z:.3f}")
print(f"SSE_x : {SSE_x:.3f}")
print(f"SSE_y : {SSE_y:.3f}")
print(f"SSE_z : {SSE_z:.3f}")

# Plot the initial and the estimated model for constant acceleration
x_new, y_new, z_new = np.zeros(6), np.zeros(6), np.zeros(6)

for i in range(6):
    x_new[i] = constant_coefficient_x + velocity_x * t[i] + acceleration_x * t[i] ** 2
    y_new[i] = constant_coefficient_y + velocity_y * t[i] + acceleration_y * t[i] ** 2
    z_new[i] = constant_coefficient_z + velocity_z * t[i] + acceleration_z * t[i] ** 2

fig_3 = visualize(x,y,z,x_new,y_new,z_new)
fig_3.savefig('./output/Initial_trajectory_and_Constant_acceleration_model.png')

# End of 2.2.b

# Start of 2.2.c

x_7 = constant_coefficient_x + velocity_x * 7 + acceleration_x * 7 ** 2
y_7 = constant_coefficient_y + velocity_y * 7 + acceleration_y * 7 ** 2
z_7 = constant_coefficient_z + velocity_z * 7 + acceleration_z * 7 ** 2

print("\nPredicted position for t = 7 :")
print(f"x = {x_7:.3f}")
print(f"y = {y_7:.3f}")
print(f"z = {z_7:.3f}")

x = np.append(x,x_7)
y = np.append(y,y_7)
z = np.append(z,z_7)

fig_4 = plot_trajectory(x,y,z)
fig_4.savefig('./output/Predicted_position.png')

# End of 2.2.c

