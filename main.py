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
num_iterations = 200
constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, SSE_x, SSE_y, SSE_z = constant_velocity_def(x, y, z, t, learning_rate, num_iterations)

print("\nFinal values for constant speed case:")
print(f"Parameters for x --> α: {constant_coefficient_x:.3f}, β: {velocity_x:.3f}")
print(f"Parameters for y --> α: {constant_coefficient_y:.3f}, β: {velocity_y:.3f}")
print(f"Parameters for z --> α: {constant_coefficient_z:.3f}, β: {velocity_z:.3f}")
print(f"SSE_x : {SSE_x:.3f}")
print(f"SSE_y : {SSE_y:.3f}")
print(f"SSE_z : {SSE_z:.3f}")

# End of 2.2.a

# Start of 2.2.b

# Set learning rate and number of iterations
learning_rate = 0.0001
num_iterations = 10000
constant_coefficient_x, velocity_x, accelaration_x, constant_coefficient_y, velocity_y, accelaration_y, constant_coefficient_z, velocity_z, accelaration_z, SSE_x, SSE_y, SSE_z = constant_acceleration_def(x, y, z, t, learning_rate, num_iterations)

print("\nFinal values for constant acceleration case:")
print(f"Parameters for x --> α0: {constant_coefficient_x:.3f}, α1: {velocity_x:.3f}, α2: {accelaration_x:.3f}")
print(f"Parameters for y --> α0: {constant_coefficient_y:.3f}, α1: {velocity_y:.3f}, α2: {accelaration_x:.3f}")
print(f"Parameters for z --> α0: {constant_coefficient_z:.3f}, α1: {velocity_z:.3f}, α2: {accelaration_x:.3f}")
print(f"SSE_x : {SSE_x:.3f}")
print(f"SSE_y : {SSE_y:.3f}")
print(f"SSE_z : {SSE_z:.3f}")

# End of 2.2.b

# Start of 2.2.c

x_7 = constant_coefficient_x + velocity_x * 7 + accelaration_x * 7 ** 2
y_7 = constant_coefficient_y + velocity_y * 7 + accelaration_y * 7 ** 2
z_7 = constant_coefficient_z + velocity_z * 7 + accelaration_z * 7 ** 2

x = np.append(x, x_7)
y = np.append(y, y_7)
z = np.append(z, z_7)

fig_2 = plot_trajectory(x,y,z)
fig_1.savefig('./output/Predicted_position.png')

