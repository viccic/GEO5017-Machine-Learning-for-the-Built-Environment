import numpy as np
from Trajectory import plot_trajectory
from Constant_velocity import constant_velocity_def
from Visualize_initial_plus_estimated_positions import visualize

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

constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, total_velocity, SSE_x, SSE_y, SSE_z, total_SSE = constant_velocity_def(x,y,z,t)

print("\nFinal values:")
print(f"Constant coefficient for x: {constant_coefficient_x:.4}")
print(f"Velocity for x: {velocity_x:.4}")

print(f"Constant coefficient for y: {constant_coefficient_y:.4}")
print(f"Velocity for y: {velocity_y:.4}")

print(f"Constant coefficient for z: {constant_coefficient_z:.4}")
print(f"Velocity for z: {velocity_z:.4}")

print(f"Total Velocity: {total_velocity:.4}")

print(f"SSE_x : {SSE_x:.4}")
print(f"SSE_y : {SSE_y:.4}")
print(f"SSE_z : {SSE_z:.4}")
print(f"Residual Error (SSE): {total_SSE:.4}")

x_new, y_new, z_new = np.zeros(6), np.zeros(6), np.zeros(6)
for i in range(len(t)):
    x_new[i] = constant_coefficient_x + velocity_x * t[i]
    y_new[i] = constant_coefficient_y + velocity_y * t[i]
    z_new[i] = constant_coefficient_z + velocity_z * t[i]

visualize(x_new,y_new,z_new,x,y,z)

# End of 2.2.a