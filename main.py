import numpy as np
from Trajectory import plot_trajectory
from constant_velocity import constant_velocity_def
from Visualize_initial_plus_estimated_positions import visualize
from constant_acceleration import acceleration_model

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

# Set learning rate and number of iterations
learning_rate = 0.01
num_iterations = 200
constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, SSE_x, SSE_y, SSE_z = constant_velocity_def(x, y, z, t, learning_rate, num_iterations)

print("\nFinal values for constant velocity case:")
print(f"Constant coefficient for x: {constant_coefficient_x:.3f}")
print(f"Velocity for x: {velocity_x:.3f}")

print(f"Constant coefficient for y: {constant_coefficient_y:.3f}")
print(f"Velocity for y: {velocity_y:.3f}")

print(f"Constant coefficient for z: {constant_coefficient_z:.3f}")
print(f"Velocity for z: {velocity_z:.3f}")

print(f"SSE_x : {SSE_x:.3f}")
print(f"SSE_y : {SSE_y:.3f}")
print(f"SSE_z : {SSE_z:.3f}")

x_new, y_new, z_new = np.zeros(6), np.zeros(6), np.zeros(6)
for i in range(len(t)):
    x_new[i] = constant_coefficient_x + velocity_x * t[i]
    y_new[i] = constant_coefficient_y + velocity_y * t[i]
    z_new[i] = constant_coefficient_z + velocity_z * t[i]

visualize(x_new,y_new,z_new,x,y,z)

# End of 2.2.a

# Start of 2.2.b
x_axis = acceleration_model()
trained_model_x = x_axis.train(x, t, learn_rate=0.0002, max_iter=100, tol=0.01)
y_axis = acceleration_model()
trained_model_y = y_axis.train(y, t, learn_rate=0.0002, max_iter=100, tol=0.01)
z_axis = acceleration_model()
trained_model_z = z_axis.train(z, t, learn_rate=0.0002, max_iter=100, tol=0.01)

print("\nFinal values for constant acceleration case:")
print(f"Parameters for x --> a0: {trained_model_x[0][0]:.3f}, a1: {trained_model_x[0][1]:.3f}, a2:{trained_model_x[0][2]:.3f}")
print(f"Parameters for y --> a0: {trained_model_y[0][0]:.3f}, a1: {trained_model_y[0][1]:.3f}, a2:{trained_model_y[0][2]:.3f}")
print(f"Parameters for z --> a0: {trained_model_z[0][0]:.3f}, a1: {trained_model_z[0][1]:.3f}, a2:{trained_model_z[0][2]:.3f}")
print(f"SSE_x : {trained_model_x[1]:.3f}")
print(f"SSE_y : {trained_model_y[1]:.3f}")
print(f"SSE_z : {trained_model_z[1]:.3f}")

# End of 2.2.b

# Start of 2.2.c
prediction_x = x_axis.predict(trained_model_x, 7)
prediction_y = y_axis.predict(trained_model_y, 7)
prediction_z = z_axis.predict(trained_model_z, 7)
predicted_position = np.array([prediction_x, prediction_y, prediction_z])
x_7 = np.append(x, prediction_x)
y_7 = np.append(y, prediction_y)
z_7 = np.append(z, prediction_z)
plot_trajectory(x_7,y_7,z_7)

# End of 2.2.c