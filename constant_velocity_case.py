import numpy as np

def constant_velocity_def(x,y,z,t,learning_rate,num_iterations,tolerance):

    # Initialize velocities and constants
    velocity_x, velocity_y, velocity_z = (x[-1] - x[0]) / (t[-1] - t[0]), (y[-1] - y[0]) / (t[-1] - t[0]), (z[-1] - z[0]) / (t[-1] - t[0])
    constant_coefficient_x, constant_coefficient_y, constant_coefficient_z = np.mean(x), np.mean(y), np.mean(z)

    for it in range(num_iterations):
        # FOR X-DIMENSION

        # Compute gradient for constant coefficient (∂SSE_x/∂constant_x)
        grad_sum_of_squares_constant_coefficient_x = -2 * np.sum(x - (constant_coefficient_x + velocity_x * t))

        # Compute gradient for velocity (∂SSE_x/∂velocity_x)
        grad_sum_of_squares_velocity_x = -2 * np.sum(t * (x - (constant_coefficient_x + velocity_x * t)))

        # Compute updates
        diff_constant_coefficient_x = learning_rate * grad_sum_of_squares_constant_coefficient_x
        diff_velocity_x = learning_rate * grad_sum_of_squares_velocity_x


        # FOR Y-DIMENSION

        # Compute gradient for constant coefficient (∂SSE_y/∂constant_y)
        grad_sum_of_squares_constant_coefficient_y = -2 * np.sum(y - (constant_coefficient_y + velocity_y * t))

        # Compute gradient for velocity (∂SSE_y/∂velocity_y)
        grad_sum_of_squares_velocity_y = -2 * np.sum(t * (y - (constant_coefficient_y + velocity_y * t)))

        # Compute updates
        diff_constant_coefficient_y = learning_rate * grad_sum_of_squares_constant_coefficient_y
        diff_velocity_y = learning_rate * grad_sum_of_squares_velocity_y

        # FOR Z-DIMENSION

        # Compute gradient for constant coefficient (∂SSE_z/∂constant_z)
        grad_sum_of_squares_constant_coefficient_z = -2 * np.sum(z - (constant_coefficient_z + velocity_z * t))

        # Compute gradient for velocity (∂SSE_z/∂velocity_z)
        grad_sum_of_squares_velocity_z = -2 * np.sum(t * (z - (constant_coefficient_z + velocity_z * t)))

        # Compute updates
        diff_constant_coefficient_z = learning_rate * grad_sum_of_squares_constant_coefficient_z
        diff_velocity_z = learning_rate * grad_sum_of_squares_velocity_z

        # Convergence check
        if np.abs(diff_constant_coefficient_x) < tolerance and np.abs(diff_velocity_x) < tolerance and np.abs(
                diff_constant_coefficient_y) < tolerance and np.abs(diff_velocity_y) < tolerance and np.abs(
            diff_constant_coefficient_z) < tolerance and np.abs(diff_velocity_z) < tolerance:
            break

        # Update parameters
        constant_coefficient_x -= diff_constant_coefficient_x
        velocity_x -= diff_velocity_x

        constant_coefficient_y -= diff_constant_coefficient_y
        velocity_y -= diff_velocity_y

        constant_coefficient_z -= diff_constant_coefficient_z
        velocity_z -= diff_velocity_z

        # Calculate sum of squared errors
        SSE_x = np.sum((x - (constant_coefficient_x + velocity_x * t)) ** 2)

        SSE_y = np.sum((y - (constant_coefficient_y + velocity_y * t)) ** 2)

        SSE_z = np.sum((z - (constant_coefficient_z + velocity_z * t)) ** 2)

    return constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z,  SSE_x, SSE_y, SSE_z