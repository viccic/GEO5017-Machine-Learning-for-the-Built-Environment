import numpy as np
import matplotlib.pyplot as plt

def constant_velocity_def(x,y,z,t):

    # Dictionary to store results and coefficients
    learning_rate_plus_iteration_dict = {}
    con_coeff_plus_velocity_x_dict = {}
    con_coeff_plus_velocity_y_dict = {}
    con_coeff_plus_velocity_z_dict = {}
    SSE_x_dict = {}
    SSE_y_dict = {}
    SSE_z_dict = {}
    total_SSE_dict = {}
    indexing = 0

    for learning_rate in np.arange(0.001,0.011,0.001):
        for num_iterations in range(100,1001,100):

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
                if np.abs(diff_constant_coefficient_x) < 0.0001 and np.abs(diff_velocity_x) < 0.0001 and np.abs(
                        diff_constant_coefficient_y) < 0.0001 and np.abs(diff_velocity_y) < 0.0001 and np.abs(
                    diff_constant_coefficient_z) < 0.0001 and np.abs(diff_velocity_z) < 0.0001:
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

            total_SSE = SSE_x + SSE_y + SSE_z

            # Store learning rate, number of iterations, velocity, con. coefficient and SSE per dimension and total SSE for each solver in dictionaries
            learning_rate_plus_iteration_dict[indexing]= [learning_rate, num_iterations]
            con_coeff_plus_velocity_x_dict[indexing]= [constant_coefficient_x,velocity_x]
            con_coeff_plus_velocity_y_dict[indexing] = [constant_coefficient_y,velocity_y]
            con_coeff_plus_velocity_z_dict[indexing] = [constant_coefficient_z,velocity_z]
            SSE_x_dict[indexing] = SSE_x
            SSE_y_dict[indexing] = SSE_y
            SSE_z_dict[indexing] = SSE_z
            total_SSE_dict[indexing] = total_SSE

            indexing += 1

    min_total_SSE = np.min(list(total_SSE_dict.values()))
    min_total_SSE_key = min(total_SSE_dict, key=total_SSE_dict.get)
    print(f"Best learning rate: {learning_rate_plus_iteration_dict[min_total_SSE_key][0]:.2f}")
    print("Best number of iteration: ", learning_rate_plus_iteration_dict[min_total_SSE_key][1])

    constant_coefficient_x, velocity_x = con_coeff_plus_velocity_x_dict[min_total_SSE_key]
    constant_coefficient_y, velocity_y = con_coeff_plus_velocity_y_dict[min_total_SSE_key]
    constant_coefficient_z, velocity_z = con_coeff_plus_velocity_z_dict[min_total_SSE_key]
    total_velocity = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
    total_SSE = min_total_SSE
    SSE_x = SSE_x_dict[min_total_SSE_key]
    SSE_y = SSE_y_dict[min_total_SSE_key]
    SSE_z = SSE_z_dict[min_total_SSE_key]

    x = np.array(list(total_SSE_dict.keys()))
    y = np.array(list(total_SSE_dict.values()))

    plt.scatter(x, y)
    # Highlight solver with the mininimum SSE
    highlight_index = min_total_SSE_key
    plt.scatter(x[highlight_index], y[highlight_index], color='red', label='Best Solver')

    # Optional: add annotation
    plt.annotate(f'Index {highlight_index}',
                 (x[highlight_index], y[highlight_index]),
                 textcoords="offset points",
                 xytext=(10, 10),
                 ha='center')

    plt.xlabel('Solver Index')
    plt.ylabel('Total SSE')
    plt.title('Total SSE per Solver')
    plt.legend()
    plt.show()

    return constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, total_velocity, SSE_x, SSE_y, SSE_z, total_SSE


# Importing data

# defining positions in 3 axis
x = np.array([2.0, 1.08, -0.83, -1.97, -1.31, 0.57])
y = np.array([0.0, 1.68, 1.82, 0.28, -1.51, -1.91])
z = np.array([1.0, 2.38, 2.49, 2.15, 2.59, 4.32])

# defining time array
t = np.array([1,2,3,4,5,6])

constant_coefficient_x, velocity_x, constant_coefficient_y, velocity_y, constant_coefficient_z, velocity_z, total_velocity, SSE_x, SSE_y, SSE_z, total_SSE = constant_velocity_def(x,y,z,t)


