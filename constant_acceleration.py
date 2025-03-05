# Loading data

import numpy as np
from Trajectory import plot_trajectory
from Visualize_initial_plus_estimated_positions import visualize


# Load data into np array
x_coords = np.array([2.00, 1.08, -0.83, -1.97, -1.31, 0.57])
y_coords = np.array([0.00, 1.68, 1.82, 0.28, -1.51, -1.91])
z_coords = np.array([1.00, 2.38, 2.49, 2.15, 2.59, 4.32])

coordinates = np.vstack([x_coords, y_coords, z_coords]).transpose()
time_record = np.array([1.00, 2.00, 3.00, 4.00, 5.00, 6.00])

fig_0 = plot_trajectory(x_coords, y_coords, z_coords)
fig_0.savefig("output/trajectory.png")

class acceleration_model:

    def __init__(self):
        pass

    def train(self, positions, times, learn_rate, max_iter, tol):
        def objective_function(positions, times, params):
            error = 0
            for p, t in zip(positions, times):
                error += (p - (params[0] + params[1] * t + params[2] * t ** 2)) ** 2
            return error

        def gradient_vector(positions, times, params):
            gradient_vector = []
            # Calculate components
            for i, param in enumerate(params):
                gradient = 0
                for p, t in zip(positions, times):
                    gradient -= 2 * (p - (params[0] + params[1] * t + params[2] * t ** 2)) * t ** i
                gradient_vector.append(gradient)
            return np.array(gradient_vector)

        def gradient_descent(positions, times, learn_rate, max_iter, tol):
            params = np.array([0,1,1])
            for it in range(max_iter):
                diff = learn_rate * gradient_vector(positions, times, params)
                if np.all(np.abs(diff) < tol):
                    break
                params = params - diff
            return params

        params = gradient_descent(positions, times, learn_rate, max_iter, tol)
        error = objective_function(positions, times, params)
        return params, error

    def predict(self, model, times):
        params = model[0]
        if isinstance(times,list) or isinstance(times, np.ndarray):
            predictions = []
            for t in times:
                predicted_position = params[0] + params[1] * t + params[2] * t ** 2
                predictions.append(predicted_position)
            return predictions
        elif type(times) == int:
            predicted_position = params[0] + params[1] * times + params[2] * times ** 2
            return predicted_position


x_axis = acceleration_model()
trained_model_x = x_axis.train(x_coords, time_record, learn_rate=0.0001, max_iter=100000, tol=0.0001)
print(trained_model_x)
y_axis = acceleration_model()
trained_model_y = y_axis.train(y_coords, time_record, learn_rate=0.0001, max_iter=10000, tol=0.0001)
print(trained_model_y)
z_axis = acceleration_model()
trained_model_z = z_axis.train(z_coords, time_record, learn_rate=0.0001, max_iter=10000, tol=0.0001)
print(trained_model_z)

prediction_x = x_axis.predict(trained_model_x, time_record)
prediction_y = y_axis.predict(trained_model_y, time_record)
prediction_z = z_axis.predict(trained_model_z, time_record)
plot_trajectory(prediction_x, prediction_y, prediction_z)
