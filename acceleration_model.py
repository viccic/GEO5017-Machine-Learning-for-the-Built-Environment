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
            params = np.array([0,0,0])
            for it in range(max_iter):
                diff = learn_rate * gradient_vector(positions, times, params)
                if np.all(np.abs(diff) < tol):
                    break
            params = params - diff
            error = objective_function(positions, times, params)
            print(error)
            return params

        params = gradient_descent(self)
        error = objective_function(self, params)
        return params, error

    def predict(self, model, times):
        params = model[0]
        if type(times) == list:
            predictions = []
            for t in times:
                predicted_position = params[0] + params[1] * t + params[2] * t ** 2
                predictions.append(predicted_position)
            return predictions
        elif type(times) == int:
            predicted_position = params[0] + params[1] * times + params[2] * times ** 2
            return predicted_position

# print(predict(trained_model, time_record))
# Predict positions from second 1 to 7
# time_record_1 = np.append(time_record, 7.00)
# predictions = predict(trained_model, time_record_1)
# sec_7 = predictions[-1]
# x_1 = np.append(x_coords, sec_7[0])
# y_1 = np.append(y_coords, sec_7[1])
# z_1 = np.append(z_coords, sec_7[2])
#
# #Plot predictions and positions
# x = []
# y = []
# z = []
# for prediction in predictions:
#     x.append(prediction[0])
#     y.append(prediction[1])
#     z.append(prediction[2])
#
#
# fig_1 = plot_trajectory(x_1, y_1, z_1)
# fig_1.savefig('output/second_7.png')
# fig_1.suptitle("Predicted postions")
# fig_2 = plot_trajectory(x, y, z)
# fig_2.savefig('output/predicted.png')
# fig_2.suptitle("Predicted position at t=7")