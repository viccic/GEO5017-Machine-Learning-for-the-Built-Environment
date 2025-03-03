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

def train(positions, times):

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
                gradient += 2 * (p - (params[0] + params[1] * t + params[2] * t ** 2)) * t ** i
            gradient_vector.append(gradient)
        return np.array(gradient_vector)

    def gradient_descent(positions, times, learn_rate=0.0001, max_iter=100, tol=0.001):
        # params = np.random.uniform(1.0, 3.0, size=3)
        a0 = np.array([0,1,1])
        a1 = np.array([0,1,1])
        a2 = np.array([0,1,1])
        params = np.array([a0, a1, a2])
        for it in range(max_iter):
            diff = learn_rate * gradient_vector(positions, times, params)
            if np.all(np.abs(diff) < tol):
                break
            params = params + diff
        return params

    params = gradient_descent(positions, times)
    e = objective_function(positions, times, params)

    return params, e

print(train(coordinates, time_record))



trained_model = train(coordinates, time_record)

def predict(model, times):
    params = trained_model[0]
    error = trained_model[1]
    predictions = []
    for t in times:
        predicted_position = params[0] + params[1] * t + params[2] * t ** 2
        predictions.append(predicted_position)

    return predictions

# print(predict(trained_model, time_record))
# Predict positions from second 1 to 7
time_record_1 = np.append(time_record, 7.00)
predictions = predict(trained_model, time_record_1)
sec_7 = predictions[-1]
x_1 = np.append(x_coords, sec_7[0])
y_1 = np.append(y_coords, sec_7[1])
z_1 = np.append(z_coords, sec_7[2])

#Plot predictions and positions
x = []
y = []
z = []
for prediction in predictions:
    x.append(prediction[0])
    y.append(prediction[1])
    z.append(prediction[2])


fig_1 = plot_trajectory(x_1, y_1, z_1)
fig_1.savefig('output/second_7.png')
fig_1.suptitle("Predicted postions")
fig_2 = plot_trajectory(x, y, z)
fig_2.savefig('output/predicted.png')
fig_2.suptitle("Predicted position at t=7")