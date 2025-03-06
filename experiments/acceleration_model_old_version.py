# Loading data

import numpy as np
from Trajectory import plot_trajectory


# Load data into np array
x_coords = np.array([2.00, 1.08, -0.83, -1.97, -1.31, 0.57])
y_coords = np.array([0.00, 1.68, 1.82, 0.28, -1.51, -1.91])
z_coords = np.array([1.00, 2.38, 2.49, 2.15, 2.59, 4.32])

coordinates = np.vstack([x_coords, y_coords, z_coords]).transpose()
time_record = np.array([1.00, 2.00, 3.00, 4.00, 5.00, 6.00])



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
        params = np.array([0,1,1])
        for it in range(max_iter):
            diff = learn_rate * gradient_vector(positions, times, params)
            if np.all(np.abs(diff) < tol):
                break
            params = params + diff
        return params

    params = gradient_descent(positions, times)
    e = objective_function(coordinates, time_record, params)

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

print(predict(trained_model, time_record))
predictions = predict(trained_model, time_record)


#Print predicitions
x = []
y = []
z = []
for prediction in predictions:
    x.append(prediction[0])
    y.append(prediction[1])
    z.append(prediction[2])



#
# # params = train()
# #
# # def predict(params):
# # def objective_function(positions, times):
#
#
# # Define error function
# def error_function(positions, times, params):
#     e = 0
#     for p, t in zip(positions, times):
#         e += (p - (params[0] + params[1]*t +params[2]*t**2))**2
#     return e
#
#
# #Define gradient function: derivative of error function
# def gradient_function(positions, times, params):
#     gradient_vector = []
#     # Calculate components
#     for i, param in enumerate(params):
#         gradient = 0
#         for p, t in zip(positions, times):
#             gradient += 2 * (p - (params[0] + params[1] * t + params[2] * t ** 2)) * t ** i
#         gradient_vector.append(gradient)
#     return np.array(gradient_vector)
#
#
# def gradient_descent(positions, times, function, gradient, learn_rate, max_iter, tol=0.001):
#     """
#     Performs gradient descent to minimize a given function.
#
#     Parameters:
#     start (float): The starting point for the algorithm.
#     function (callable): The function to minimize.
#     gradient (callable): The gradient of the function.
#     learn_rate (float): The learning rate (step size).
#     max_iter (int): The maximum number of iterations.
#     tol (float): The tolerance for stopping the algorithm.
#
#     Returns:
#     float: The point at which the function is minimized.
#     """
#     params = np.random.randint(1, 10, size=3) # Initialize the starting point
#     for it in range(max_iter):
#         diff = learn_rate * gradient(positions, times, params)  # Calculate the step size
#         if np.any(np.abs(diff) < tol):  # Check if the step size is smaller than the tolerance
#             break  # If yes, stop the algorithm
#         #print("iteration =", it, "\t\tparams =", "{:.5f}".format(params), "\t\tf(x) =", "{:.3f}".format(function(x)))
#         params = params + diff  # Update the current point
#     return params
#
# plot_trajectory(x_coords, y_coords, z_coords)
# print(gradient_descent(coordinates, time_record, error_function, gradient_function, 0.0001, 100))


