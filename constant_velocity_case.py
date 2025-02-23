import numpy as np

def sum_for_variance(x):

    # computing average value for x values
    x_average = np.average(x)

    sum = 0
    for i in range(len(x)):
        sum += (x[i] - x_average)**2

    return sum

def sum_for_covariance(x,y):

    # computing average value for x and y values
    x_average = np.average(x)
    y_average = np.average(y)

    sum = 0
    for i in range(len(x)):
        sum += (x[i] - x_average)*(y[i] - y_average)

    return sum

def simple_linear_regression(x,y):
    n = len(y)  # Sample size

    # computing sum for variance
    sum_variance = sum_for_variance(x)

    # computing variance
    variance = 1 / (n - 1) * sum_variance

    # computing sum for covariance
    sum_covariance = sum_for_covariance(x,y)

    # computing variance
    covariance = 1 / (n - 1) * sum_covariance

    # computing average value for x and y values
    x_average = np.average(x)
    y_average = np.average(y)

    # calculating beta and alpha
    slope = covariance / variance
    intercept = y_average - slope * x_average

    return slope, intercept