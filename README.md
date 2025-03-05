# GEO5017-Machine-Learning-for-the-Built-Environment

## Purpose of the code

In this project two ways to estimate the path of a drone from 3D input points are implemented.
The first on is a simple regression specified in constant_velocity.py, the second is a polynomial regression with second order polynomials in acceleration_model.py.

Both methods make use of a gradient descent to iteratively approach the optimal solution. This is specified in gradient_descent.py. 

## Output

* The parameters of the linear regression and polynomial regression
* The estimated next location
* Graphs showing the path of the drone compared to the estimates

## Setup Instructions
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running the Code
In main.py:
* Specify 3D input points
* Specify Learning Rate 
* Specify Iterations

When the output results in very high values, the learning rate might be too large