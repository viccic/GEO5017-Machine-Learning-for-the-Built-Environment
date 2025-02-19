# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# defining all 3 axis
z = np.array([1.0, 2.38, 2.49, 2.15, 2.59, 4.32])
x = np.array([2.0, 1.08, -0.83, -1.97, -1.31, 0.57])
y = np.array([0.0, 1.68, 1.82, 0.28, -1.51, -1.91])
# c = x + y

time = []
for i in range(1, 7):
    time.append("T:" + str(i))

# plotting
ax.scatter(x, y, z)
ax.plot3D(x, y, z)
ax.set_title('Trajectory')


for i, txt in enumerate(time):
    if i == 0 or i == 4 or i == 5:
        ax.text(x[i], y[i], z[i] - 0.2, txt, ha='center', va='top')
    else:
        ax.text(x[i], y[i], z[i] + 0.2, txt, ha='center', va='bottom')

plt.show()
