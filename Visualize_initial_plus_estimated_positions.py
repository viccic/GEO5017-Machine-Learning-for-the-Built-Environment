import matplotlib.pyplot as plt

def visualize(x,y,z, x0, y0, z0):

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    time = []
    for i in range(1, 7):
        time.append("T:" + str(i))

    # plotting
    ax.scatter(x, y, z)
    ax.plot3D(x, y, z)
    ax.set_title('Trajectory')

    # plotting
    ax.scatter(x0, y0, z0)
    ax.plot3D(x0, y0, z0)

    # creating annotations for timestamps
    for i, txt in enumerate(time):
        if i == 0 or i == 4 or i == 5:
            ax.text(x[i], y[i], z[i] - 0.2, txt, ha='center', va='top')
        else:
            ax.text(x[i], y[i], z[i] + 0.2, txt, ha='center', va='bottom')

    plt.show()


