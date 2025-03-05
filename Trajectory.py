import matplotlib.pyplot as plt

def plot_trajectory(x,y,z):

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #Set limit for axes
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-5, 2)
    # ax.set_zlim(0, 5)

    time = []
    for i in range(1, len(x)+1):
        time.append("T:" + str(i))

    # plotting
    ax.scatter(x, y, z)
    ax.plot3D(x, y, z)
    # ax.set_title('Trajectory')

    # creating annotations for timestamps
    for i, txt in enumerate(time):
        if i == 0 or i == 4 or i == 5:
            ax.text(x[i], y[i], z[i] - 0.2, txt, ha='center', va='top')
        else:
            ax.text(x[i], y[i], z[i] + 0.2, txt, ha='center', va='bottom')

    plt.show()
    return fig


