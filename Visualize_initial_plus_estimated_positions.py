import matplotlib.pyplot as plt

def visualize(x,y,z, x0, y0, z0):

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plotting
    ax.scatter(x, y, z)
    ax.plot3D(x, y, z)
    ax.set_title('Initial and Estimated Positions')

    # plotting
    ax.scatter(x0, y0, z0)
    ax.plot3D(x0, y0, z0)

    plt.show()

    return fig


