import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.stats import multivariate_normal


def gif(data, group_list, mean_1, mean_2, cov_1, cov_2):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    x, y = np.mgrid[np.amin(data):np.amax(data):.01, np.amin(data):np.amax(data):.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    rv_1 = multivariate_normal(mean_1[0], cov_1[0])
    rv_2 = multivariate_normal(mean_2[0], cov_2[0])

    cont1 = plt.contour(x, y, rv_1.pdf(pos), colors='r')
    cont2 = plt.contour(x, y, rv_2.pdf(pos), colors='b')
    scat = plt.scatter(data[:,0], data[:,1], c=group_list[0])


    def update(i):
        label = 'timestep {0}'.format(i)
        ax.clear()

        rv_1 = multivariate_normal(mean_1[i], cov_1[i])
        rv_2 = multivariate_normal(mean_2[i], cov_2[i])

        cont1 = ax.contour(x, y, rv_1.pdf(pos), colors='r')
        cont2 = ax.contour(x, y, rv_2.pdf(pos), colors='b')

        scat = plt.scatter(data[:,0], data[:,1], c=group_list[i])

        ax.set_xlabel(label)
        return scat, cont1, cont2, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(group_list)), interval=500)
    anim.save('mooi.mp4', writer = writer)
    plt.show()
