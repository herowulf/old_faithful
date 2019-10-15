import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def gif(data, group_list, mean_1, mean_2):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    scat = plt.scatter(data[:,0], data[:,1], c=group_list[0])
    scat_1 = plt.scatter(mean_1[0][0], mean_1[0][1], c='r')
    scat_2 = plt.scatter(mean_2[0][0], mean_2[0][1], c='b')


    def update(i):
        label = 'timestep {0}'.format(i)
        scat.set_array(group_list[i])
        scat_1.set_offsets(mean_1[i])
        scat_2.set_offsets(mean_2[i])
        ax.set_xlabel(label)
        return scat, scat_1, scat_2, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(group_list)), interval=500)
    # anim.save('mooi.mp4', writer = writer)
    plt.show()
