import numpy as np
import matplotlib.pyplot as plt

import plot_gif

from scipy.stats import multivariate_normal


data = np.genfromtxt('data.txt', usecols=(1,2))

data = (data - np.mean(data, axis=0))/np.std(data, axis=0)


#
# plt.scatter(data[:,0], data[:,1])
# plt.show()

group = []
mean_1 = []
mean_2 = []

group_1 = 0
group_2 = 1



'Step 1'
# mean_1.append([np.amin(data[:,0]), np.amax(data[:,1])])
# mean_2.append([np.amax(data[:,0]), np.amin(data[:,1])])

mean_1.append([-1,0.8])
mean_2.append([1.6, -1.2])

# mean_1.append(np.random.normal(size=2))
# mean_2.append(np.random.normal(size=2))

cov_1 = np.array([[1,0],[0,1]])
cov_2 = np.array([[1,0],[0,1]])

prob_1 = multivariate_normal.pdf(data, mean_1[-1], cov_1)
prob_2 = multivariate_normal.pdf(data, mean_2[-1], cov_2)

group.append(np.where(prob_1 > prob_2, group_1, group_2))

'Step 2'
mean_1.append(np.mean(data[group[-1]==group_1], axis=0))
cov_1 = np.cov(data[group[-1]==group_1], rowvar=0)

mean_2.append(np.mean(data[group[-1]==group_2], axis=0))
cov_2 = np.cov(data[group[-1]==group_2], rowvar=0)

prob_1 = multivariate_normal.pdf(data, mean_1[-1], cov_1)
prob_2 = multivariate_normal.pdf(data, mean_2[-1], cov_2)

group.append(np.where(prob_1 > prob_2, group_1, group_2))


'Steps till groups no longer change'
while np.any(group[-1] != group[-2]):
    mean_1.append(np.mean(data[group[-1]==group_1], axis=0))
    cov_1 = np.cov(data[group[-1]==group_1], rowvar=0)

    mean_2.append(np.mean(data[group[-1]==group_2], axis=0))
    cov_2 = np.cov(data[group[-1]==group_2], rowvar=0)

    prob_1 = multivariate_normal.pdf(data, mean_1[-1], cov_1)
    prob_2 = multivariate_normal.pdf(data, mean_2[-1], cov_2)

    group.append(np.where(prob_1 > prob_2, group_1, group_2))

plot_gif.gif(data, group, mean_1, mean_2)
