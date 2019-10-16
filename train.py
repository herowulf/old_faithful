import numpy as np
import matplotlib.pyplot as plt

import plot_gif

from scipy.stats import multivariate_normal

# Retrieve data
data = np.genfromtxt('data.txt', usecols=(1,2))

# Normalise data
data = (data - np.mean(data, axis=0))/np.std(data, axis=0)

# Make lists to record progress
group = []
mean_1 = []
mean_2 = []

group_1 = 0
group_2 = 1

'Step 1'

# Start points 1
# mean_1.append([np.amin(data[:,0]), np.amax(data[:,1])])
# mean_2.append([np.amax(data[:,0]), np.amin(data[:,1])])

#Start points chosen for nice number of iterations
mean_1.append([-1,0.8])
mean_2.append([1.6, -1.2])

# Randomized start points
# mean_1.append(np.random.normal(size=2))
# mean_2.append(np.random.normal(size=2))

# Initial covariance matrix
cov_1 = np.array([[1,0],[0,1]])
cov_2 = np.array([[1,0],[0,1]])

# Calculate propability of each point belonging to group 1,2
prob_1 = multivariate_normal.pdf(data, mean_1[-1], cov_1)
prob_2 = multivariate_normal.pdf(data, mean_2[-1], cov_2)

# Assign point to group dependingon where its probability is highest
group.append(np.where(prob_1 > prob_2, group_1, group_2))

'Steps till groups no longer change'
while len(group) < 2 or np.any(group[-1] != group[-2]):
    # Calculate mean of points in each group
    mean_1.append(np.mean(data[group[-1]==group_1], axis=0))
    mean_2.append(np.mean(data[group[-1]==group_2], axis=0))

    # Calculate covariance matrix of points in each group
    cov_1 = np.cov(data[group[-1]==group_1], rowvar=0)
    cov_2 = np.cov(data[group[-1]==group_2], rowvar=0)

    # Calculate propability of each point belonging to group 1,2
    prob_1 = multivariate_normal.pdf(data, mean_1[-1], cov_1)
    prob_2 = multivariate_normal.pdf(data, mean_2[-1], cov_2)

    # Assign point to group dependingon where its probability is highest
    group.append(np.where(prob_1 > prob_2, group_1, group_2))

# Make a gif of all the steps taken
plot_gif.gif(data, group, mean_1, mean_2)
