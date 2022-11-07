import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM

# Read in the data
def read_data(filename):
    data = pd.read_csv(filename)
    return data

# Fit a gaussian model for the data and return the parameters using fitgmdist function
def fit_gaussian(data, mixtureComponents, classLabel):
    X = np.empty((0,2), float)
    # Fit a Gaussian mixture with EM using fitgmdist function
    GMModel = GMM(n_components = mixtureComponents, covariance_type = 'diag', verbose = 0, tol = 1e-3)
    # for i, rows in data.iterrows():
    #     if rows['True Class Label'] == classLabel:
    #         if i == 0:
    #             X = np.array([rows['x'], rows['y']])
    #         else:
    #             try:
    #                 X = np.vstack((X, np.array([rows['x'], rows['y']])))
    #             except UnboundLocalError:
    #                 X = np.array([rows['x'], rows['y']])
    X = data[data['True Class Label'] == classLabel][['x', 'y']].to_numpy()
    # print(X.shape)
    # print(X)
    GMModel = GMModel.fit(X)
    #gmm = stats.fitgmdist(data, 1)
    # Return the parameters
    return GMModel.means_, GMModel.covariances_, GMModel.weights_

if __name__ == "__main__":
    # Read in the data
    data = read_data("./HW3/Train_10000samples.csv")
    # Fit a Gaussian mixture model for the data
    gmm_mean, gmm_cov, gmm_weight = fit_gaussian(data, 2, 1)
    # Print the parameters
    print('mu: ', gmm_mean)
    print('sigma: ', gmm_cov)
    print('prior: ', gmm_weight)

    # Plot contours of the estimated mixture model for class 1
    #fig, ax = plt.subplots(1,2, figsize=(5,5))
    x, y = np.mgrid[-10:10:.01, -10:10:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = stats.multivariate_normal(gmm_mean[1], gmm_cov)
    plt.contour(x, y, rv.pdf(pos))
    plt.scatter(data[data['True Class Label'] == 1]['x'], data[data['True Class Label'] == 1]['y'], s=1)

    class2_mean, class2_cov, class2_weight = fit_gaussian(data, 1, 2)
    print('mu2: ', class2_mean)
    print('sigma2: ', class2_cov)
    print('prior2: ', class2_weight)

    # # Plot contours of the estimated mixture model for class 2
    # x, y = np.mgrid[-10:10:.01, -10:10:.01]
    # pos = np.empty(x.shape + (2,))
    # pos[:, :, 0] = x
    # pos[:, :, 1] = y
    # rv = stats.multivariate_normal(class2_mean, class2_cov)
    # ax[0,1].contour(x, y, rv.pdf(pos))

    plt.show()
