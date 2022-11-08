import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from LikelihoodRatio import likelihood_ratio_test

# Read in the data
def read_data(filename):
    data = pd.read_csv(filename)
    return data

# Fit a gaussian model for the data and return the parameters using fitgmdist function
def fit_gaussian(data, mixtureComponents, classLabel):
    X = np.empty((0,2), float)
    prior = len(data[data['True Class Label'] == classLabel])/len(data)
    # Fit a Gaussian mixture with EM using fitgmdist function
    GMModel = GMM(n_components = mixtureComponents, covariance_type = 'full', verbose = 0, tol = 1e-6)
    X = data[data['True Class Label'] == classLabel][['x', 'y']].to_numpy()
    print(X.shape)
    # print(X)
    GMModel = GMModel.fit(X)
    #gmm = stats.fitgmdist(data, 1)
    # Return the parameters
    return GMModel.means_, GMModel.covariances_, GMModel.weights_, prior

def TrainAndPlot(data, NameofFile):
    # Fit a Gaussian mixture model for class 1
    mu1, sigma1, w1, prior1 = fit_gaussian(data, 2, 1)
    print('mu1: ', mu1)
    print('sigma1: ', sigma1)
    print('w1: ', w1)
    print('prior1: ', prior1)
    #Plot Gaussian mixture model of Class 1
    # Plot contours of the estimated mixture model for class 1
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    x, y = np.mgrid[-10:10:.01, -10:10:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    # create multivariate mixture gaussian pdf
    class11 = stats.multivariate_normal(mu1[0], sigma1[0])
    class12 = stats.multivariate_normal(mu1[1], sigma1[1])
    # plot contours of class1
    ax[0].contour(x, y, w1[0]*class11.pdf(pos) + w1[1]*class12.pdf(pos))
    ax[0].set_title('Estimated Mixture Model for Class 1')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_ylim(-3,8)
    ax[0].set_xlim(-3,9)
    ax[0].scatter(data[data['True Class Label'] == 1]['x'], data[data['True Class Label'] == 1]['y'], s=0.5, c='r')
    # Fit a Gaussian mixture model for class 2
    mu2, sigma2, w2, prior2 = fit_gaussian(data, 1, 2)
    print('mu2: ', mu2)
    print('sigma2: ', sigma2)
    print('w2: ', w2)
    print('prior2: ', prior2)
    # Plot contours of the estimated mixture model for class 2
    # Plot contours of the estimated mixture model for class 2
    x, y = np.mgrid[-10:10:.01, -10:10:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    ax[1].set_title('Estimated Model for Class 2')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_ylim(-2,6)
    ax[1].set_xlim(0,6)
    rv = stats.multivariate_normal(mu2[0], sigma2[0])
    ax[1].contour(x, y, rv.pdf(pos))
    ax[1].scatter(data[data['True Class Label'] == 2]['x'], data[data['True Class Label'] == 2]['y'], s=1)
    #Combine the parameters
    mu = np.array([mu1[0], mu1[1], mu2[0]])
    sigma = np.array([sigma1[0], sigma1[1], sigma2[0]])
    prior = np.array([w1[0]*prior1, w1[1]*prior1, prior2])
    w = np.array([w1[0], w1[1], w2[0]])
    #Save plots
    plt.savefig('./HW3/'+NameofFile+'Estimated Gaussian models.png', dpi=500)
    plt.clf()
    plt.close()
    # Return the parameters
    return mu, sigma, prior, w

if __name__ == "__main__":
    # Read in the 10K training data
    Train10KSamples = read_data("./HW3/Train_10000samples.csv")
    # Read in the 100 training data
    Train100Samples = read_data("./HW3/Train_100samples.csv")
    # Read in the 1000 training data
    Train1000Samples = read_data("./HW3/Train_1000samples.csv")
    Validation20KSamples = read_data("./HW3/Validation_20Ksamples.csv")
    # Fit a Gaussian mixture model for the data
    mu, sigma, prior, weights = TrainAndPlot(Train10KSamples, "10K")
    # Fit a gaussian model for 1000 samples
    mu1000, sigma1000, prior1000, weights1000 = TrainAndPlot(Train1000Samples,"1000Samples")
    # Fit a gaussian model for 100 samples
    mu100, sigma100, prior100, weights100 = TrainAndPlot(Train100Samples,"100Samples")
    # Test the model with likelihood ratio test
    likelihood_ratio_test(Validation20KSamples, mu, sigma, prior, './HW3/MLE_TrainedWith_10K_Samples', [weights[0], weights[1]], "Train with 10K samples")
    likelihood_ratio_test(Validation20KSamples, mu100, sigma100, prior100, './HW3/MLE_TrainedWith_1000_Samples', [weights100[0], weights100[1]], "Train with 1000 samples")
    likelihood_ratio_test(Validation20KSamples, mu1000, sigma1000, prior1000, './HW3/MLE_TrainedWith_100_Samples', [weights1000[0], weights1000[1]], "Train with 100 samples")



