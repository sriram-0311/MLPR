import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import random
import sys
import os
import scipy

TrueTheta = np.random.randint(0, 360)
TrueTheta = np.radians(TrueTheta)
TrueR = np.random.random()*0.75

TruePos = np.array([TrueR*np.cos(TrueTheta), TrueR*np.sin(TrueTheta)])

print("True Position: ", TruePos)

# plot the centre of the contour
def center_of_mass(X):
    # calculate center of mass of a closed polygon
    x = X[:,0]
    y = X[:,1]
    g = (x[:-1]*y[1:] - x[1:]*y[:-1])
    A = 0.5*g.sum()
    cx = ((x[:-1] + x[1:])*g).sum()
    cy = ((y[:-1] + y[1:])*g).sum()
    return 1./(6*A)*np.array([cx,cy])

# MAP Log Likelihood function
def MAPLogLikelihood(x, y, sigma_x, sigma_y, sigma, theta):
    K = x.shape[0]
    log_likelihood = 0
    for i in range(K):
        log_likelihood += (1/(sigma[i]**2))*(np.subtract(y[i],np.linalg.norm(np.subtract(x[i], theta))))**2

    cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
    prior = np.dot(np.dot(theta.T, np.linalg.inv(cov)), theta)

    return (-1/(2*K))*(log_likelihood + prior)

for K in [1,2,3,4,40]:
    plt.figure()
    plt.plot(TruePos[0], TruePos[1], 'r+', label="True Position") # Plot the true position
    # print("K = ", K)
    ReferenceTheta = np.linspace(0, 360, K+1)
    x = np.zeros((K,2))
    # print("X: ", x)
    y = np.zeros((K,1))
    # print("Y: ", y)
    sigma = 0.3*np.ones((K,1))
    # print("Sigma: ", sigma)
    # print("sigma shape: ", sigma[0][0])
    for i in range(K):
        x[i][0] = np.cos(np.radians(ReferenceTheta[i]))
        x[i][1] = np.sin(np.radians(ReferenceTheta[i]))
        # Normalizing the reference vectors
        y[i] = np.linalg.norm(np.subtract(x[i],TruePos))
        noise = np.random.normal(0, sigma[i])
        while y[i] + noise < 0:
            noise = np.random.normal(0, sigma[i])
        y[i] = y[i] + noise

    # print("X: ", x)
    # print("Y: ", y)

    sigma_x = 0.25
    sigma_y = 0.25

    # Plotting the contour plot for the MAP Log Likelihood
    mapEstimate = np.inf
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    F = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            # print("i: ", i, "j: ", j)
            # print("X1: ", X1[i][j], "X2: ", X2[i][j])
            # print("MAPLogLikelihood: ", MAPLogLikelihood(x, y, sigma_x, sigma_y, sigma, np.array([X1[i][j], X2[i][j]])))
            F[i,j] = MAPLogLikelihood(x, y, sigma_x, sigma_y, sigma, np.array([X1[i,j], X2[i,j]]))
            if F[i,j] < mapEstimate:
                mapEstimate = F[i,j]
                mapEstimatePos = np.array([X1[i,j], X2[i,j]])
    # Plot the minimum of the loglikelihood function as the MAP estimate
    #min_result = scipy.optimize.minimize(MAPLogLikelihood(x,y,sigma_x, sigma_y, sigma,np.array([0,0])),[0,0],args=(ReferenceTheta), method='SLSQP', bounds = [(-2,2),(-2,2)])
    # print("MAP Estimate: ", mapEstimatePos)
    # plt.plot(mapEstimatePos[0], mapEstimatePos[1], 'r*', label='MAP Estimate')
    # plot the position estimated by MAP as a red star
    cont = plt.contourf(X1, X2, F)
    # plot the centre of the contour
    plt.plot(center_of_mass(cont.allsegs[-1][0])[0], center_of_mass(cont.allsegs[-1][0])[1], 'rx', label='Map Estimate')
    # plot colorbar
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('MAP Log Likelihood')
    # plot center of the contour
    plt.plot(x[:,0], x[:,1], 'bo', label='Reference Vectors')
    plt.title('MAP Log Likelihood')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./HW3/MAPLogLikelihood'+ str(K) + '.png')