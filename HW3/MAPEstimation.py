import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import random
import sys
import os

TrueTheta = np.random.randint(0, 360)
TrueTheta = np.radians(TrueTheta)
TrueR = np.random.random()*0.75

TruePos = np.array([TrueR*np.cos(TrueTheta), TrueR*np.sin(TrueTheta)])

print("True Position: ", TruePos)

# MAP Log Likelihood function
def MAPLogLikelihood(x, y, sigma_x, sigma_y, sigma, theta):
    K = x.shape[0]
    log_likelihood = 0
    for i in range(K):
        log_likelihood += (1/(sigma[i]**2))*(np.subtract(y[i],np.linalg.norm(np.subtract(x[i], theta))))**2

    cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
    prior = np.dot(np.dot(theta.T, np.linalg.inv(cov)), theta)

    return (1/(2*K))*(log_likelihood + prior)

for K in [1,2,3,4,40]:
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
    plt.figure()
    cont = plt.contourf(X1, X2, F)
    plt.plot(TruePos[0], TruePos[1], 'ro')
    plt.plot(x[:,0], x[:,1], 'bo', label='Reference Vectors')
    plt.title('MAP Log Likelihood')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./HW3/MAPLogLikelihood'+ str(K) + '.png')


# function LL = map_loglike(theta, sig_x,sig_y, x, y, sigma)
#     [~, K] = size(x);

#     p1 = 0; % sum of ln(p(y_i given x_i theta))
#     for i=1:K
#         p1 = p1 + sigma(i)^-2 * (y(i)-norm(x(:, i)-theta))^2; % data points based
#     end

#     cov = [sig_x^2 0 ; 0 sig_y^2];
#     p2 = theta.'*inv(cov)*theta; % ln(p(theta)) part.

#     LL = p1 + p2;
#     LL = 1/(2*K) * LL;
# end

#    sigma_y = 0.25;

#    % plot pieces
#    figure;
#    plot(x(1, :), x(2,:), 'ob'); hold on;
#    plot(pos_true(1), pos_true(2), '+g');
#    title(sprintf('plot for k=%i', K));
#    xlim([-2 2]);
#    ylim([-2 2]);
#    % contours functions, takes in a [xe0, ye0] (candidate theta), and finds
#    % the LL at that point.
#    pdf = @(xe,ye) arrayfun(@(xe0,ye0) map_loglike([xe0;ye0], sigma_x, sigma_y, x,y, sigma),xe,ye);
#    g = gca;
#    fc = fcontour(pdf,[g.XLim g.YLim]);
#    fc.LineStyle='--';
#    fc.LevelList = exp(-1:0.1:3);
#    colorbar
#    hold off;
#    disp(map_loglike(pos_true, sigma_x, sigma_y, x,y, sigma));
# end