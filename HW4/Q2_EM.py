# code to run EM algorithm for GMM on 2D data with k fold cross validation

from typing import List, Tuple
import sklearn.mixture as mixture
import numpy as np
import numpy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# plot the model's performance versus the data
def PlotModelPerformance(model, data, outfile=None):
    '''Plot the model's performance versus the data.
    model: the model to plot
    data: the data to plot
    outfile: the file to save the plot to, or None to show the plot'''

    if outfile is not None:
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        
        # showing predicted scores as contour plot
        x = np.linspace(-10,10)
        y = np.linspace(-10,10)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -model.score_samples(XX)
        Z = Z.reshape(X.shape)

        ContourPlot = ax.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))

        ColorBar = fig.colorbar(ContourPlot, ax=ax, shrink=0.9, extend='both')

        ax.scatter(data[:,0], data[:,1], .8)
        ax.set_title('Negative log-likelihood predicted by a GMM')

        fig.savefig(outfile, dpi=700)

        plt.close(fig)
    else:
        raise NotImplementedError('Plotting to screen not implemented')

# run the EM algorithm on the data
def ModelForKComponents(data, k, numStarts=1):
    '''Run the EM algorithm on the data.
    data: the data to run the algorithm on
    k: the number of components to use
    max_iterations: the maximum number of iterations to run the algorithm for'''

    model = mixture.GaussianMixture(n_components=k, covariance_type='full', n_init=numStarts)
    model.fit(data)

    return model

# return the log likelihood value
def LogLikelihood(model, data):
    LogLikelihood = model.score_samples(data.T)
    return np.sum(LogLikelihood)

# define the K fold cross validation
def KFoldCrossValidation(data, k, nFolds=5):
    data = data.T

    # shuffle the data
    np.random.shuffle(data)
    samples = data.shape[0]
    dataIndexes = np.arange(samples)
    np.random.shuffle(dataIndexes)

    folds = np.array_split(dataIndexes, nFolds)
    exp = []

    for i in range(nFolds):
        test = data[folds[i]]
        train = data[np.concatenate(folds[:i] + folds[i+1:])]

        model = ModelForKComponents(train.T, k)
        exp.append([folds[i],LogLikelihood(model, test.T)])

    SumScore = 0
    for i in exp:
        SumScore += (i[0]/samples) * i[1]

    return SumScore

# select the best component
def SelectBestComponent(Components, data, testData, plotSelections):
    '''Select the best component.
    Components: the components to select from
    data: the data to select the best component for
    testData: the test data to use to select the best component
    plotSelections: whether to plot the model performance for each component'''

    # select the best component
    bestComponent = None
    bestScore = None
    for component in tqdm(Components):
        # run the EM algorithm
        model = ModelForKComponents(data, component)
        score = KFoldCrossValidation(data, component)

        if bestComponent is None or score > bestScore:
            bestComponent = component
            bestScore = score

        if plotSelections:
            PlotModelPerformance(model, testData, 'Q2_EM_Component_{}.png'.format(component))

    if plotSelections:
        fig, ax = plt.subplots()
        

    return bestComponent





