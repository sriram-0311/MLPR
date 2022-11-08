import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys
#import minimiser optimiser
from scipy.optimize import minimize
import LLF

def quadratic(x):
    return np.array([1, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[1]])

def optimise_w(x, y):
    w0 = np.zeros(6)
    w = minimize(loss, w0, args=(x, y), method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
    #print("w: ", w)
    return w.x

def fit(x, y):
    w = optimise_w(x, y)
    return w

def sigmoid(x, w):
    return 1/(1+np.exp(-np.dot(quadratic(x),w)))

# Binary loss function
def loss(w, x, y):
    ErrorSum = 0
    for i in range(len(x)):
        sigmoidValue = y[i]*np.log(sigmoid(x[i], w)) + (1-y[i])*np.log(1-sigmoid(x[i], w))
        ErrorSum += sigmoidValue

    loss = 1/(len(x)*ErrorSum)

    return loss

# Calculate threshold for minimum PError
def threshold(x, y, w,prior, data):
    #initialize variables
    Discriminant = []
    Decision = []
    # Calculate theoritical threshold value
    # Calculate the likelihood ratio
    for i in range(0, len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        thy_threshold = prior[0]/prior[1]
        # Calculate the likelihood ratio
        likelihood_ratio = sigmoid([x,y], w)
        Discriminant.append(likelihood_ratio)
        # If the likelihood ratio is larger than the threshold value, then the point is classified as class 1
        if likelihood_ratio < thy_threshold:
            Decision.append(1)
        # If the likelihood ratio is smaller than the threshold value, then the point is classified as class 2
        else:
            Decision.append(2)

    data['Decision'] = Decision
    data['Discriminant'] = Discriminant

    data.sort_values(by=['Discriminant'])
    class1 = data[data['True Class Label'] == 1]['Discriminant'].tolist()
    class2 = data[data['True Class Label'] == 2]['Discriminant'].tolist()
    # Create dataFrame to store the True positive and False positive rate
    df = pd.DataFrame(columns=['True Positive Rate', 'False Positive Rate', 'Threshold', 'PError'])
    # Generate the ROC Curve
    for i, rows in data.iterrows():
        discriminant = rows['Discriminant']
        FalsePositive = len([x for x in class1 if x > discriminant])/len(class1)
        TruePositive = len([x for x in class2 if x > discriminant])/len(class2)
        df.loc[i] = [TruePositive, FalsePositive, discriminant, ((prior[0])*FalsePositive)+((prior[1])*(1-TruePositive))]

    df = df.sort_values(by=['PError'])
    ExperimentalMinimum = df.iloc[0]
    print('Experimental Minimum PError: ', ExperimentalMinimum['PError'])
    print('Experimental Threshold: ', ExperimentalMinimum['Threshold'])
    print('Experimental True Positive Rate: ', ExperimentalMinimum['True Positive Rate'])
    print('Experimental False Positive Rate: ', ExperimentalMinimum['False Positive Rate'])

    return ExperimentalMinimum['Threshold'], ExperimentalMinimum['PError']

if __name__ == "__main__":
    # Read in the data
    data = LLF.read_data('HW3/Train_10000samples.csv')
    # Read validation data
    validationData = LLF.read_data('HW3/Validation_20Ksamples.csv')
    # Extract the data
    x = data[['x','y']].to_numpy()
    y = data['True Class Label'].to_numpy()
    y = np.subtract(y, 1)
    # print("y", y)
    # Fit the data to the model
    w = fit(x, y)
    print(w)

    # Extract the validation data
    x_validation = validationData[['x','y']].to_numpy()
    y_validation = validationData['True Class Label'].to_numpy()
    y_validation = np.subtract(y_validation, 1)
    # Find the threshold for minimum PError
    Priors = [len(y_validation[y_validation == 0])/len(y_validation), len(y_validation[y_validation == 1])/len(y_validation)]
    print("Priors", Priors)
    minThreshold, minPError = threshold(x_validation, y_validation, w, Priors, validationData)
    print("minThreshold", minThreshold)
    print("minPError", minPError)

    # Plot the contours of the model
    x1 = np.linspace(-15, 15, 100)
    x2 = np.linspace(-15, 15, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.array([X1, X2])
    X = np.reshape(X, (2, 10000))
    X = np.transpose(X)
    Y = np.zeros(10000)
    for i in range(10000):
        Y[i] = sigmoid(X[i], w)
    Y = np.reshape(Y, (100, 100))
    plt.contour(X1, X2, Y, levels=[0.5])
    plt.title('Contour plot of the model')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-4, 9)
    plt.ylim(-4, 9)

    # Scatter plot of the validation data
    scatt = plt.scatter(x_validation[:, 0], x_validation[:, 1], c=y_validation, cmap=plt.cm.Paired, s=0.3)
    legend1 = plt.legend(*scatt.legend_elements(),loc="upper right", title="Classes")
    #add legend1 to plt
    plt.gca().add_artist(legend1)
    # Add legend for class label

    plt.savefig('./HW3/LQF.png', dpi=700)
    plt.clf()
    plt.close()