import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys

# Read in the data
def read_data(filename):
    data = pd.read_csv(filename)
    return data

# Likelihood ratio test for loaded data
def likelihood_ratio_test(data, mu, sigma, prior):
    #initialize variables
    Discriminant = []
    Decision = []
    # Calculate theoritical threshold value
    thy_threshold = (prior[0]+prior[1])/prior[2]
    w1, w2 = 0.5, 0.5
    # Calculate the likelihood ratio
    for i in range(0, len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        # Calculate the likelihood ratio
        likelihood_ratio = ((stats.multivariate_normal.pdf([x, y], mean=mu[0], cov=sigma[0]) * w1) + (stats.multivariate_normal.pdf([x, y], mean=mu[1], cov=sigma[1]) * w2))/stats.multivariate_normal.pdf([x, y], mean=mu[2], cov=sigma[2])
        Discriminant.append(likelihood_ratio)
        # If the likelihood ratio is larger than the threshold value, then the point is classified as class 1
        if likelihood_ratio > thy_threshold:
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
        FalsePositive = len([x for x in class1 if x < discriminant])/len(class1)
        TruePositive = len([x for x in class2 if x < discriminant])/len(class2)
        df.loc[i] = [TruePositive, FalsePositive, discriminant, (prior[0]+prior[1])*FalsePositive+prior[2]*(1-TruePositive)]

    df = df.sort_values(by=['PError'])
    ExperimentalMinimum = df.iloc[0]
    print('Experimental Minimum PError: ', ExperimentalMinimum['PError'])
    print('Experimental Threshold: ', ExperimentalMinimum['Threshold'])
    print('Experimental True Positive Rate: ', ExperimentalMinimum['True Positive Rate'])
    print('Experimental False Positive Rate: ', ExperimentalMinimum['False Positive Rate'])

    TheoriticalRates = [len([x for x in class1 if x < thy_threshold])/len(class1),len([x for x in class2 if x < thy_threshold])/len(class2)]
    TheoriticalPError = (prior[0]+prior[1])*TheoriticalRates[0]+prior[2]*(1-TheoriticalRates[1])

    print('Theoritical Minimum PError: ', TheoriticalPError)
    print('Theoritical Threshold: ', thy_threshold)
    print('Theoritical True Positive Rate: ', TheoriticalRates[1])
    print('Theoritical False Positive Rate: ', TheoriticalRates[0])

    # Plot the ROC Curve
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    ax.plot(df['False Positive Rate'], df['True Positive Rate'], 'bo', markersize=2)
    ax.plot(TheoriticalRates[0], TheoriticalRates[1], 'ro', markersize=5, label='Theoritical Minimum')
    ax.plot(ExperimentalMinimum['False Positive Rate'], ExperimentalMinimum['True Positive Rate'], 'go', markersize=5, label='Experimental Minimum')
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    ax.yaxis.grid(color='lightgrey')
    ax.xaxis.grid(color='lightgrey')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.legend()
    plt.savefig('./HW3/ROC_curve.png', dpi=500)
    plt.clf()
    plt.close()

    # Plot the Decision boundary
    # fig, ax = plt.subplots(1,1, figsize=(5, 5))
    # ax.plot(data[data['True Class Label'] == 1]['x'], data[data['True Class Label'] == 1]['y'], 'bo', markersize=2)
    # ax.plot(data[data['True Class Label'] == 2]['x'], data[data['True Class Label'] == 2]['y'], 'ro', markersize=2)
    # ax.plot(data[data['Decision'] == 1]['x'], data[data['Decision'] == 1]['y'], 'go', markersize=2)
    # ax.plot(data[data['Decision'] == 2]['x'], data[data['Decision'] == 2]['y'], 'yo', markersize=2)

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for idx,row in data.iterrows():
        true_label = row['True Class Label']
        decision   = row['Decision']
        x = row['x']
        y = row['y']
        if(true_label==1):
            if(true_label==decision):
                ax.plot(x,y,'go', alpha=0.1)
            else:
                ax.plot(x,y,'ro', alpha=0.1)
        else:
            if(true_label==decision):
                ax.plot(x,y,'g^', alpha=0.1)
            else:
                ax.plot(x,y,'r^', alpha=0.1)

    legend = ax.legend({'True Class 1':'go'},loc="upper right", title="Decisions")
    ax.add_artist(legend)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.yaxis.grid(color='lightgrey')
    ax.xaxis.grid(color='lightgrey')
    ax.set_axisbelow(True)
    plt.savefig('./HW3/Decision_boundary.png', dpi=500)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    data = read_data("./HW3/Validation_20Ksamples.csv")
    mu = np.array([[5,0],[0,4],[3,2]])
    sigma = np.array([[[4,0],[0,2]],[[1,0],[0,3]],[[2,0],[0,2]]])
    prior = np.array([0.3,0.3,0.4])
    likelihood_ratio_test(data, mu, sigma, prior)

