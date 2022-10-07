import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from math import floor, ceil

def generate_data(mus, sigmas, priors, N):

    rng = default_rng()
    overall_size = N
    n = mus.shape[0]
    priors = np.cumsum(priors)
    size_1a = 0
    size_1b = 0
    size_2 = 0
    for i in range(0, overall_size) :
        r = random.random()
        if(r < priors[0]):
            size_1a = size_1a + 1
        elif(r < priors[1]):
            size_1b = size_1b + 1
        elif(r < priors[2]):
            size_2 = size_2 + 1

    samples_1a = rng.multivariate_normal(mean=mus[0], cov=sigmas[0], size=size_1a)
    samples_1a = pd.DataFrame(samples_1a, columns=['x','y'])
    samples_1a['True Class Label'] = 1

    samples_1b = rng.multivariate_normal(mean=mus[1], cov=sigmas[1], size=size_1b)
    samples_1b = pd.DataFrame(samples_1b, columns=['x','y'])
    samples_1b['True Class Label'] = 1

    samples_2 = rng.multivariate_normal(mean=mus[2], cov=sigmas[2], size=size_2)
    samples_2 = pd.DataFrame(samples_2, columns=['x','y'])
    samples_2['True Class Label'] = 2

    samples   = samples_1a.append([samples_1b, samples_2])
    return samples

def implement_classifier_and_plots(samples, mus, sigmas, priors, save_path='./ROC_curve.pdf'):
    '''
    Plots the minimum risk and ROC curve with theorectical and experimental probabilites.
    Parameters
    ----------
    samples_path: string
        File containing the sample data
    mus: array
        The vectors of mu for the two classes.
    sigmas: array
        The matrixes of sigma for the two classes.
    priors: array
        The probabilites of the labels.
    Returns
    -------
    exp_min: dict
        Info of the experimental minimum error.
    thy_min: dict
        Info of the theorectical minimum error.
    '''
    # Make decisions
    discriminants = []
    decisions = []
    prior_1 = (priors[0]+priors[1])
    prior_2 = priors[2]
    gamma = prior_1/prior_2
    print(gamma)
    w_1 = 1/2
    w_2 = 1/2
    for i in range(0, samples.shape[0]):
        sample = samples.iloc[i].to_numpy()[:-1]
        discriminant = (w_1*multivariate_normal.pdf(sample, mus[0], sigmas[0])+w_2*multivariate_normal.pdf(sample, mus[1], sigmas[1]))/multivariate_normal.pdf(sample, mus[2], sigmas[2])
        discriminants.append(discriminant)
        if(discriminant>gamma):
            decisions.append(1)
        else:
            decisions.append(2)
    samples['Discriminant'] = discriminants
    samples['Decision'] = decisions

    # Plot ROC curve
    samples = samples.sort_values('Discriminant')
    dis_0 = samples[samples['True Class Label']==1]['Discriminant'].tolist()
    dis_1 = samples[samples['True Class Label']==2]['Discriminant'].tolist()
    df = pd.DataFrame(columns=['False Positive', 'True Positive', 'Gamma', 'Probability Error'])
    for index, row in samples.iterrows():
        discriminant   = row['Discriminant'] 
        false_positive = len([class_dis for class_dis in dis_0 if class_dis>=discriminant])/len(dis_0)
        true_positive = len([class_dis for class_dis in dis_1 if class_dis>=discriminant])/len(dis_1)
        p_err = false_positive*prior_1+(1-true_positive)*prior_2
        d = {'False Positive': false_positive, 'True Positive': true_positive, 
             'Gamma': discriminant, 'Probability Error': p_err}
        df = df.append(d, ignore_index=True)
    df = df.sort_values('Probability Error')
    print(df)
    # Get info of minimum experimental probablility error
    exp_min = df.iloc[0]
    print('Experimental Mimimum Error Info:\n')
    print(exp_min)
    # Calculate theorectical error
    thy_gamma = gamma
    thy_lambdas = [len([class_dis for class_dis in dis_0 if class_dis>=thy_gamma])/len(dis_0),
                len([class_dis for class_dis in dis_1 if class_dis>=thy_gamma])/len(dis_1)]
    thy_p_err = thy_lambdas[0]*prior_1 + (1-thy_lambdas[1])*prior_2
    thy_min = {'False Positive': thy_lambdas[0], 'True Positive': thy_lambdas[1], 'Gamma': thy_gamma, 'Probability Error': thy_p_err}
    print('Theoretical Mimimum Error Info:\n')
    print(thy_min)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    # Plot ROC curve
    ax.plot(df['False Positive'], df['True Positive'], 'ro', markersize=4)
    # Plot experimental minimum
    ax.plot(exp_min['False Positive'], exp_min['True Positive'], 'bo', label='Experimental', markersize=10)
    # Plot theorectical minimum
    ax.plot(thy_min['False Positive'], thy_min['True Positive'], 'go', label='Theoretical', markersize=10)
    ax.legend(title='Minimum Error Probabilities', loc='upper left')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('Probability of False Positive')
    ax.set_ylabel('Probability of True Positive')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig('ROC_curve.pdf')
    plt.clf()
    plt.close()

    # Plot data set and outcomes
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for idx,row in samples.iterrows():
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
    plt.savefig('q2_p1.pdf')

def ldaClassifier(samples, mu, sigma, priors):
    prior1 = priors[0] + priors[1]
    prior2 = priors[2]

    class1 = np.where(samples['True Class Label'] == 1)

if __name__ == "__main__":
    priors = [.325,.325,.35]
    mus = np.array([[3, 0], [0, 3], [2, 2]])
    covs = np.zeros((3, 2, 2))
    covs[0,:,:] = np.array([[2, 0], [0, 1]])
    covs[1,:,:] = np.array([[1, 0], [0, 2]])
    covs[2,:,:] = np.array([[1, 0], [0, 1]])
    test = generate_data(mus, covs, priors, 10000)

    ldaClassifier(test, mus, covs, priors)
    #implement_classifier_and_plots(test, mus, covs, priors)