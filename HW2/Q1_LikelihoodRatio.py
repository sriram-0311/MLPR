from matplotlib.markers import MarkerStyle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import multivariate_normal, norm
from math import floor, ceil
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings

warnings.filterwarnings("ignore")

from stack_data import MarkerInLine

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
    ax.plot(df['True Positive'], df['False Positive'], 'go', markersize=2)
    # Plot experimental minimum
    ax.plot(exp_min['False Positive'], exp_min['True Positive'], 'bx', label='Experimental', markersize=10)
    # Plot theorectical minimum
    ax.plot(thy_min['False Positive'], thy_min['True Positive'], 'rx', label='Theoretical', markersize=10)
    ax.legend(title='Minimum Error Probabilities', loc='upper left')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    ax.yaxis.grid(color='lightgrey')
    ax.xaxis.grid(color='lightgrey')
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

def imp(samples, mean_0, mean_1, mean_2, cov_0, cov_1, cov_2):
    discriminants = []
    for i in range(0, samples.shape[0]):
        sample = samples.iloc[i].to_numpy()[:-1]
        discriminant = multivariate_normal.pdf(sample, mean_1, cov_1)/multivariate_normal.pdf(sample, mean_0, cov_0)
        discriminant = (0.5*multivariate_normal.pdf(sample, mean_1, cov_1)+0.5*multivariate_normal.pdf(sample, mean_2, cov_2))/multivariate_normal.pdf(sample, mean_2, cov_2)
        discriminants.append(discriminant)
    samples['Discriminant'] = discriminants
    samples = samples.sort_values('Discriminant')
    dis_0 = samples[samples['True Class Label']==1]['Discriminant'].tolist()
    dis_1 = samples[samples['True Class Label']==2]['Discriminant'].tolist()
    df = pd.DataFrame(columns=['False Positive', 'True Positive', 'Gamma', 'Probability Error'])
    for index, row in samples.iterrows():
        discriminant   = row['Discriminant'] 
        false_positive = len([class_dis for class_dis in dis_0 if class_dis>=discriminant])/len(dis_0)
        true_positive = len([class_dis for class_dis in dis_1 if class_dis>=discriminant])/len(dis_1)
        p_err = false_positive*0.65+(1-true_positive)*0.325
        d = {'False Positive': false_positive, 'True Positive': true_positive, 
             'Gamma': discriminant, 'Probability Error': p_err}
        df = df.append(d, ignore_index=True)
    df = df.sort_values('Probability Error')
    # Get info of minimum experimental probablility error
    exp_min = df.iloc[0]
    print('Experimental Mimimum Error Info:\n')
    print(exp_min)
    # Calculate theorectical error
    thy_gamma = 0.7/0.3
    thy_lambdas = [len([class_dis for class_dis in dis_0 if class_dis>=thy_gamma])/len(dis_0),
                len([class_dis for class_dis in dis_1 if class_dis>=thy_gamma])/len(dis_1)]
    thy_p_err = thy_lambdas[0]*0.7 + (1-thy_lambdas[1])*0.3
    thy_min = {'False Positive': thy_lambdas[0], 'True Positive': thy_lambdas[1], 'Gamma': thy_gamma, 'Probability Error': thy_p_err}
    print('Theoretical Mimimum Error Info:\n')
    print(thy_min)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    # Plot ROC curve
    ax.plot(df['False Positive'], df['True Positive'], 'go', markersize=2)
    # Plot experimental minimum
    ax.plot(exp_min['False Positive'], exp_min['True Positive'], 'bx', label='Experimental', markersize=5)
    # Plot theorectical minimum
    ax.plot(thy_min['False Positive'], thy_min['True Positive'], 'rx', label='Theoretical', markersize=5)
    ax.legend(title='Minimum Error', loc='upper right')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    ax.yaxis.grid(color='lightgrey')
    ax.xaxis.grid(color='lightgrey')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    plt.savefig('ROCCurve.pdf')

def classSeparator(samples, mu, sigma, priors):
    print("samples iloc : ",samples.iloc[1].to_numpy()[:-1])
    prior1 = priors[0] + priors[1]
    prior2 = priors[2]
    class1 = np.array([0,0])
    # print("class1 : ",class1)
    class2 = np.array([0,0])
    classover = np.array([0,0])
    labels = np.array(0)
    for index, row in samples.iterrows():
        if (row['True Class Label'] == 1):
            #class1 = np.insert(class1, [row['x'], row['y']])
            class1 = np.vstack([class1, [row['x'], row['y']]])
        elif (row['True Class Label'] == 2):
            class2 = np.vstack([class2, [row['x'], row['y']]])
        classover = np.vstack([classover, [row['x'],row['y']]])
        labels = np.append(labels, row['True Class Label'])
    # print("class1 : ",class1)
    # print("class2 : ",class2)
    return class1, class2, classover, labels

def ldaClassifierVector(class1, class2):
    mu1 = np.mean(class1, axis=0)
    mu2 = np.mean(class2, axis=0)

    sigma1 = np.cov(np.transpose(class1))
    sigma2 = np.cov(np.transpose(class2))

    print("class 1 covariance : ",np.shape(np.reshape(np.transpose(np.subtract(mu1,mu2)),(1,2))))
    print("class 2 covariane : ", np.shape(mu2))

    Sw = sigma1 + sigma2
    projvec = np.matmul(np.linalg.inv(Sw), np.subtract(mu1,mu2))
    Sb = np.matmul(np.reshape(np.transpose(np.subtract(mu1,mu2)),(2,1)),np.reshape(np.subtract(mu1,mu2),(1,2)))
    print("sb matrix : \n", Sb)
    print("shape of sb: \n", np.linalg.inv(Sw))

    ldaProjectionmatrix = np.matmul(np.linalg.inv(Sw),Sb)

    [V,d] = np.linalg.eig(ldaProjectionmatrix)
    print("V : \n",V)
    print("d : \n ",d)
    if V[0] < V[1]:
        projectionVec = d[0]
    else:
        projectionVec = d[1]
    #projectionVector = V[:,1]

    print(d[1])

    return projectionVec

def ldaClassifier(vect, samples, mus, sigmas, priors):
    discriminants = []
    decisions = []
    prior_1 = (priors[0]+priors[1])
    prior_2 = priors[2]
    gamma = prior_1/prior_2
    print(gamma)
    print("number of data : \n", len(samples))
    for i in range(0, samples.shape[0]):
        sample = samples.iloc[i].to_numpy()[:-1]
        discriminant = np.matmul(np.transpose(vect), sample)
        #print("discriminant :",discriminant)
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

    # ThresholdValues = np.arange(-5000, 5000,1)
    # for thresh in ThresholdValues:
    #     false_positive = len([class_dis for class_dis in dis_0 if class_dis>=thresh])/len(dis_0)
    #     true_positive = len([class_dis for class_dis in dis_1 if class_dis>=thresh])/len(dis_1)
    #     p_err = false_positive*prior_1+(1-true_positive)*prior_2
    #     d = {'False Positive': false_positive, 'True Positive': true_positive,
    #          'Gamma': thresh, 'Probability Error': p_err}
    #     df = df.append(d, ignore_index=True)
    #     if (thresh == 0):
    #         print("ROC : ",false_positive,"\t",true_positive)
    df = df.sort_values('Probability Error')
    print(df)
    # Get info of minimum experimental probablility error
    exp_min = df.iloc[0]
    print('Experimental Mimimum Error Info LDA:\n')
    print(exp_min)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    # Plot ROC curve
    ax.plot(df['False Positive'], df['True Positive'], 'bo', markersize=4)
    # Plot experimental minimum
    ax.plot(exp_min['False Positive'], exp_min['True Positive'], 'ro', label='experimental', markersize=4)
    ax.legend(title='Minimum Error Probabilities', loc='upper left')
    #ax.set_title('Minimum Expected Risk ROC Curve')
    ax.set_xlabel('Probability of False Positive')
    ax.set_ylabel('Probability of True Positive')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig('ROC_curve_LDA.pdf')
    plt.clf()
    plt.close()

def scatterplot(samples, ldaVector):
    rng = default_rng()
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    transformedClass1 = np.array([0,0])
    transformedClass2 = np.array([0,0])
    for idx,row in samples.iterrows():
        true_label = row['True Class Label']
        x = row['x']
        y = row['y']
        if(true_label==1):
            ax.plot(x,y,'go', alpha=0.1)
            #transformedClass1 = np.append(transformedClass1, np.matmul(np.transpose(ldaVector), [x,y]))
        else:
            ax.plot(x,y,'r^', alpha=0.1)
            #transformedClass2 = np.append(transformedClass2, np.matmul(np.transpose(ldaVector), [x,y]))

    # transmu1 = np.mean(transformedClass1)
    # transmu2 = np.mean(transformedClass2)

    # transcov1 = np.cov(transformedClass1)
    # transcov2 = np.cov(transformedClass2)

    # rv1 = norm(transmu1,transcov1)
    # rv2 = norm(transmu2,transcov2)

    # minx = min(min(transformedClass1),min(transformedClass2))
    # maxx = max(max(transformedClass1), max(transformedClass2))

    # print("u1 and u2 : \t",transmu1,transmu2)

    # ax.plot(np.arange(minx,maxx,0.01),rv1.pdf(np.arange(minx,maxx,0.01)), 'bo', markersize=1)
    # ax.plot(np.arange(minx,maxx,0.01),rv2.pdf(np.arange(minx,maxx,0.01)), 'go', markersize=1)

    # ax.set_ylim(0,0.5)
    # ax.set_xlim(-3,3)

    ax.plot([-100*ldaVector[0],100*ldaVector[0]],[-100*ldaVector[1],100*ldaVector[1]],'black', markersize=10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(title='LDA Vector and Random generated data', loc='upper left')

    ax.set_xlim(-6,7)
    ax.set_ylim(-4,8)

    plt.savefig('LDATransformedVector.pdf')

def scikitLDA(datum, labels, c1, c2):
    clf = LDA()
    clf.fit(datum, labels)
    tc1 = clf.transform(c1)
    tc2 = clf.transform(c2)
    print("Printing parameters: \n",clf.get_params(True))

    fig, ax = plt.subplots(1,1, figsize=(5,5))

    # print(transformedData[:,0])
    # print(transformedData[:,1])

    # ax.plot([transformedData[:,0]],[transformedData[:,1]], 'g^')
    ax.plot([tc1[:,0]],[tc1[:,1]], 'g^')
    ax.plot([tc2[:,0]],[tc2[:,1]], 'bo')
    plt.show()
    print(clf.predict([[-0.8, -1]]))


if __name__ == "__main__":
    priors = [.325,.325,.35]
    mus = np.array([[3, 0], [0, 3], [2, 2]])
    covs = np.zeros((3, 2, 2))
    covs[0,:,:] = np.array([[2, 0], [0, 1]])
    covs[1,:,:] = np.array([[1, 0], [0, 2]])
    covs[2,:,:] = np.array([[1, 0], [0, 1]])
    test = generate_data(mus, covs, priors, 10000)

    class1, class2, datum, labels = classSeparator(test, mus, covs, priors)
    ldavector = ldaClassifierVector(class1,class2)

    #scatterplot(test, ldavector)

    #scikitLDA(datum, labels, class1, class2)

    #ldaClassifier(ldavector,test, mus, covs, priors)

    implement_classifier_and_plots(test, mus, covs, priors)

    #imp(test, mus[0], mus[1],mus[2], covs[0], covs[1], covs[2])