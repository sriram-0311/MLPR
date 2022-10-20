import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.patches as mpatches

def randomSampling(mu_1, mu_2, mu_3a, mu_3b, cov_1, cov_2, cov_3a, cov_3b, cp_1, cp_2, cp_3):
    rng = default_rng()
    overall_size = 10000
    size_1 = 0
    size_2 = 0
    size_3a = 0
    size_3b = 0
    for i in range(0, overall_size) :
        r = random.random()
        if(r < cp_1):
            size_1 = size_1 + 1
        elif(r < cp_1+cp_2):
            size_2 = size_2 + 1
        elif(r < cp_1+cp_2+(cp_3/2)):
            size_3a = size_3a + 1
        else:
            size_3b = size_3b + 1
    Class1 = rng.multivariate_normal(mean=mu_1, cov=cov_1, size=size_1)
    Class1 = pd.DataFrame(Class1, columns=['x','y','z'])
    Class1['True Class Label'] = 1
    Class2 = rng.multivariate_normal(mean=mu_2, cov=cov_2, size=size_2)
    Class2 = pd.DataFrame(Class2, columns=['x','y','z'])
    Class2['True Class Label'] = 2
    Class3a = rng.multivariate_normal(mean=mu_3a, cov=cov_3a, size=size_3a)
    Class3a = pd.DataFrame(Class3a, columns=['x','y','z'])
    Class3a['True Class Label'] = 3
    Class3b = rng.multivariate_normal(mean=mu_3b, cov=cov_3b, size=size_3b)
    Class3b = pd.DataFrame(Class3a, columns=['x','y','z'])
    Class3b['True Class Label'] = 3
    samples   = Class1.append([Class2, Class3a, Class3b])
    return samples

def writeToFile(samples, path):
    samples.to_csv(path)

def readFromFile(path):
    samples = pd.read_csv(path, index_col=0)
    return samples

def plot_samples(samples_path, path='samples_scatterplot.pdf'):
    samples = readFromFile(path=samples_path)
    fig = plt.figure(figsize = (5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    Class1 = samples[samples['True Class Label']==1]
    Class2 = samples[samples['True Class Label']==2]
    Class3 = samples[samples['True Class Label']==3]
    x_1 = Class1['x'].tolist()
    y_1 = Class1['y'].tolist()
    z_1 = Class1['z'].tolist()
    x_2 = Class2['x'].tolist()
    y_2 = Class2['y'].tolist()
    z_2 = Class2['z'].tolist()
    x_3 = Class3['x'].tolist()
    y_3 = Class3['y'].tolist()
    z_3 = Class3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', color='blue',alpha=0.2)
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', color='red',alpha=0.2)
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', color='green',alpha=0.2)
    #ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    ax.legend(loc='upper left', title='Class Label')
    plt.savefig(path)
    plt.clf()
    return None

def plot_decision_matrix(samplePath, path='./decision_matrix.pdf'):
    samples = readFromFile(path=samplePath)
    pred = samples['ERM Classification'].tolist()
    act  = samples['True Class Label'].tolist()
    confusion = confusion_matrix(act, pred, normalize='true')
    print(confusion)
    sns.heatmap(data=confusion,cmap="YlOrRd",annot=True,)
    plt.xlabel('Decision')
    plt.ylabel('True Class Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()

def plotClassifiedSamples(samplePath, path='samples_scatterplot.pdf'):
    samples = readFromFile(path=samplePath)
    fig = plt.figure(figsize = (5, 5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    Class1 = samples[samples['ERM Classification']==1]
    Class2 = samples[samples['ERM Classification']==2]
    Class3 = samples[samples['ERM Classification']==3]
    x_1 = Class1['x'].tolist()
    y_1 = Class1['y'].tolist()
    z_1 = Class1['z'].tolist()
    x_2 = Class2['x'].tolist()
    y_2 = Class2['y'].tolist()
    z_2 = Class2['z'].tolist()
    x_3 = Class3['x'].tolist()
    y_3 = Class3['y'].tolist()
    z_3 = Class3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', color='blue',alpha=0.2)
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', color='red',alpha=0.2)
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', color='green',alpha=0.2)
    #ax.set_title("Samples from Multivariate Gaussian Distributions")
    ax.set_xlabel('1st Dimension, x')
    ax.set_ylabel('2nd Dimension, y')
    ax.set_zlabel('3rd Dimension, z')
    ax.legend(loc='upper left', title='Class Label')
    plt.savefig(path)
    plt.clf()
    return None

def plotCorrectClassifiedSamples(samplePath, path='samples_classified_scatterplot.pdf'):
    samples = readFromFile(path=samplePath)
    fig = plt.figure(figsize = (5,5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.99, bottom=0.01, wspace=0)
    ax = plt.axes(projection ="3d")
    # Plot correct
    correct = samples[samples['Correct']==True]
    Class1 = correct[correct['True Class Label']==1]
    Class2 = correct[correct['True Class Label']==2]
    Class3 = correct[correct['True Class Label']==3]
    x_1 = Class1['x'].tolist()
    y_1 = Class1['y'].tolist()
    z_1 = Class1['z'].tolist()
    x_2 = Class2['x'].tolist()
    y_2 = Class2['y'].tolist()
    z_2 = Class2['z'].tolist()
    x_3 = Class3['x'].tolist()
    y_3 = Class3['y'].tolist()
    z_3 = Class3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2, color='green')
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2, color='green')
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2, color='green')
    # Plot incorrect
    correct = samples[samples['Correct']==False]
    Class1 = correct[correct['True Class Label']==1]
    Class2 = correct[correct['True Class Label']==2]
    Class3 = correct[correct['True Class Label']==3]
    x_1 = Class1['x'].tolist()
    y_1 = Class1['y'].tolist()
    z_1 = Class1['z'].tolist()
    x_2 = Class2['x'].tolist()
    y_2 = Class2['y'].tolist()
    z_2 = Class2['z'].tolist()
    x_3 = Class3['x'].tolist()
    y_3 = Class3['y'].tolist()
    z_3 = Class3['z'].tolist()
    ax.scatter3D(x_1, y_1, z_1, label='1', marker='o', alpha=0.2, color='red')
    ax.scatter3D(x_2, y_2, z_2, label='2', marker='^', alpha=0.2, color='red')
    ax.scatter3D(x_3, y_3, z_3, label='3', marker='s', alpha=0.2, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.get_legend().remove()
    green_patch = mpatches.Patch(color='green', label='Correct')
    red_patch = mpatches.Patch(color='red', label='Incorrect')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', title='Classification')
    plt.savefig(path)
    plt.clf()
    return None