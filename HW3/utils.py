import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import random
from scipy.stats import multivariate_normal

def generate_data(mus, sigmas, priors, N, name):

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

    samples = pd.concat([samples_1a, samples_1b, samples_2], ignore_index=True)

    samples.to_csv("./HW3/"+name, index=False)
    return samples

if __name__ == "__main__":
    mus = np.array([[5,0],[0,4],[3,2]])
    sigmas = np.array([[[4,0],[0,2]],[[1,0],[0,3]],[[2,0],[0,2]]])
    priors = np.array([0.3,0.3,0.4])
    #Generate training data and Validation data and save them to csv files
    Train_100samples = generate_data(mus, sigmas, priors, 100, "Train_100samples.csv")
    Train_1000samples = generate_data(mus, sigmas, priors, 1000, "Train_1000samples.csv")
    Train_10000samples = generate_data(mus, sigmas, priors, 10000, "Train_10000samples.csv")
    Validation_20Ksamples = generate_data(mus, sigmas, priors, 20000, "Validation_20Ksamples.csv")
    Train_100samples.plot.scatter(x='x', y='y', c='True Class Label', colormap='viridis')

    #plot the data and save figures to png files
    colors = ['red','blue']
    figure, ax = plt.subplots(2, 2, figsize=(7, 7))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    scatter = ax[0,0].scatter(Train_100samples['x'], Train_100samples['y'], c=Train_100samples['True Class Label'], cmap='viridis',marker='.', s=7)
    ax[0,0].set_title('Train_100samples')
    legend1 = ax[0,0].legend(*scatter.legend_elements(),loc="upper right", title="Classes")
    ax[0,0].add_artist(legend1)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].grid(color='lightgray', linestyle='-', linewidth=0.2)
    ax[0,0].legend()
    scatter = ax[0,1].scatter(Train_1000samples['x'], Train_1000samples['y'], c=Train_1000samples['True Class Label'], cmap='viridis', marker='.', s=5)
    ax[0,1].set_title('Train_1000samples')
    legend2 = ax[0,1].legend(*scatter.legend_elements(),loc="upper right", title="Classes")
    ax[0,1].add_artist(legend2)
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    ax[0,1].grid(color='lightgray')
    ax[0,1].legend(labelcolor='white', )
    scatter = ax[1,0].scatter(Train_10000samples['x'], Train_10000samples['y'], c=Train_10000samples['True Class Label'], cmap='viridis', marker='.', s=2)
    ax[1,0].set_title('Train_10000samples')
    legend3 = ax[1,0].legend(*scatter.legend_elements(),loc="upper right", title="Classes")
    ax[1,0].add_artist(legend3)
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].grid(color='lightgray')
    ax[1,0].legend()
    ax[1,1].scatter(Validation_20Ksamples['x'], Validation_20Ksamples['y'], c=Validation_20Ksamples['True Class Label'], cmap='viridis', marker='.', s=0.3)
    ax[1,1].set_title('Validation_20Ksamples')
    legend4 = ax[1,1].legend(*scatter.legend_elements(),loc="upper right", title="Classes")
    ax[1,1].add_artist(legend4)
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].grid(color='lightgray')
    ax[1,1].legend()
    plt.savefig("./HW3/Generated Data.png", dpi=500)