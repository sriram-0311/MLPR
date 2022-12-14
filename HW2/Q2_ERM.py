import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.stats import multivariate_normal
from Utilities import *

def make_decisions(samplePath, sample_info, loss_matrix=[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]):
    samples = readFromFile(path=samplePath)
    choices  = []
    correct = []
    for i, row in samples.iterrows():
        distribution = int(row['True Class Label'])
        choice = np.argmin([risk(0,[row['x'],row['y'],row['z']],loss_matrix,sample_info), risk(1,[row['x'],row['y'],row['z']],loss_matrix,sample_info),
                            risk(2,[row['x'],row['y'],row['z']],loss_matrix,sample_info), risk(3,[row['x'],row['y'],row['z']],loss_matrix,sample_info)])
        if(choice==0):
            choices.append(1)
            choice = 1
        elif(choice==1):
            choices.append(2)
            choice = 2
        else:
            choices.append(3)
            choice = 3
        if(choice==distribution):
            correct.append(True)
        else:
            correct.append(False)
    samples['ERM Classification'] = choices
    samples['Correct']            = correct
    return samples

def risk(i , x , loss_matrix, sample_info):
    risk = 0
    for j, field in sample_info.iterrows():
        if(i==j):
            continue
        risk = risk + loss_matrix[i][j]*field['P']*multivariate_normal.pdf(x,field['mu'],field['cov'])
    return risk

if __name__=='__main__':
    sample_info = pd.DataFrame(columns=['P','mu','cov'])
    cp_1 = 0.3
    mu_1  = [2,2,1]
    cov_1 = [[0.5, 0,   1  ],[0,   2,   0  ],[0,   0.5, 0.5]]
    d = {'P':cp_1,'mu':mu_1,'cov':cov_1}
    sample_info = sample_info.append(d,ignore_index=True)
    cp_2 = 0.3
    mu_2  = [2,1,2]
    cov_2 = [[1,   0,   1  ],[0,   1,   0  ],[0,   0.5, 1  ]]
    d = {'P':cp_2,'mu':mu_2,'cov':cov_2}
    sample_info = sample_info.append(d,ignore_index=True)
    cp_3 = 0.4
    mu_3a  = [3,2,2]
    cov_3a = [[3,   0,   0  ],[0,   1,   0  ],[0,   0,   0.5]]
    d = {'P':(cp_3/2),'mu':mu_3a,'cov':cov_3a}
    sample_info = sample_info.append(d,ignore_index=True)
    mu_3b  = [1,2,1]
    cov_3b = [[1,   0,   1  ],[0,   1,   0  ],[0,   0,   2  ]]
    d = {'P':(cp_3/2),'mu':mu_3b,'cov':cov_3b}
    sample_info = sample_info.append(d,ignore_index=True)

    loss_matrix_10 = [[0,   1,   10  , 10],[1,   0,   10  , 10],[1,   1,   0   , 0 ],[1,   1,   0   , 0 ]]
    loss_matrix_100 = [[0,   1,  100  , 100],[1,   0,  100  , 100],[1,   1,  0    , 0  ],[1,   1,  0    , 0  ]]

    samplePath = './samplesa.csv'
    samples_b_10  = './samples10.csv'
    samples_b_100 = './samples100.csv'
    samples = randomSampling(mu_1, mu_2, mu_3a, mu_3b, cov_1, cov_2, cov_3a, cov_3b, cp_1, cp_2, cp_3)
    writeToFile(samples, samplePath)
    plot_samples(samplePath, path='samples_scatterplot_a.pdf')
    samples = make_decisions(samplePath, sample_info)
    writeToFile(samples, samplePath)
    plotClassifiedSamples(samplePath, path='samples_classified_scatterplot.pdf')
    plotCorrectClassifiedSamples(samplePath, path='samples_correct_classified_scatterplot.pdf')

    samples_10 = make_decisions(samplePath, sample_info, loss_matrix=loss_matrix_10)
    writeToFile(samples_10, samples_b_10)
    plotClassifiedSamples(samples_b_10, path='samples_classified_scatterplot_10.pdf')
    plotCorrectClassifiedSamples(samples_b_10, path='samples_correct_classified_scatterplot_10.pdf')
    samples_100 = make_decisions(samplePath, sample_info, loss_matrix=loss_matrix_100)
    writeToFile(samples_100, samples_b_100)
    plotClassifiedSamples(samples_b_100, path='samples_classified_scatterplot_100.pdf')
    plotCorrectClassifiedSamples(samples_b_100, path='samples_correct_classified_scatterplot_100.pdf')
    plot_decision_matrix(samplePath, path='./decision_matrix.pdf')
    plot_decision_matrix(samples_b_10, path='./decision_matrix_10.pdf')
    plot_decision_matrix(samples_b_100, path='./decision_matrix_100.pdf')