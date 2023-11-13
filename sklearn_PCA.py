from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from scipy.io import arff
import pandas as pd
import scipy as sc
from sklearn import metrics


class SklearnAlgorithms:

    def __init__(self, df):
        self.df = df

    def func_sklearn(self, algorithm, n_components, plot=False, dataset=None):
        al = None
        title = None
        if algorithm == 0:
            al = PCA(n_components=n_components)
            title = 'PCA'
        elif algorithm == 1:
            al = IncrementalPCA(n_components=n_components)
            title = 'IPCA'
        reduced_data = al.fit_transform(self.df)
        components = al.components_
        explained_variances = al.explained_variance_ratio_
        var = sum(al.explained_variance_ratio_)
        cov_matrix = al.get_covariance()
        if plot:
            plt.figure(figsize=(5, 4))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')
            plt.title(f'Reduced dataset in {title} space')
            plt.xlabel('Principal component 1')
            plt.ylabel('Principal component 2')
            plt.savefig(f'figures/sklearn_pca/{dataset}_{title}.png')
            plt.close()
        return al, components, var, cov_matrix, explained_variances, reduced_data

    def explore_ncomponents(self, n_components, plot=False, dataset=None, algorithm=0):
        vars = []
        control = True
        thresh = 0
        for component in n_components:
            al, components, var, cov_matrix, explained_variances,  reduced_data = self.func_sklearn(algorithm, component)
            vars.append(var)
            if var > 0.90 and control:
                thresh = component
                print(component)
                control = False
        title = ''
        if algorithm == 0:
            title = 'PCA'
        elif algorithm == 1:
            title = 'IPCA'
        if plot:
            plt.figure(figsize=(6, 4))
            plt.grid()
            plt.plot(n_components, vars, marker='o')
            plt.xlabel('n_components')
            plt.ylabel('Explained variance ratio')
            plt.title(f'n_components vs. Explained Variance Ratio with {title}')
            plt.savefig(f'figures/sklearn_pca/components_vs_exp_var_ratio_{title}_{dataset}.png')
            plt.close()
            # plt.show()
        return vars, thresh




