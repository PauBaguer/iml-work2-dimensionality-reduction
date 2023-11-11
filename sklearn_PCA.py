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

    def func_sklearn(self, algorithm, n_components):
        al = None
        if algorithm == 0:
            al = PCA(n_components=n_components)
        elif algorithm == 1:
            al = IncrementalPCA(n_components=n_components)
        al.fit_transform(self.df)
        components = al.components_
        explained_variances = al.explained_variance_ratio_
        var = sum(al.explained_variance_ratio_)
        cov_matrix = al.get_covariance()
        return al, components, var, cov_matrix, explained_variances

    def explore_ncomponents(self, n_components, plot=False, algorithm=0):
        vars = []
        for component in n_components:
            al, components, var, cov_matrix, explained_variances = self.func_sklearn(algorithm, component)
            vars.append(var)
        title = ''
        if algorithm == 0:
            title = 'PCA'
        elif algorithm == 1:
            title = 'IPCA'
        if plot:
            plt.figure(figsize=(6, 3))
            plt.grid()
            plt.plot(n_components, vars, marker='o')
            plt.xlabel('n_components')
            plt.ylabel('Explained variance ratio')
            plt.title(f'n_components vs. Explained Variance Ratio with {title}')
            plt.show()
        return vars




