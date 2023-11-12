from pca import Pca
from kmeans import kmeans
from birch import birch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score


class Cluster:
    def __init__(self, dataset_name, data, gs):
        self.dataset_name = dataset_name
        self.data = data
        self.gs = gs
    
    def plot_explained_variance(self):
        eigenvalues = Pca(self.data, self.dataset_name, min(self.data.shape)).eigenvalues
        explained_variance = eigenvalues/sum(eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variance)
        plt.plot(range(self.data.shape[1]), cumulative_explained_variance, marker='x')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title(f'Explained variance with {self.dataset_name} dataset')
        plt.show()
            
    def clustering(self, clustering_method, c):
        if clustering_method == 'kmeans':
            clustering = kmeans(self.reduced, c)
        elif clustering_method == 'birch':
            clustering = birch(self.reduced, c)
        else:
            raise ValueError('Invalid clustering method')
        return clustering