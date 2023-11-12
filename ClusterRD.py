from pca import Pca
from sklearn.decomposition import TruncatedSVD
from kmeans import kmeans
from birch import birch

class Cluster:
    def __init__(self, dataset_name, k, data, gs, reduction_method):
        self.dataset_name = dataset_name
        self.k = k
        self.data = data
        self.gs = gs
        
        if reduction_method == 'pca':
            self.reduced = Pca(data, dataset_name, k).rotated_values
        elif reduction_method == 'svd':
            self.reduced = TruncatedSVD(n_components=k, random_state=42).fit(data).transform(data)
        else:
            raise ValueError('Invalid reduction method')
        
    def clustering(self, clustering_method, c):
        if clustering_method == 'kmeans':
            clustering = kmeans(self.reduced, c)
        elif clustering_method == 'birch':
            clustering = birch(self.reduced, c)
        else:
            raise ValueError('Invalid clustering method')
        return clustering