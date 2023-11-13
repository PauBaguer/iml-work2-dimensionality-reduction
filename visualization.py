from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

"""
You need to visualize your original data sets, the result of the k-Means and BIRCH algorithms without the dimensionality
reduction, and the result of the k-Means and BIRCH algorithms with the dimensionality reduction.You need to visualize
your original data sets, the result of the k-Means and BIRCH algorithms without the dimensionality reduction, and the
result of the k-Means and BIRCH algorithms with the dimensionality reduction.
"""


class Visualization:

    def __init__(self, df, labels_gt, labels_birch_or, labels_kmeans_or):
        self.df = df
        self.labels_gt = labels_gt
        self.labels_birch_or = labels_birch_or
        self.labels_kmeans_or = labels_kmeans_or
        # self.labels_birch_pca = labels_birch_red_pca
        # self.labels_kmeans_pca = labels_kmeans_pca

    def func_pca(self, dataset, n_components=2):
        al = PCA(n_components=n_components)
        pca = al.fit_transform(self.df)
        labels = [self.labels_gt, self.labels_birch_or, self.labels_kmeans_or]
        i = 0
        # titles = ['Ground Truth', 'Clustering BIRCH w/o dim. reduction', 'Clustering K-Means w/o dim. reduction']
        titles = ['Ground Truth', 'Clustering BIRCH w/ PCA', 'Clustering K-Means w/ PCA']
        for label in labels:
            plt.figure(figsize=(8, 6))
            plt.scatter(pca[:, 0], pca[:, 1], c=label, cmap='tab20', s=20)
            plt.xlabel('Principal component 1')
            plt.ylabel('Principal component 2')
            plt.title(f'PCA with {titles[i]} in {dataset}')
            plt.savefig(f'figures/visualization/pca_{dataset}_{i}_reduced.png')
            plt.close()
            # plt.show()
            i += 1

    def func_isomap(self, dataset, n_neighbors=100, n_components=2):
        iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        isomap = iso.fit_transform(self.df)
        labels = [self.labels_gt, self.labels_birch_or, self.labels_kmeans_or]
        # titles = ['Ground Truth', 'Clustering BIRCH w/o dim. reduction', 'Clustering K-Means w/o dim. reduction',
        # 'Clustering BIRCH w/ PCA', 'Clustering K-Means w/ PCA']
        titles = ['Ground Truth', 'Clustering BIRCH w/ PCA', 'Clustering K-Means w/ PCA']
        i = 0
        for label in labels:
            plt.figure(figsize=(8, 6))
            plt.scatter(isomap[:, 0], isomap[:, 1], c=label, cmap='tab20', s=20)
            plt.title(f'ISOMAP with {titles[i]} in {dataset}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.savefig(f'figures/visualization/isomap_{dataset}_{i}_reduced.png')
            plt.close()
            # plt.show()
            i += 1

    def func_isomap_subsample(self, dataset, n_neighbors=100, sample_size=15000, n_components=2):
        sample_indices = np.random.choice(self.df.shape[0], sample_size, replace=False)
        sample_data = self.df[sample_indices, :]
        # sample_data_temp = self.df[:sample_size, :]
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        transformed_sample = isomap.fit_transform(sample_data) # we fit ISOMAP on the sample data
        transformed_full_dataset = isomap.transform(self.df)  # apply the learned transformation to the entire dataset
        labels = [self.labels_gt, self.labels_birch_or, self.labels_kmeans_or]
        # titles = ['Ground Truth', 'Clustering BIRCH w/o dim. reduction', 'Clustering K-Means w/o dim. reduction']
        titles = ['Ground Truth', 'Clustering BIRCH w/ PCA', 'Clustering K-Means w/ PCA']
        i = 0
        for label in labels:
            plt.figure(figsize=(8, 6))
            plt.scatter(transformed_full_dataset[:, 0], transformed_full_dataset[:, 1], c=label, cmap='tab20', s=20)
            plt.title(f'ISOMAP with {titles[i]} in {dataset}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.savefig(f'figures/visualization/isomap_subsampled_{dataset}_{i}_reduced.png')
            plt.close()
            # plt.show()
            i += 1


