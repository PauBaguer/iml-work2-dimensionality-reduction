import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pca import Pca
from kmeans import kmeans
from birch import birch
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score


class Cluster:
    def __init__(self, dataset_name, k, data, gs, reduction_method):
        self.dataset_name = dataset_name
        self.k = k
        self.data = data
        self.gs = gs
        self.reduction_method = reduction_method
        if self.reduction_method == 'PCA':
            self.transformed_dataset = Pca(self.data, self.dataset_name, min(self.data.shape)).reduced_original_values
        elif self.reduction_method == 'truncatedSVD':
            self.transformed_dataset = TruncatedSVD(n_components=min(self.data.shape)-1).fit_transform(self.data)

    def plot_total_explained_variance(self):
        if self.reduction_method == 'PCA':
            eigenvalues = Pca(self.data, self.dataset_name, min(self.data.shape)).eigenvalues
            explained_variance = eigenvalues/sum(eigenvalues)
            vars = np.cumsum(explained_variance)
            fig = plt.figure()
            plt.grid()
            plt.plot(range(self.data.shape[1]), vars, marker='x')
            plt.xlabel('Number of Components')
            plt.ylabel('Explained Variance')
            plt.title(f'Explained variance with {self.dataset_name} dataset')
            fig.savefig("figures/"+self.reduction_method+'/'+self.dataset_name+'-variance.png')
            
        elif self.reduction_method == 'truncatedSVD':
            vars = []
            num_attributes = self.data.shape[1]
            num_components = list(range(2, num_attributes))

            for n in num_components:
                svd = TruncatedSVD(n_components=n)
                svd.fit(self.data)
                var = svd.explained_variance_ratio_.sum()
                vars.append(var)
            algorithm = 'Truncated SVD'
            fig = plt.figure()
            plt.grid()
            plt.plot(num_components, vars, marker='x')
            plt.xlabel('Number of Components')
            plt.ylabel('Total explained variance ratio')
            plt.title(f'{algorithm} explained variance with {self.dataset_name} dataset')
            # plt.show()
            fig.savefig("figures/"+self.reduction_method+'/'+self.dataset_name+'-variance.png')

            return vars

    def plot_clustering(self, n_min=2, n_max=3, range_k=1, c_algorithm='kmeans'):
        n_clusters = []
        n_components = []

        k_min = max(self.k - range_k, 2)
        k_max = self.k + range_k + 1

        db_scores = [[] for _ in range(k_max - k_min)]
        sil_scores = [[] for _ in range(k_max - k_min)]

        for n in range(n_min, n_max + 1):
            n_components.append(n)
            for k in range(k_min, k_max):
                n_clusters.append(k)
                labels = []
                if c_algorithm == 'kmeans':
                    centroid, labels = kmeans(self.transformed_dataset, k, metric='l2', centroid_init='kmeans++')
                elif c_algorithm == 'birch':
                    labels, brc = birch(self.transformed_dataset, 0.5, k)
                else:
                    print('Invalid clustering algorithm')

                db = davies_bouldin_score(self.transformed_dataset, labels)
                sil = silhouette_score(self.transformed_dataset, labels)
                db_scores[k - k_min].append(db)
                sil_scores[k - k_min].append(sil)

        results_df = pd.DataFrame({"n_components": n_components})

        cmap = plt.cm.get_cmap('hsv', k_max - k_min)
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()

        for k in range(k_min, k_max):
            results_df["SC_" + str(k)] = np.array(sil_scores[k - k_min])
            results_df["DB_" + str(k)] = np.array(db_scores[k - k_min])

            ax.plot(results_df["n_components"], results_df["SC_" + str(k)], color=cmap(k-k_min), label="K=" + str(k))
            ax2.plot(results_df["n_components"], results_df["DB_" + str(k)], color=cmap(k-k_min), label="K=" + str(k))

        ax.legend()
        ax.set_title(f"K-Means Silhouette, {self.dataset_name} dataset")
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Silhouette')
        fig.savefig("figures/"+self.reduction_method+'/'+self.dataset_name+'-kmeans-silhouette.png')
        fig.show()

        ax2.legend()
        ax2.set_title(f"K-Means Davies-Bouldin, {self.dataset_name} dataset")
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('Davies-Bouldin')
        fig2.savefig("figures/"+self.reduction_method+'/'+self.dataset_name+'-kmeans-db.png')
        fig2.show()
        return

    def plot_external_index(self, n_min, n_max,  c_algorithm='kmeans', external_index='Purity'):
        scores = []
        num_components = list(range(n_min, n_max + 1))

        for n in num_components:
            transformed_dataset = self.transformed_dataset[:, :n]

            labels = []
            if c_algorithm == 'kmeans':
                centroid, labels = kmeans(transformed_dataset, self.k, metric='l2', centroid_init='kmeans++')
            elif c_algorithm == 'birch':
                labels, brc = birch(transformed_dataset, 0.5, self.k)
            else:
                print('Invalid clustering algorithm')

            score = None
            if external_index == 'Purity':
                score = homogeneity_score(self.gs, labels)
            elif external_index == 'ARI':
                score = adjusted_rand_score(self.gs, labels)
            elif external_index == 'NMI':
                score = normalized_mutual_info_score(self.gs, labels)
            scores.append(score)

        algorithm = 'Truncated SVD'
        fig = plt.figure()
        plt.grid()
        plt.plot(num_components, scores, marker='x')
        plt.xlabel('Number of Components')
        plt.ylabel(f'{external_index}')
        plt.title(f'{algorithm} external index results ({self.dataset_name} dataset, K={self.k})')
        # plt.show()
        
        fig.savefig(f"figures/{self.reduction_method}/{self.dataset_name}-{external_index}.png")

        return





