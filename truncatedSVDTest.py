import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from kmeans import kmeans
from birch import birch
from sklearn.metrics import davies_bouldin_score, silhouette_score

class Cluster:
    def __init__(self, dataset_name, k, data, gs):
        self.dataset_name = dataset_name
        self.k = k
        self.data = data
        self.gs = gs

    def plot_total_explained_variance(self, plot=False):
        vars = []
        num_attributes = self.data.shape[1]
        num_components = list(range(2, num_attributes))

        for n in num_components:
            svd = TruncatedSVD(n_components=n)
            svd.fit(self.data)
            var = svd.explained_variance_ratio_.sum()
            vars.append(var)
        algorithm = 'Truncated SVD'
        if plot:
            fig = plt.figure()
            plt.grid()
            plt.plot(num_components, vars, marker='x')
            plt.xlabel('Number of Components')
            plt.ylabel('Total explained variance ratio')
            plt.title(f'{algorithm} explained variance with {self.dataset_name} dataset')
            # plt.show()
            fig.savefig(f"figures/truncatedSVD/{self.dataset_name}-variance.png")

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
            transformed_dataset = TruncatedSVD(n_components=n).fit_transform(self.data)
            for k in range(k_min, k_max):
                n_clusters.append(k)
                labels = []
                if c_algorithm == 'kmeans':
                    centroid, labels = kmeans(transformed_dataset, k, metric='l2', centroid_init='kmeans++')
                elif c_algorithm == 'birch':
                    labels, brc = birch(transformed_dataset, 0.5, k)
                else:
                    print('Invalid clustering algorithm')

                db = davies_bouldin_score(transformed_dataset, labels)
                sil = silhouette_score(transformed_dataset, labels)
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
        fig.savefig(f"figures/truncatedSVD/{self.dataset_name}-kmeans-silhouette.png")
        fig.show()

        ax2.legend()
        ax2.set_title(f"K-Means Davies-Bouldin, {self.dataset_name} dataset")
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('Davies-Bouldin')
        fig2.savefig(f"figures/truncatedSVD/{self.dataset_name}-kmeans-db.png")
        fig2.show()
        return





