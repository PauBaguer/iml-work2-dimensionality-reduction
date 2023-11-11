import numpy as np
import pandas as pd
class Pca:
    def __init__(self, original, dataset_name, k):
        self.original_values = original
        self.dataset_name = dataset_name
        self.k = k # The K largest eigenvalues will be selected.
        self.shape = self.original_values.shape
        self.mean_vector = self.d_dimensional_mean_vector()
        self.covariance_matrix = self.comp_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.comp_eigenvectors()
        self.k_eigenvectors_matrix = self.k_eigenvalues()
        self.data_adjust = self.comp_data_adjust()
        self.rotated_values = self.rotate_space()
        print()

    def d_dimensional_mean_vector(self):
        return self.original_values.mean(axis=0)

    def covariance(self, A, mean_A, B, mean_B):
        dist_A = A - mean_A
        dist_B = B - mean_B

        covar = np.sum(np.multiply(dist_A , dist_B)) / len(dist_A)
        return covar

    def comp_covariance_matrix(self):
        c_matrix = np.zeros((self.shape[1], self.shape[1]))

        for row_idx , _ in enumerate(c_matrix):
            for col_idx, _ in enumerate(c_matrix.T):
                co_var = self.covariance(self.original_values[:, row_idx], self.mean_vector[row_idx], self.original_values[:, col_idx], self.mean_vector[col_idx])
                c_matrix.itemset((row_idx, col_idx), co_var)

        print()
        print(f"== {self.dataset_name} COVARIANCE MATRIX ==")
        print(c_matrix)
        print()
        return c_matrix

    def comp_eigenvectors(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)

        eigenvalues = np.array([x for x in sorted(eigenvalues, reverse=True)])
        eigenvectors = np.array([x for _, x in sorted(zip(eigenvalues, eigenvectors.T), reverse=False)]).T
        eigen_df = pd.DataFrame(
            {'Eigen values': eigenvalues, 'Eigen vectors': [str(v) for v in eigenvectors]})
        # eigen_df.sort_values('Eigen values', ascending=False, inplace=True)

        pd.set_option('display.max_colwidth', 10000)
        print(eigen_df)
        return eigenvalues, eigenvectors

    def k_eigenvalues(self):
        k_eigenvectors = self.eigenvectors[0:self.k]
        return k_eigenvectors.T

    def comp_data_adjust(self):
        data_adjust_T = np.array([col - self.mean_vector[i] for i, col in enumerate(self.original_values.T)])
        return data_adjust_T.T

    def rotate_space(self):
        transformed = np.matmul(self.k_eigenvectors_matrix.T, self.data_adjust.T)
        return transformed