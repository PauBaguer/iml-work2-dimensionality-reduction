import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Pca:
    def __init__(self, original, dataset_name, k):
        self.original_values = original
        self.dataset_name = dataset_name
        self.k = k # The K largest eigenvalues will be selected. If K=-1, eigenvalues > 1 will be selected
        self.shape = self.original_values.shape
        self.mean_vector = self.d_dimensional_mean_vector()
        self.covariance_matrix = self.comp_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.comp_eigenvectors()
        self.k_eigenvectors_matrix = self.k_eigenvalues()
        self.data_adjust = self.comp_data_adjust()
        self.rotated_values = self.rotate_space()
        self.reduced_data_adjust = self.rotate_back_space()
        self.reduced_original_values = self.return_original_values()
        self.plots()
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
        # If k = -1, take only eigenvectors where eigenvalues are > 1.
        if self.k == -1:
            k_eigenvectors = np.array([ self.eigenvectors[i] for i, v in enumerate(self.eigenvalues) if v >= 1 ])
        else:
            k_eigenvectors = self.eigenvectors[0:self.k]
        return k_eigenvectors.T

    def comp_data_adjust(self):
        data_adjust_T = np.array([col - self.mean_vector[i] for i, col in enumerate(self.original_values.T)])
        return data_adjust_T.T

    def rotate_space(self):
        transformed = np.matmul(self.k_eigenvectors_matrix.T, self.data_adjust.T)
        return transformed.T

    def rotate_back_space(self):
        inverse_transformed = np.matmul(self.k_eigenvectors_matrix, self.rotated_values.T)
        return inverse_transformed.T

    def return_original_values(self):
        orig_T = np.array([col + self.mean_vector[i] for i, col in enumerate(self.reduced_data_adjust.T)])
        return orig_T.T

    def plots(self):
        
        fig, ax = plt.subplots(2,2, figsize=plt.figaspect(0.5))#, sharex=True, sharey=True)
        fig.subplots_adjust(bottom=.05,top=.95,left=.06,right=.98, wspace=0.1, hspace=0.4)


        ax[0][0].scatter(self.data_adjust[:, 0], self.data_adjust[:, 1])
        # ax[0][0].set_xlim([-2, 2])
        # ax[0][0].set_ylim([-2, 2])
        ax[0][0].axhline(y=0, color='k')
        ax[0][0].axvline(x=0, color='k')
        # ax[0][0].axis('square')
        ax[0][0].set_title(f"{self.dataset_name} Mean data adjust")


        zeros = np.zeros(len(self.original_values.T[0])).T
        yaxis = self.rotated_values[:, 1] if len(self.rotated_values.T) > 1 else zeros
        ax[0][1].scatter(self.rotated_values[:, 0], yaxis)#self.rotated_values[:, 1])
        # ax[0][1].set_xlim([-2, 2])
        # ax[0][1].set_ylim([-2, 2])
        ax[0][1].axhline(y=0, color='k')
        ax[0][1].axvline(x=0, color='k')
        # ax[0][1].axis('square')
        ax[0][1].set_title(f"{self.dataset_name} Transformed data")

        ax[1][0].scatter(self.reduced_data_adjust[:, 0], self.reduced_data_adjust[:, 1])
        ax[1][0].axhline(y=0, color='k')
        ax[1][0].axvline(x=0, color='k')
        # ax[1][0].axis('square')
        ax[1][0].set_title(f"{self.dataset_name} Mean data adjust w/ {len(self.k_eigenvectors_matrix.T)} dimensions")

        ax[1][1].scatter(self.reduced_original_values[:, 0], self.reduced_original_values[:, 1])
        ax[1][1].axhline(y=0, color='k')
        ax[1][1].axvline(x=0, color='k')
        # ax[1][1].axis('square')
        ax[1][1].set_title(f"{self.dataset_name} Original values w/ {len(self.k_eigenvectors_matrix.T)} dimensions")
        
        fig.savefig(f"figures/pca/{self.dataset_name}")
        fig.show()