import numpy as np
import warnings
from scipy.linalg import cholesky, svd

warnings.filterwarnings("ignore")


class SPCA():
    def __init__(self, num, sigma='auto'):
        self._num = num
        self._sigma = sigma

    def RBF_kernel(self, sample1, sample2):
        return np.exp(-np.linalg.norm(sample1 - sample2) ** 2 / (self._sigma ** 2))

    def kernel_matrix(self, label):
        if self._sigma == 'auto':
            self._sigma = 1 / np.mat(label).T.shape[1]
        mat = np.zeros((label.shape[0], label.shape[0]))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i > j:
                    mat[i, j] = self.RBF_kernel(label[i], label[j])
                    mat[j, i] = self.RBF_kernel(label[i], label[j])
                elif i == j:
                    mat[i, j] = self.RBF_kernel(label[i], label[j])
        return mat

    def fit(self, data, label):
        n = data.shape[0]
        H = np.eye(n) - (1 / n) * np.ones((n, n)) * np.ones((n, n)).T
        L = self.kernel_matrix(label)
        Q = data.T.dot(H.T).dot(L).dot(H).dot(data)
        e_vals, e_vecs = np.linalg.eig(Q)
        self._U = e_vecs[:, 0:self._num]

    def transform(self, data):
        return (self._U.T.dot(data.T)).T























