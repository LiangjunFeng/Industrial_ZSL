from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings
from scipy.linalg import cholesky, svd


def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


class DSPCA():
    def __init__(self, num, sigma='auto'):
        self._num = num
        self._sigma = sigma

    def kernel_matrix(self, label):
        if self._sigma == 'auto':
            self._sigma = 1 / np.mat(label).T.shape[1]
        mat = np.zeros((label.shape[0], label.shape[0]))
        for i in range(label.shape[0]):
            mat[i] = np.linalg.norm(np.mat(label[i] - label), axis=0)
        mat = np.exp(-1 * np.multiply(mat, mat) / (self._sigma ** 2))
        return mat

    def fit(self, data, label):
        n = data.shape[0]
        L = self.kernel_matrix(label)
        deta = cholesky(L + 1e-5 * np.eye(L.shape[0])).T
        H = np.eye(n) - (1 / n) * np.ones((n, n)) * np.ones((n, n)).T
        fai = data.T.dot(H).dot(deta)
        U, S, V = svd(fai, full_matrices=False)
        U, V = svd_flip(U, V)
        V = V[0:self._num].T
        xi = np.diag(S[0:self._num])
        self._U = fai.dot(V).dot(np.mat(xi).I)

    def transform(self, data):
        return (self._U.T.dot(data.T)).T

def accuracy(res, label):
    print('accuracy: ', accuracy_score(res, label))