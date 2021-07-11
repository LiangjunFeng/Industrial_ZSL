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
from util import *
from scipy.linalg import cholesky, svd

warnings.filterwarnings("ignore")


def creat_dataset(test_index = [8, 12, 14]):
    path = './TE_mat_data/'
    print("loading data...")

    fault1 = loadmat(path + 'd01.mat')['data']
    fault2 = loadmat(path + 'd02.mat')['data']
    fault3 = loadmat(path + 'd03.mat')['data']
    fault4 = loadmat(path + 'd04.mat')['data']
    fault5 = loadmat(path + 'd05.mat')['data']
    fault6 = loadmat(path + 'd06.mat')['data']
    fault7 = loadmat(path + 'd07.mat')['data']
    fault8 = loadmat(path + 'd08.mat')['data']
    fault9 = loadmat(path + 'd09.mat')['data']
    fault10 = loadmat(path + 'd10.mat')['data']
    fault11 = loadmat(path + 'd11.mat')['data']
    fault12 = loadmat(path + 'd12.mat')['data']
    fault13 = loadmat(path + 'd13.mat')['data']
    fault14 = loadmat(path + 'd14.mat')['data']
    fault15 = loadmat(path + 'd15.mat')['data']

    attribute_matrix_ = pd.read_excel('./attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values

    train_index = list(set(np.arange(15)) - set(test_index))

    test_index.sort()
    train_index.sort()

    print("test classes: {}".format(test_index))
    print("train classes: {}".format(train_index))

    data_list = [fault1, fault2, fault3, fault4, fault5,
                 fault6, fault7, fault8, fault9, fault10,
                 fault11, fault12, fault13, fault14, fault15]

    trainlabel = []
    train_attributelabel = []
    traindata = []
    for item in train_index:
        trainlabel += [item] * 480
        train_attributelabel += [attribute_matrix[item, :]] * 480
        traindata.append(data_list[item])
    trainlabel = np.row_stack(trainlabel)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.column_stack(traindata).T

    testlabel = []
    test_attributelabel = []
    testdata = []
    for item in test_index:
        testlabel += [item] * 480
        test_attributelabel += [attribute_matrix[item, :]] * 480
        testdata.append(data_list[item])
    testlabel = np.row_stack(testlabel)
    test_attributelabel = np.row_stack(test_attributelabel)
    testdata = np.column_stack(testdata).T

    return traindata, trainlabel, train_attributelabel, \
           testdata, testlabel, test_attributelabel, \
           attribute_matrix_.iloc[test_index,:], attribute_matrix_.iloc[train_index, :]


def feature_extraction(traindata, testdata, train_attributelabel, test_attributelabel):
    trainfeatures = []
    testfeatures = []
    for i in range(train_attributelabel.shape[1]):
        spca = DSPCA(20)
        spca.fit(traindata, train_attributelabel[:, i])
        trainfeatures.append(spca.transform(traindata))
        testfeatures.append(spca.transform(testdata))
    return np.column_stack(trainfeatures), np.column_stack(testfeatures)


def pre_model(model, traindata, trainlabel, train_attributelabel, testdata, testlabel, test_attributelabel,
              attribute_matrix):
    model_dict = {'SVC_linear': SVC(kernel='linear', C=1), 'lr': LogisticRegression(), 'SVC_rbf': SVC(kernel='rbf'),
                  'rf': RandomForestClassifier(n_estimators=50), 'Ridge': Ridge(alpha=1), 'NB': GaussianNB(),
                  'Lasso': Lasso(alpha=0.1)}

    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix.index[loc] - 1)
    label_lis = np.mat(np.row_stack(label_lis))
    print(model)
    accuracy(label_lis, testlabel)
    return label_lis, testlabel


def pre_model_proba(model, traindata, trainlabel, train_attributelabel, testdata, testlabel, test_attributelabel,
                    attribute_matrix):
    model_dict = {'SVC_linear': SVC(kernel='linear', probability=True), 'lr': LogisticRegression(),
                  'SVC_rbf': SVC(kernel='rbf', probability=True),
                  'rf': RandomForestClassifier(n_estimators=10), 'Ridge': Ridge(alpha=1), 'NB': GaussianNB(),
                  'Lasso': Lasso(alpha=0.1)}

    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict_proba(testdata)
        else:
            res = np.ones((testdata.shape[0], 2))
            res[:, 1] = res[:, 1] * 0.001
            res[:, 0] = res[:, 0] * 0.999
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        res_list = [1] * attribute_matrix.shape[0]
        pre_res = np.ravel(test_pre_attribute[i, :])
        for j in range(attribute_matrix.shape[0]):
            for k in range(attribute_matrix.shape[1]):
                if attribute_matrix.iloc[j, k] == 0:
                    res_list[j] = res_list[j] * pre_res[k * 2]
                else:
                    res_list[j] = res_list[j] * pre_res[k * 2 + 1]
        loc = np.array(res_list).argmax()
        label_lis.append(attribute_matrix.index[loc] - 1)
    label_lis = np.mat(np.row_stack(label_lis))
    print(model)
    accuracy(label_lis, testlabel)
    return label_lis, testlabel


print("==========================[test classes][3 ,6, 9]===================================")
print("beginning...with feature extraction")
traindata, trainlabel, train_attributelabel, testdata, testlabel, \
test_attributelabel, attribute_matrix, train_attribute_matrix = creat_dataset([3, 6, 9])
print("SPCA extracting feature (takes lots of time)...")
traindata, testdata = feature_extraction(traindata,testdata,train_attributelabel,test_attributelabel)
print(traindata.shape, trainlabel.shape, train_attributelabel.shape)
print(testdata.shape, testlabel.shape, test_attributelabel.shape)
print("model training...")
pre_model('NB', traindata, trainlabel, train_attributelabel, testdata, testlabel,
          test_attributelabel, attribute_matrix)
pre_model('rf', traindata, trainlabel, train_attributelabel, testdata, testlabel,
          test_attributelabel, attribute_matrix)


print("==========================[test classes][0 ,5, 13]===================================")
print("beginning...with feature extraction")
traindata, trainlabel, train_attributelabel, testdata, testlabel, \
test_attributelabel, attribute_matrix, train_attribute_matrix = creat_dataset([0, 5, 13])
print("SPCA extracting feature (takes lots of time)...")
traindata, testdata = feature_extraction(traindata,testdata,train_attributelabel,test_attributelabel)
print(traindata.shape, trainlabel.shape, train_attributelabel.shape)
print(testdata.shape, testlabel.shape, test_attributelabel.shape)
print("model training...")
pre_model('NB', traindata, trainlabel, train_attributelabel, testdata, testlabel,
          test_attributelabel, attribute_matrix)
pre_model('rf', traindata, trainlabel, train_attributelabel, testdata, testlabel,
          test_attributelabel, attribute_matrix)





































