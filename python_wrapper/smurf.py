# coding: utf-8
"""
smuRF: simple multithreaded Random Forest

Version: 1.0
Authors: Christoph Loschen
"""
import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary('/home/loschen/calc/smuRF/src/python_interface/libsmuRF.so')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator


def ctypes2numpy(cptr, length, dtype):
    """Convert a ctypes pointer array to a numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        raise RuntimeError('expected float pointer')
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res


class DF(object):
    def __init__(self, X, label=None):
        self.ptr_ = ctypes.c_void_p()
        # self.handle = lib.DF_new()
        # lib.DF_setParameters(self.obj)
        X = X.astype(np.float32)
        # print type(self.ptr_)
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            # attach label to X
            if isinstance(label, np.ndarray) and len(label.shape) == 1:
                label = label.astype(np.float32)
                X = np.append(X, label.reshape(label.shape[0], -1), axis=1)
            lib.DF_createFromNumpy(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.shape[0], X.shape[1],
                                   ctypes.byref(self.ptr_))
        else:
            print X.shape
            print "ERROR: Need numpy matrix...!"
            sys.exit(1)

    def printSummary(self):
        lib.DF_printSummary(self.ptr_)

    def __del__(self):
        lib.DF_Free(self.ptr_)


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=500, mtry=2, node_size=5, max_depth=30, n_jobs=1, verbose_level=0):
        self.obj = lib.RandomForest_new()
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.node_size = node_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose_level = verbose_level
        lib.RandomForest_setParameters(self.obj, ctypes.c_int(n_estimators), ctypes.c_int(mtry),
                                       ctypes.c_int(node_size), ctypes.c_int(max_depth), ctypes.c_int(n_jobs),
                                       ctypes.c_int(verbose_level))


    def printInfo(self):
        lib.RandomForest_printInfo(self.obj)

    def setDataFrame(self, df):
        lib.RandomForest_setDataFrame(self.obj, df.ptr_)

    def train(self):
        lib.RandomForest_train(self.obj)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values.astype(np.float32)

        X = X.astype(np.float32)
        y = y.astype(np.float32)
        df = DF(X, label=y)
        self.setDataFrame(df)
        lib.RandomForest_train(self.obj)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        df = DF(X)
        length = ctypes.c_ulong()
        preds = lib.RandomForest_predict.restype = ctypes.py_object
        preds = lib.RandomForest_predict(self.obj, df.ptr_)
        return np.asarray(preds)

        # def get_params(self,deep):

    #	params = super(RandomForest, self).get_params(deep=deep)
    #       return params

    def set_params(self,**params):
        for parameter,value in params.items():
            setattr(self,parameter,value)
        lib.RandomForest_setParameters(self.obj, ctypes.c_int(self.n_estimators), ctypes.c_int(self.mtry),
                                       ctypes.c_int(self.node_size), ctypes.c_int(self.max_depth), ctypes.c_int(self.n_jobs),
                                       ctypes.c_int(self.verbose_level))
        return self

if __name__ == "__main__":
    # X = pd.read_csv('../data/katritzky_n_small.csv',sep=',')
    X = pd.read_csv('../data/mp_cdk.csv', sep=',')
    X = X._get_numeric_data()
    print X.describe()
    y = X['Ave °C']
    # y = X['n_exp']
    X.drop(['train', 'Ave °C'], axis=1, inplace=True)
    # X.drop(['train','n_exp'],axis=1,inplace=True)
    print X.describe()
    print y.describe()
    model = RandomForest(n_estimators=100, mtry=5, node_size=5, max_depth=30, n_jobs=4)
    # rf.setDataFrame(df)
    # rf.printInfo()
    model.fit(X, y)
    # model.printInfo()
    y_pred = model.predict(X)
    # print y_pred
    plt.scatter(y, y_pred)
    plt.show()
