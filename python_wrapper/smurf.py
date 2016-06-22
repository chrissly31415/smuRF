# coding: utf-8
"""
smuRF: simple multithreaded Random Forest

Version: 1.0
Authors: Christoph Loschen
"""
import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary('/home/loschen/calc/smuRF/src/python_interface/libsmuRF.so')
#lib = ctypes.cdll.LoadLibrary('/home/loschen/calc/smuRF/src/noopenmp/libsmuRF.so')
#lib = ctypes.cdll.LoadLibrary('/home/loschen/calc/smuRF/bin/libsmuRF.so')

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
        # Copying changes array style!!
        X = X.copy()
        # We need array as row-major FORTRAN STYLE
        X = X.astype(np.float32, order='F')
        # print type(self.ptr_)
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            # attach label to X
            if isinstance(label, np.ndarray) and len(label.shape) == 1:
                label = label.copy()
                label = label.astype(np.float32, order='F')
                X = np.append(X, label.reshape(label.shape[0], -1), axis=1)

            if not np.isfortran(X):
                raise RuntimeError('ERROR: Need fortran-style array!')
            lib.DF_createFromNumpy(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.shape[0], X.shape[1],
                                   ctypes.byref(self.ptr_))
        else:
            print X.shape
            print X.flags
            print "ERROR: Need numpy matrix...!"
            sys.exit(1)

    def copy(self):
        print "JUHU copy!"
        df_ptr_ = ctypes.c_void_p()
        lib.DF_copy(self.ptr_,df_ptr_)

    def printSummary(self):
        lib.DF_printSummary(self.ptr_)

    def __del__(self):
        #print "skip DF_Free"
        lib.DF_Free(self.ptr_)
        #pass


class RandomForest(BaseEstimator):
    """
    Python constructor
    .cpp constructor called in fit method
    (otherwise we get problems when running parallel)
    """
    def __init__(self, n_estimators=500, mtry=2, node_size=5, max_depth=30, n_jobs=1, verbose_level=1,regression=True):
        #print self.obj
        #self.obj = lib.RandomForest_new()
        self.obj = None
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.node_size = node_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose_level = verbose_level
        self.regression = regression
        #lib.RandomForest_setParameters(self.obj, ctypes.c_int(n_estimators), ctypes.c_int(mtry),
        #                               ctypes.c_int(node_size), ctypes.c_int(max_depth), ctypes.c_int(n_jobs),
        #                               ctypes.c_int(verbose_level),ctypes.c_bool(regression))
        #lib.RandomForest_printInfo()

    def printInfo(self):
        lib.RandomForest_printInfo(self.obj)

    def setDataFrame(self, df):
        lib.RandomForest_setDataFrame(self.obj, df.ptr_)

    def get_cppmodel(self):
        if self.obj is None:
            raise RuntimeError('Estimator need to be fitted first!')
        return self.obj


    #def train(self):
    #    lib.RandomForest_train(self.obj)

    def fit(self, lX, ly):
        if isinstance(lX, pd.DataFrame):
            lX = lX.values
        if isinstance(ly, pd.Series):
            ly = ly.values

        #create DF
        df = DF(lX, label=ly)
        #create RF
        self.obj = lib.RandomForest_new()
        lib.RandomForest_setParameters(self.obj, ctypes.c_int(self.n_estimators), ctypes.c_int(self.mtry),
                                       ctypes.c_int(self.node_size), ctypes.c_int(self.max_depth), ctypes.c_int(self.n_jobs),
                                       ctypes.c_int(self.verbose_level),ctypes.c_bool(self.regression))

        lib.RandomForest_train(self.obj,df.ptr_)

    def predict(self, lX):
        #print "python predict...not thread safe or clone does not work...?????"
        #print "Re-implement clone with all!!! or print address of RF/DF and c++ level"
        #print "No output!?"
        #print "get_params:",super(RandomForest,self).get_params(True)
        #print "http://scikit-learn.org/stable/faq.html!"
        if isinstance(lX, pd.DataFrame):
            lX = lX.values
        df = DF(lX)
        #self.setDataFrame(df)
        #length = ctypes.c_ulong()
        lib.RandomForest_predict.restype = ctypes.py_object
        #preds = lib.RandomForest_predict.restype = ctypes.py_object
        preds = lib.RandomForest_predict(self.obj, df.ptr_)
        #preds = np.ones((lX.shape[0],1))
        preds = np.asarray(preds)
        preds = preds.reshape((preds.shape[0],1))
        #print "python predict end..."
        #print preds
        return preds

    def predict_proba(self, lX):
        raise RuntimeError('not implemented yet!')

    def get_params(self, deep=True):
        return super(RandomForest,self).get_params(deep)

    def set_params(self,**params):
        for parameter,value in params.items():
            setattr(self,parameter,value)
        return self

if __name__ == "__main__":
    X = pd.read_csv('../data/katritzky_n_small.csv',sep=',')
    #X = X.iloc[:20,-5:]
    #X = pd.read_csv('../data/mp_cdk.csv', sep=',')
    print X.head(20)

    X = X._get_numeric_data()
    #y = X['Ave °C']
    y = X['n_exp']
    #X.drop(['train', 'Ave °C'], axis=1, inplace=True)
    #X.drop(['train','n_exp'],axis=1,inplace=True)
    model = RandomForest(n_estimators=100, mtry=5, node_size=1, max_depth=30, n_jobs=2)
    # rf.setDataFrame(df)
    # rf.printInfo()
    model.fit(X, y)
    print X.describe()
    model.printInfo()
    y_pred = model.predict(X)
    print y_pred
    plt.scatter(y, y_pred)
    plt.show()
