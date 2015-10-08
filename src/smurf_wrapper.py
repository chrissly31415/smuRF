# coding: utf-8
"""
smuRF: smart multithreaded Random Forest

Version: 1.0
Authors: Christoph Loschen
"""
import numpy as np
import ctypes 
lib = ctypes.cdll.LoadLibrary('./python_interface/libsmuRF.so')

import pandas as pd
import matplotlib.pyplot as plt

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
    def __init__(self,X,label=None):
	self.ptr_ = ctypes.c_void_p()
	#self.handle = lib.DF_new()
	#lib.DF_setParameters(self.obj)
	X = X.astype(np.float32)
	#print type(self.ptr_)
	if isinstance(X, np.ndarray) and len(X.shape) == 2:
	    #attach label to X
	    if isinstance(label, np.ndarray) and len(label.shape) == 1:
		label = label.astype(np.float32)
		X = np.append(X,label.reshape(label.shape[0],-1),axis=1)
	    lib.DF_createFromNumpy(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),X.shape[0], X.shape[1],ctypes.byref(self.ptr_))
	else:
	  print X.shape
	  print "ERROR: Need numpy matrix...!"
	
    def printSummary(self):
	lib.DF_printSummary(self.ptr_)

    def __del__(self):
	lib.DF_Free(self.ptr_)

class RandomForest(object):
    def __init__(self,n_estimators=500,mtry=2,node_size=5,max_depth=30,n_jobs=1):
        self.obj = lib.RandomForest_new()
        print type(self.obj)
        lib.RandomForest_setParameters(self.obj,ctypes.c_int(n_estimators),ctypes.c_int(mtry),ctypes.c_int(node_size),ctypes.c_int(max_depth),ctypes.c_int(n_jobs))

    def printInfo(self):
	lib.RandomForest_printInfo(self.obj)

    def setDataFrame(self,df):
	lib.RandomForest_setDataFrame(self.obj,df.ptr_)
      
    def train(self):
        lib.RandomForest_train(self.obj)
    
    def fit(self,X,y):
	df = DF(X.values,label=y.values)
	self.setDataFrame(df)
        lib.RandomForest_train(self.obj)
        
    def predict(self, X):
	df = DF(X.values)
	length = ctypes.c_ulong()
        preds = lib.RandomForest_predict.restype = ctypes.py_object
	preds = lib.RandomForest_predict(self.obj, df.ptr_)
	return np.asarray(preds)

#X = pd.read_csv('../data/katritzky_n_small.csv',sep=',')

X = pd.read_csv('../data/mp_cdk.csv',sep=',')
X = X._get_numeric_data()
print X.describe()
y = X['Ave °C']
#y = X['n_exp']
X.drop(['train','Ave °C'],axis=1,inplace=True)
#X.drop(['train','n_exp'],axis=1,inplace=True)
print X.describe()

model = RandomForest(n_estimators=240,mtry=5,node_size=5,max_depth=30,n_jobs=4)
#rf.setDataFrame(df)
#rf.printInfo()
model.fit(X,y)
model.printInfo()
y_pred = model.predict(X)
#print y_pred
plt.scatter(y,y_pred)
plt.show()
