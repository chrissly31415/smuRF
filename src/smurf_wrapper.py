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
	print type(self.obj)
	lib.RandomForest_printInfo(self.obj)

    def setDataFrame(self,df):
	print type(df.ptr_)
	lib.RandomForest_setDataFrame(self.obj,df.ptr_)
      
    def train(self):
        lib.RandomForest_train(self.obj)
        
    #def predict(self, a):
	#lib.RF_predict.restype = ctypes.c_double
        #res = lib.RF_predict(self.obj, ctypes.c_double(a))
        #return res

#X = np.asarray([[1,2,0,7],[3,9,3,7.2],[-4,9.1,-2.1,1]],dtype=np.float32)
#y = np.asarray([2.1,0.1,0.2],dtype=np.float32)

#X = pd.read_csv('../data/reg_test4.csv',sep=',')
X = pd.read_csv('../data/katritzky_n_small.csv',sep=',')

#X = pd.read_csv('../data/mp_cdk.csv',sep=',')
X = X._get_numeric_data()
print X.describe()
#y = X['Ave °C']
y = X['np_exp']
#X.drop(['train','Ave °C'],axis=1,inplace=True)
X.drop(['train','np_exp'],axis=1,inplace=True)
print X.describe()
print type(y.values)

df = DF(X.values,label=y.values)
df.printSummary()

rf = RandomForest(n_estimators=100,mtry=5,node_size=5,max_depth=30,n_jobs=4)
rf.setDataFrame(df)
rf.printInfo()
rf.train()
#f.fit()
#a = f.predict(2.0)
#print a


