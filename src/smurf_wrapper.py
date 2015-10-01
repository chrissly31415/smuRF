
import numpy as np
import ctypes 
lib = ctypes.cdll.LoadLibrary('./python_interface/libsmuRF.so')



class DF(object):
    def __init__(self,X=None):
	self.obj = lib.DF_new()
	lib.DF_set_parameters(self.obj)
	if isinstance(X, np.ndarray) and len(X.shape) == 2:
	  #self.handle = ctypes.c_void_p()
	  #lib.DF_createFromNumpy(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),X.shape[0], X.shape[1],ctypes.byref(self.handle))
	  lib.DF_createFromNumpy(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),X.shape[0], X.shape[1])
	else:
	  print X.shape
	  print "Need numpy matrix...!"


class RF(object):
    def __init__(self,n_estimators=500,max_depth=30):
        self.obj = lib.RF_new()
        lib.RF_set_parameters(self.obj,ctypes.c_int(n_estimators),ctypes.c_int(max_depth))

    def fit(self):
        lib.RF_fit(self.obj)
        
    def predict(self, a):
	lib.RF_predict.restype = ctypes.c_double
        res = lib.RF_predict(self.obj, ctypes.c_double(a))
        return res


X = np.asarray([[1,2,0,4],[3,9,3,7],[-4,9.1,-2.1,1]],dtype=np.float32)
print X

df = DF(X)

f = RF()
f.fit()
a = f.predict(2.0)
print a


