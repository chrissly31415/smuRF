#ifndef PYSMURF_H_
#define PYSMURF_H_

#include <string>

#include "RandomForest.h"
#include "DataFrame.h"
#include <boost/make_shared.hpp>
#include <python2.7/Python.h>

using namespace std;

//void pointer
typedef void* DataFrameHandle;

//http://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes
//return types: http://stackoverflow.com/questions/6073599/python-ctypes-return-values-question
//http://stackoverflow.com/questions/19198872/how-do-i-return-objects-from-a-c-function-with-ctypes
//http://stackoverflow.com/questions/3131854/how-to-return-a-pointer-to-a-structure-in-ctypes
//http://stackoverflow.com/questions/13908226/how-to-pass-pointer-back-in-ctypes
//have a look at xgboost_wrapper.cpp & xgboost.py

extern "C" {
RandomForest* RandomForest_new() {
	return new RandomForest();
}

void RandomForest_setParameters(RandomForest *rf, int nrTrees, int mTry, int min_node, int max_depth, int n_jobs, int verbose_level, bool regression=true) {
	rf->setParameters(nrTrees, mTry, min_node, max_depth,n_jobs,verbose_level,regression);
}

void RandomForest_printInfo(RandomForest *rf) {
	rf->printInfo();
}

//void RandomForest_setDataFrame(RandomForest *rf,DataFrameHandle handle) {
//	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
//	//we want our own dataframe
//	//DataFrame copy_df = dsrc.copy();
//	//rf->setDataFrame(dsrc);
//}

void RandomForest_train(RandomForest *rf,DataFrameHandle handle) {
	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
	rf->train(dsrc);
}

PyObject* RandomForest_predict(RandomForest *rf,DataFrameHandle handle) {
	DataFrame &df = *static_cast<DataFrame*>(handle);
	//printf("RandomForest_predict... Address of DF is %p\n", (void *) &df);
	Eigen::VectorXd p = rf->predict(df,false);
	//const Eigen::IOFormat fmt(3, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
	//cout.precision(3);
	//const Eigen::IOFormat npformat(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "[", "]");
	//std::cout << p.format(npformat) <<endl;
	PyObject* result = PyList_New(0);
	for (int i = 0; i < p.size(); ++i) {
		float val = p(i);
		PyList_Append(result,  Py_BuildValue("d", val));
	}
	return result;
}

void RandomForest_Free(RandomForest *rf) {
	//cout<<"RandomForest_Free..."<<endl;
	delete rf;
}

void DF_copy(DataFrameHandle handle,DataFrameHandle* out) {
	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
	DataFrame dsrc_new = dsrc.copy();
	//*out =  dsrc_new;
}

//DataFrame
//DataFrame* DF_new() {
//	cout << "C: creating dataframe..." << endl;
//	return new DataFrame();
//}
//void DF_setParameters(DataFrameHandle handle) {
//	cout << "C: setting parameters...deprecated!" << endl;
	//df->setParameters();
//}

void DF_printSummary(DataFrameHandle handle) {
	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
	dsrc.printSummary();
}

void DF_createFromNumpy(const float *data, int nrow, int ncol, DataFrameHandle* out) {
	//cout<<"DF_createFromNumpy... "<<endl;
	//expects data in col-major i.e. FORTRAN style!
	DataFrame* df = new DataFrame(nrow, ncol, ncol - 1, true);
	DataFrame &tmp = *df;
	tmp.nrrows = nrow;
	tmp.nrcols = ncol;
	tmp.regression = true;

	for (int i = 0; i < ncol; ++i, data += nrow) {
		auto str = std::to_string(i);
		tmp.header[i] = str;
        //cout<<"row: "<<i<<" value:"<<data[i]<<endl;
		for (int j = 0; j < nrow; ++j) {
			tmp.matrix(j, i) = (double) data[j];
			//cout<<"row:"<<j<<" col:"<<i<<" value:"<<data[j]<<endl;
			if (i == (ncol - 1)) {
				tmp.y(j) = data[j];
			}
			if (i==0) tmp.order.at(j) = j;
		}
	}
    tmp.analyze();
	*out = df;
	//df->printData();
	//potential memory leak here!!!
	//delete df;
}

void DF_Free(DataFrameHandle handle) {
	//cout<<"DF_Free..."<<endl;
	delete static_cast<DataFrame*>(handle);
}

}

#endif /* PYSMURF_H_ */
