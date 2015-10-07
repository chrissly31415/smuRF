#ifndef PYSMURF_H_
#define PYSMURF_H_

#include <string>

#include "RandomForest.h"
#include "DataFrame.h"
#include <boost/make_shared.hpp>

using namespace std;

//void pointer
typedef void* DataFrameHandle;

//http://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes
//return types: http://stackoverflow.com/questions/6073599/python-ctypes-return-values-question
//http://stackoverflow.com/questions/19198872/how-do-i-return-objects-from-a-c-function-with-ctypes
//http://stackoverflow.com/questions/3131854/how-to-return-a-pointer-to-a-structure-in-ctypes
//http://stackoverflow.com/questions/13908226/how-to-pass-pointer-back-in-ctypes
//have a look at xgboost_wrapper.cpp & xgboost.py

//class DF {
//public:
//	void set_parameters() {
//	}
//
//};


extern "C" {
//RandomForest
RandomForest* RandomForest_new() {
	cout << "C: creating RF..." << endl;
	return new RandomForest();
}

void RandomForest_setParameters(RandomForest *rf, int nrTrees, int mTry, int min_node, int max_depth, int n_jobs) {
	rf->setParameters(nrTrees, mTry, min_node, max_depth,n_jobs);
}

void RandomForest_printInfo(RandomForest *rf) {
	rf->printInfo();
}

void RandomForest_setDataFrame(RandomForest *rf,DataFrameHandle handle) {
	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
	rf->setDataFrame(dsrc);
}

void RandomForest_train(RandomForest *rf) {
	rf->train();
}

void RandomForest_fit_Free(RandomForest *rf) {
	delete rf;
}

//double RF_predict(RF *rf, double a) {
//	double res = rf->predict(a);
//	return res;
//}

//DataFrame
DataFrame* DF_new() {
	cout << "C: creating dataframe..." << endl;
	return new DataFrame();
}

void DF_setParameters(DataFrameHandle handle) {
	cout << "C: setting parameters..." << endl;
	//df->setParameters();
}

void DF_printSummary(DataFrameHandle handle) {
	DataFrame &dsrc = *static_cast<DataFrame*>(handle);
	dsrc.printSummary();
}

void DF_createFromNumpy(const float *data, int nrow, int ncol, DataFrameHandle* out) {
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
	//tmp.printData();
	*out = df;
	//if deleted then problem????
	//delete df;
	//now it shoudld be returned...!
}

void DF_Free(DataFrameHandle handle) {
	delete static_cast<DataFrame*>(handle);
}

}

#endif /* PYSMURF_H_ */