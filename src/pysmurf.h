#ifndef PYSMURF_H_
#define PYSMURF_H_

#include <string>

#include "RandomForest.h"
#include "DataFrame.h"

using namespace std;

//http://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes
//have a look at xgboost_wrapper.cpp & xgboost.py

class DF {
public:

	void set_parameters() {
	}

};

class RF {
public:
	int nr_trees = 100;
	int max_depth = 30;

	void set_parameters(int n, int d) {
		nr_trees = n;
		max_depth = d;
	}

	void fit(double a) {
		std::cout << "\n\nTraining..." << std::endl;
	}

	double predict(double a) {
		std::cout << "Predicting...." << std::endl;
		return 2 * a;
	}

};

extern "C" {
//RandomForest
RF* RF_new() {
	cout << "C: creating RF..." << endl;
	return new RF();
}

void RF_set_parameters(RF *rf, int n, int d) {
	rf->set_parameters(n, d);
}

void RF_fit(RF *rf, double a) {
	rf->fit(a);
}

double RF_predict(RF *rf, double a) {
	double res = rf->predict(a);
	return res;
}

//DataFrame
DF* DF_new() {
	cout << "C: creating dataframe..." << endl;
	return new DF();
}

void DF_set_parameters(DF *df) {
	cout << "C: setting parameters..." << endl;
	df->set_parameters();
}

void DF_createFromNumpy(const float *data, int nrow, int ncol) {
	//y??
	DataFrame df(nrow, ncol, ncol - 1, true);
	for (int i = 0; i < nrow; ++i, data += ncol) {
		//cout<<"row: "<<i<<" value:"<<data[i]<<endl;
		for (int j = 0; j < ncol; ++j) {
			auto str = std::to_string(j);
			if (i==0) df.header[j] = str;
			df.matrix(i, j) = (double) data[j];
			//if (j == (ncol - 1)) {
			//	df.y(i) = data[j];
			//}
		}
		df.order.at(i) = i;
	}
	df.analyze();
	df.printData();
	df.printSummary();
}

}

#endif /* PYSMURF_H_ */
