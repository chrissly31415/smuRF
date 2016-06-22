/*
 * smuRF.cpp
 *
 *  Created on: Sep 16, 2012
 *      Author: Christoph Loschen
 */

#ifdef __unix__
#include <sys/time.h>
#endif

#if defined( _WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include <iostream>
#include <typeinfo>
#include <map>
#include <sstream>
#include "RandomForest.h"
#include "OnlineSGD.h"
#include "Tree.h"
#include "IOHelper.h"
#include "LUtils.h"
#include "RandomGen.h"
#include "Parameters.h"
#include "pysmurf.h"

using namespace std;

//compiling with gcc: -std=c++0x -I/home/loschen/programs/boost_1_57_0/boost -I/usr/include/eigen3 -O2 -Wall -static -c -fmessage-length=0 -fopenmp -pthread -floop-optimize  -funroll-loops -march=native -fomit-frame-pointer
//linking with gcc: g++ -L/usr/local/lib -static -pthread -o "smuRF"  ./DataFrame.o ./IOHelper.o ./Node.o ./RandomForest.o ./Tree.o ./smuRF.o  -lboost_regex -lgomp

//TODO Possible Improvements:
//TODO write metrics class
//TODO can we write a RF which only uses/extends matrix class from Eigen?
//TODO static allocation of arrays for tree structure, e.g. depth = f(n_samples)
//TODO improve predictor statistics: -> near zero variance and skewness
//TODO implement normal equation and stochastic gradient descent
//TODO interface to python: python interface http://stackoverflow.com/questions/145270/calling-c-c-from-python
//TODO improve parsing
//TODO Should not we use featX==a for categorical values? -> only numerical ons
//TODO Saving of models
//TODO transform categorical variable according to class outcome,  according to friedmann, p.328
//TODO check loss improvement per new node, -> feature importances!
//TODO use header names or orig header ints as string in order to work with small ds //small dataframes with header numbers
//TODO opt flags: -ffast-math -floop-optimize  -funroll-loops -march=native:-flto together with -fwhole-program -Ofast
//TODO -fomit-frame-pointer
//TODO flags: -floop-optimize  -funroll-loops -march=native -fomit-frame-pointer
//TODO no open mp for MSVC express...:-(

//split & remove features
DataFrame prepareDF(Parameters params) {
	IOHelper *iohelper = new IOHelper;
	DataFrame df = iohelper->readCSVfile(params.dataset.back());
	DataFrame df0;
	//1. split data
	if (params.splitinfo.splitcolumn > -1) {
		cout << "Splitting at column:" << params.splitinfo.splitcolumn
				<< " at value:" << params.splitinfo.splitvalue << endl;
		df.splitFrame(params.splitinfo.splitvalue, params.splitinfo.splitcolumn,
				df0, df);
	}
	//2. remove columns
	if (params.remove.size() > 0) {
		for (unsigned i = 0; i < params.remove.size(); i++) {
			cout << "Removing column:" << params.remove.at(i) << " ["
					<< df.header[params.remove.at(i)] << "]" << endl;
			df = df.removeColumn(params.remove.at(i));
		}
	}
	return df;
}

//crossvalidation
void xval(RandomGen rng, Parameters params, DataFrame df) {
	//if test (validation) set exists remove it
//	if (df.containsFeature("train") > -1) {
//		DataFrame trainDF;
//		DataFrame testDF;
//		df.splitFrame(0.5, df.containsFeature("train"), testDF, trainDF);
//		cout << "Splitting of test set and removing train variable." << endl;
//		df = trainDF.removeColumn(trainDF.containsFeature("train"));
//	}
	if (df.containsFeature("name") > -1) {
		df = df.removeColumn(df.containsFeature("name"));
		cout << "Removing column names." << endl;
	}
	df.printSummary();
	LUtils::Xvalidation(5, df, rng, params);
}

//transforms categorical string variables into numeric ones, needs header, asumes class to be last column
void transform(string filename) {
	IOHelper *iohelper = new IOHelper;
	iohelper->transformCSVfile(filename, "m" + filename);
	DataFrame df = iohelper->readCSVfile("m" + filename);
	df.printSummary();
}

//training_prediction, needs a column with train=1,test=0 column as first column
void protocol_special(RandomGen rng, Parameters params, DataFrame df0) {
	IOHelper *iohelper = new IOHelper;
	DataFrame trainDF;
	DataFrame testDF;
	if (params.testset.size() == 0) {
		df0.splitFrame(0.5, 0, testDF, trainDF);
		trainDF = trainDF.removeColumn(0);
		testDF = testDF.removeColumn(0);
	} else {
		string testsetfile = params.testset.back();
		cout << "Loading testset:" << testsetfile << endl;
		trainDF = df0;
		testDF = iohelper->readCSVfile(testsetfile);
	}

	//TRAIN
	trainDF.printSummary();
	RandomForest *myRF = new RandomForest();
	myRF->setParameters(params.nrtrees[0],params.mtry[0],params.min_nodes,params.max_depth,params.numjobs,params.verbose,params.regression);

	//myRF->setDataFrame(trainDF);
	myRF->printInfo();
	myRF->train(trainDF);
	//LOSS
	cout << "#Training (out-of-bag):" << endl;
	LUtils::evaluate(trainDF, myRF->poob_all, false, 1);
	//LUtils::aucLoss(myRF->dataframe.matrix.col(myRF->dataframe.classCol),
	//		myRF->poob_all, true);

	//TEST SET
	cout << "\n#Testset" << endl;
	testDF.printSummary();
	Eigen::VectorXd ptest = myRF->predict(testDF);
	LUtils::evaluate(testDF, ptest, false, 1);
	iohelper->writePredictions("prediction.csv", ptest);
	delete myRF;
	delete iohelper;
}


void showData(RandomGen rng, Parameters params, DataFrame df0) {
	df0.printSummary();
}


void simpleRF(RandomGen rng, Parameters params, DataFrame df0) {
	cout<<"\ntree size: "<<params.nrtrees.size();
	df0.printSummary();
	map<string, double> results;
	string min_pos = "";
	double min_loss = 10e15;
	for (unsigned i = 0; i < params.nrtrees.size(); i++) {
		for (unsigned j = 0; j < params.mtry.size(); j++) {
			RandomForest *myRF = new RandomForest();

			myRF->setParameters(params.nrtrees[i],params.mtry[j],params.min_nodes,params.max_depth,params.numjobs,params.verbose,params.regression);

			//myRF->setDataFrame(df0);
			//myRF->nrTrees = params.nrtrees[i];
			//myRF->mTry = params.mtry[j];
			myRF->printInfo();
			myRF->train(df0);
			double loss = myRF->oob_loss;
			//save results
			stringstream info;
			info << "->ntree: " << params.nrtrees[i] << " try_features: "
					<< params.mtry[j];
			const std::string& tmp = info.str();
			results.insert(std::make_pair(tmp, loss));
			if (loss < min_loss) {
				min_loss = loss;
				min_pos = info.str();
			}
			//cout<<"\nForest structure:\n"<<myRF->forest2string();
			delete myRF;
		}
	}
	if (params.nrtrees.size() + params.mtry.size() > 1) {
		printf("SUMMARY:\n");
		cout << fixed << setprecision(3);
		for (map<string, double>::const_iterator it = results.begin();
				it != results.end(); ++it) {
			if (!df0.regression) {
				cout << it->first << " - Correctly classified: "
						<< (1.0 - it->second) * 100 << "%" << endl;
			} else {
				cout << setw(24) << it->first << " RMSE: " << setw(6)
						<< it->second << endl;
			}
		}
		if (!df0.regression) {
			cout << ">>Optimum: " << min_pos << " with loss:"
					<< 100 * (1.0 - min_loss) << "%" << endl;
		} else {
			cout << ">>Optimum: " << min_pos << " with loss:" << min_loss
					<< endl;
		}

	}
}

void simpleTree(RandomGen rng, Parameters params, DataFrame df0) {
	Tree *myTree = new Tree(params.min_nodes, params.probability,
			df0.regression);
	myTree->max_depth = params.max_depth;
	myTree->train(df0, false);
	myTree->showTree();
	string tmp = myTree->tree2string();
	cout << tmp<<flush<<endl;
	Eigen::VectorXd p = myTree->predict(df0);
	LUtils::evaluate(df0, p, params.probability, 0);
}

void multiTree(RandomGen rng, Parameters params, DataFrame df0) {
	for (unsigned i = 0; i < 2; i++) {
		rng.setSeed(params.seed + i);
		simpleTree(rng, params, df0);
	}
}

void linear_model(RandomGen rng, Parameters params, DataFrame df0) {
	MLModel *model = new OnlineSGD();
	model->train(df0.matrix,df0.y);
	model->predict(df0.matrix);
}

void selectProtocol(RandomGen rng, Parameters params) {
	DataFrame df;
	if (params.splitinfo.splitcolumn > -1 || params.remove.size() > 0) {
		df = prepareDF(params);
	} else {
		IOHelper *iohelper = new IOHelper;
		df = iohelper->readCSVfile(params.dataset.back());
	}

	for (unsigned i = 0; i < params.protocol.size(); i++) {
		cout << "Job #" << (i + 1) << " ";
		if (params.protocol[i].find("rf") != std::string::npos) {
			cout << "Random Forest" << endl;
			simpleRF(rng, params, df);
		} else if (params.protocol[i].find("test") != std::string::npos) {
			//
		} else if (params.protocol[i].find("show") != std::string::npos) {
			cout << "Show Data" << endl;
			showData(rng, params, df);
		} else if (params.protocol[i].find("xval") != std::string::npos) {
			cout << "xvalidation" << endl;
			xval(rng, params, df);
		} else if (params.protocol[i].find("tree") != std::string::npos) {
			cout << "Decision Tree" << endl;
			simpleTree(rng, params, df);
		} else if (params.protocol[i].find("onlineSGD") != std::string::npos) {
			cout << "LinearModel" << endl;
			linear_model(rng, params, df);
		} else if (params.protocol[i].find("train_predict")
				!= std::string::npos) {
			cout << "Training&Prediction" << endl;
			protocol_special(rng, params, df);
		} else if (params.protocol[i].find("transform") != std::string::npos) {
			cout << "Transform dataset (categorical->numeric variables):"
					<< params.dataset.at(0) << endl;
			transform(params.dataset.at(0));
		} else {
			cout << "No protocol defined! Trying random forest." << endl;
			simpleRF(rng, params, df);
		}
	}
}

int main() {
	string version = "1.0";
	cout << "\n##########################################################\n";
	cout << "###    smuRF - simple multithreaded Random Forest      ###\n";
	cout << "###	                                               ###\n";
	cout << "###    version:" << version
			<< " (c)Christoph Loschen, 2012-2015     ###\n";
	cout << "##########################################################\n\n";
#ifdef __unix__
	timeval t1, t2;
	gettimeofday(&t1, NULL);
#endif

#if defined( _WIN32) || defined(_WIN64)
	cout<<"Detecting windows machine...";
	LARGE_INTEGER frequency;// ticks per second
	LARGE_INTEGER t1, t2;// ticks
	// get ticks per second
	QueryPerformanceFrequency(&frequency);
	// start timer
	QueryPerformanceCounter(&t1);
#endif

	IOHelper *iohelper = new IOHelper;
	Parameters params = iohelper->parseParameters("setup.txt");
	static RandomGen rng(params.seed);
	selectProtocol(rng, params);



#ifdef __unix__
	gettimeofday(&t2, NULL);
	LUtils::printTiming(t1, t2);
#endif
#if defined( _WIN32) || defined(_WIN64)
	QueryPerformanceCounter(&t2);
	LUtils::printTiming_win(t1,t2,frequency);
#endif
	delete iohelper;
}
