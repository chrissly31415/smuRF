/*
 * smuRF.cpp
 *
 *  Created on: Sep 16, 2012
 *      Author: Christoph Loschen
 */

#include <sys/time.h>
#include <iostream>
#include <typeinfo>
#include <map>
#include <sstream>
#include "RandomForest.h"
#include "Tree.h"
#include "IOHelper.h"
#include "LUtils.h"
#include "RandomGen.h"
#include "Parameters.h"

using namespace std;

//TODO Possible Improvements:
//TODO write metrics class
//TODO parallize RF by copying data.frame
//TODO can we write a RF which only uses matrix class?
//TODO implement normal equation and stochastic gradient descent
//TODO interface to python
//TODO improve parsing
//TODO Shouldnot we use featX==a fro categorical values?
//TODO Saving of models
//TODO transform categorical variable according to class outcome,  according to friedmann, p.328
//TODO check loss improvement per new node, -> feature importances!
//TODO clever checking of pure nodes???
//TODO use header names or orig header ints as string in order to work with small ds //small dataframes with header numbers
//TODO classes for DF and NODES necessary?
//TODO logloss as split criterion?
//TODO opt flags: -ffast-math -floop-optimize  -funroll-loops -march=native:-flto together with -fwhole-program -Ofast
//TODO -fomit-frame-pointer

//split & remove features
DataFrame prepareDF(Parameters params) {
	IOHelper *iohelper = new IOHelper;
	DataFrame df = iohelper->readCSVfile(params.dataset.back());
	DataFrame df0;
	//1. split data
	if (params.splitinfo.splitcolumn > -1) {
		cout << "Splitting at column:" << params.splitinfo.splitcolumn
				<< " at value:" << params.splitinfo.splitvalue << endl;
		df.splitFrame(params.splitinfo.splitvalue,
				params.splitinfo.splitcolumn, df0, df);
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

//split up male and female
void protocol3(RandomGen rng, Parameters params) {
	IOHelper *iohelper = new IOHelper;
	DataFrame df0 =
			iohelper->readCSVfile(params.dataset.back()).removeColumn(0);
	//split male-female
	DataFrame maleDF;
	DataFrame femaleDF;
	df0.splitFrame(0.5, 2, femaleDF, maleDF);
	//train,test male
	DataFrame mtrainDF;
	DataFrame mtestDF;
	maleDF.splitFrame(0.5, 0, mtestDF, mtrainDF);
	mtrainDF = mtrainDF.removeColumn(0);
	mtestDF = mtestDF.removeColumn(0);
	mtrainDF.printSummary();
	//mtestDF.printSummary();
	//RF
	RandomForest *maleRF = new RandomForest(mtrainDF, rng, params);
	maleRF->printInfo();
	maleRF->growForest();
	//train,test female
	DataFrame ftrainDF;
	DataFrame ftestDF;
	femaleDF.splitFrame(0.5, 0, ftestDF, ftrainDF);
	ftrainDF = ftrainDF.removeColumn(0);
	ftestDF = ftestDF.removeColumn(0);
	ftrainDF.printSummary();
	//ftestDF.printSummary();
	RandomForest *femaleRF = new RandomForest(ftrainDF, rng, params);
	femaleRF->printInfo();
	femaleRF->growForest();
	delete maleRF;
	delete femaleRF;
}

void xval(RandomGen rng, Parameters params) {
	//iohelper->transformCSVfile("complete_mini.csv", "test.csv");
	IOHelper *iohelper = new IOHelper;
	DataFrame df = iohelper->readCSVfile(params.dataset.back(), params.entropy);
	if (df.containsFeature("train") > -1) {
		DataFrame trainDF;
		DataFrame testDF;
		df.splitFrame(0.5, df.containsFeature("train"), testDF, trainDF);
		cout << "Splitting of test set and removing train variable." << endl;
		df = trainDF.removeColumn(trainDF.containsFeature("train"));
	}
	if (df.containsFeature("name") > -1) {
		df = df.removeColumn(df.containsFeature("name"));
		cout << "Removing column names." << endl;
	}
	df.printSummary();
	//DataFrame trainDF;
	//DataFrame testDF;
	//df.splitFrame(0.5, 0, testDF, trainDF);
	//trainDF = trainDF.removeColumn(0);
	//testDF = testDF.removeColumn(0);
	LUtils::Xvalidation(5, df, rng, params);
}

void transform(string filename) {
	IOHelper *iohelper = new IOHelper;
	iohelper->transformCSVfile(filename, "m" + filename);
	DataFrame df = iohelper->readCSVfile("m" + filename);
	df.printSummary();
}


void special_transform(RandomGen rng, Parameters params) {
	int iter = params.iter;
	IOHelper *iohelper = new IOHelper;
	DataFrame df0 =
			iohelper->readCSVfile(params.dataset.back()).removeColumn(0);
	df0 = df0.removeColumn(9);
	string filename = params.dataset.back();
	for (int i = 0; i < iter; i++) {
		cout << "\n##### Creation of shuffled data: " << i + 1 << "/" << iter
				<< " ######" << endl;
		string is =
				static_cast<ostringstream*> (&(ostringstream() << i))->str();
		string nfilename = "m" + is + "_" + filename;
		DataFrame dfl;
		if (i == 0) {
			cout << "Skipping shuffling in iteration: " << i << endl;
			dfl = df0;
		} else {
			dfl = df0.shuffleCategoricals();

		}
		iohelper->writeDataframe2CSV(nfilename, dfl);
	}
}

void protocol_special(RandomGen rng, Parameters params) {
	int iter = params.iter;
	IOHelper *iohelper = new IOHelper;
	DataFrame df0 =
			iohelper->readCSVfile(params.dataset.back()).removeColumn(0);
	df0 = df0.removeColumn(9);
	DataFrame trainDF;
	DataFrame testDF;
	df0.splitFrame(0.5, 0, testDF, trainDF);
	Eigen::MatrixXd pall(trainDF.nrrows, iter);
	Eigen::MatrixXd pall_test(testDF.nrrows, iter);

	for (int i = 0; i < iter; i++) {
		cout << "\n##### Blending iteration: " << i + 1 << "/" << iter
				<< " ######" << endl;
		DataFrame dfl;
		if (i == -1) {
			cout << "Skipping shuffling in iteration: " << i << endl;
			dfl = df0;
		} else {
			dfl = df0.shuffleCategoricals();

		}
		DataFrame trainDF;
		DataFrame testDF;
		dfl.splitFrame(0.5, 0, testDF, trainDF);
		trainDF = trainDF.removeColumn(0);
		testDF = testDF.removeColumn(0);

		//TRAIN
		trainDF.printSummary();
		RandomForest *myRF = new RandomForest(trainDF, rng, params);
		myRF->printInfo();
		myRF->growForest();
		pall.col(i) = myRF->poob_all;

		//LOSS
		LUtils::evaluate(myRF->dataframe, myRF->poob_all, false, 1);
		LUtils::aucLoss(myRF->dataframe.matrix.col(myRF->dataframe.classCol),
				myRF->poob_all, true);

		//TEST SET
		cout << "#TESTSET#" << endl;
		testDF.printSummary();
		Eigen::VectorXd ptest = myRF->predict(testDF);
		pall_test.col(i) = ptest;
		LUtils::evaluate(testDF, ptest, true, false);
		delete myRF;
	}

	Eigen::VectorXd pfinal = pall.rowwise().sum() / (double) iter;
	cout << endl << "Blending results (" << iter << " datasets):" << endl;
	LUtils::evaluate(trainDF, pfinal, params.probability, 1);
	LUtils::aucLoss(trainDF.matrix.col(trainDF.classCol), pfinal, true);
	//prediction test set
	Eigen::VectorXd pfinal_test = pall_test.rowwise().sum() / (double) iter;
	iohelper->writePred2CSV("blending_oob.csv", trainDF, pfinal);
	iohelper->writePred2CSV("blending_prediction.csv", testDF, pfinal_test);
	delete iohelper;
}

void blending(RandomGen rng, Parameters params) {
	IOHelper *iohelper = new IOHelper;
	DataFrame df0 =
			iohelper->readCSVfile(params.dataset.back()).removeColumn(0);
	DataFrame trainDF;
	DataFrame testDF;
	df0.splitFrame(0.5, 0, testDF, trainDF);
	Eigen::MatrixXd pall(trainDF.nrrows, params.dataset.size());
	Eigen::MatrixXd pall_test(testDF.nrrows, params.dataset.size());
	for (unsigned i = 0; i < params.dataset.size(); i++) {
		rng.setSeed(params.seed + i);
		DataFrame df = iohelper->readCSVfile(params.dataset.at(i));
		if (df0.nrrows != df.nrrows) {
			cout << "Dataframes in blending routine are of unequal length!"
					<< endl;
			exit(1);
		}
		if (df.containsFeature("train") > -1) {
			df.splitFrame(0.5, df.containsFeature("train"), testDF, trainDF);
			cout << "Splitting of test set and removing train variable."
					<< endl;
			trainDF = trainDF.removeColumn(trainDF.containsFeature("train"));
			testDF = testDF.removeColumn(testDF.containsFeature("train"));
		} else {
			cout << "Dataset should contain column 'train'" << endl;
			exit(1);
		}
		if (df.containsFeature("name") > -1) {
			trainDF = trainDF.removeColumn(trainDF.containsFeature("name"));
			testDF = testDF.removeColumn(testDF.containsFeature("name"));
			cout << "Removing column names." << endl;
		}
		trainDF.printSummary();
		//train set
		params.mtry.push_back(max(4, (trainDF.nrcols / 2)));
		RandomForest *myRF = new RandomForest(trainDF, rng, params);
		myRF->printInfo();
		myRF->growForest();
		pall.col(i) = myRF->poob_all;

		//test set
		//testDF = testDF.removeColumn(0);
		testDF.printSummary();
		Eigen::VectorXd ptest = myRF->predict(testDF);
		pall_test.col(i) = LUtils::round(ptest);

	}
	Eigen::VectorXd pfinal = pall.rowwise().sum()
			/ (double) params.dataset.size();
	cout << endl << "Blending results (" << params.dataset.size()
			<< " datasets):" << endl;
	LUtils::evaluate(trainDF, pfinal, params.probability, 1);
	//prediction test set
	Eigen::VectorXd pfinal_test = pall_test.rowwise().sum()
			/ (double) params.dataset.size();
	iohelper->writePred2CSV("predicted_blending.csv", testDF, LUtils::round(
			pfinal_test));
}

void showData(RandomGen rng, Parameters params, DataFrame df0) {
	df0.printSummary();
}

void simpleRF(RandomGen rng, Parameters params, DataFrame df0) {
	df0.printSummary();
	map<string, double> results;
	string min_pos = "";
	double min_loss = 10e15;
	for (unsigned i = 0; i < params.nrtrees.size(); i++) {
		for (unsigned j = 0; j < params.mtry.size(); j++) {
			RandomForest *myRF = new RandomForest(df0, rng, params.nrtrees[i],
					params.mtry[j], params.min_nodes, params.max_depth,
					params.probability, params.weight,params.entropy,params.verbose);
			myRF->printInfo();
			myRF->growForest();
			//TEST SET
			//iohelper->writePred2CSV("predicted.csv", df0, p, true, true);
			double loss = myRF->oob_loss;
			//save results
			stringstream info;
			info << "ntree: " << params.nrtrees[i] << " mtry: "
					<< params.mtry[j];
			const std::string& tmp = info.str();
			results.insert(std::make_pair(tmp, loss));
			if (loss < min_loss) {
				min_loss = loss;
				min_pos = info.str();
			}
			delete myRF;
		}
	}
	if (params.nrtrees.size() + params.mtry.size() > 1) {
		printf("SUMMARY\n");
		cout<<fixed<<setprecision(2);
		for (map<string, double>::const_iterator it = results.begin(); it
				!= results.end(); ++it) {
			if (!df0.regression) {
				cout << it->first << " - Correctly classified: " << (1.0
						- it->second) * 100 << "%" << endl;
			} else {
				cout<<setw(24) <<it->first << " RMSE: "<<setw(6) << it->second << endl;
			}
		}
		if (!df0.regression) {
			cout << ">>Optimum: " << min_pos << " with loss:" << 100 * (1.0
					- min_loss) << "%" << endl;
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
	myTree->growTree(df0, false);
	myTree->showTree();
	Eigen::VectorXd p = myTree->predict(df0);
	LUtils::evaluate(df0, p, params.probability, 0);
}

void multiTree(RandomGen rng, Parameters params, DataFrame df0) {
	for (unsigned i = 0; i < 2; i++) {
		rng.setSeed(params.seed + i);
		simpleTree(rng, params, df0);
	}
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
		} else if (params.protocol[i].find("show") != std::string::npos) {
			cout << "Show Data" << endl;
			showData(rng, params, df);
		} else if (params.protocol[i].find("multitransform")
				!= std::string::npos) {
			cout << "Multi transform dataset (categorical->numeric variables):"
					<< params.dataset.at(0) << endl;
			special_transform(rng, params);
		} else if (params.protocol[i].find("transform") != std::string::npos) {
			cout << "Transform dataset (categorical->numeric variables):"
					<< params.dataset.at(0) << endl;
			transform(params.dataset.at(0));

		} else if (params.protocol[i].find("tree") != std::string::npos) {
			cout << "Decision Tree" << endl;
			simpleTree(rng, params, df);
		} else if (params.protocol[i].find("special") != std::string::npos) {
			cout << "Special" << endl;
			protocol_special(rng, params);
		} else {
			cout << "No protocol defined! Trying random forest." << endl;
			simpleRF(rng, params, df);
		}
	}
}

int main() {
	cout << "### smuRF - simple & userfriendly Random Forest ###\n";
	cout << "###	(c) Christoph Loschen, 2012-2014        ###\n";
	timeval t1, t2;
	gettimeofday(&t1, NULL);
	IOHelper *iohelper = new IOHelper;
	Parameters params = iohelper->parseParameters("setup.txt");
	static RandomGen rng(params.seed);
	selectProtocol(rng, params);
	gettimeofday(&t2, NULL);
	LUtils::printTiming(t1, t2);
	delete iohelper;
}
