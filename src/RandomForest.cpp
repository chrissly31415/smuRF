/*
 * RandomForest.cpp
 *
 *  Created on: Jan 2, 2013
 *      Author: Christoph Loschen
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "RandomForest.h"
#include "LUtils.h"
#include "Tree.h"
#include "IOHelper.h"
#include <iostream>

RandomForest::RandomForest() {
	nrTrees = 100;
	mTry = 5;
	min_node = 1;
	max_depth = 30;
	verbose_level = 0;
	probability = true;
	numjobs = 1;
	weight = 1.0;
	entropy_loss = false;
	oob_loss = 0.0;
}

RandomForest::RandomForest(DataFrame ldf, RandomGen lrng, Parameters params) :
		dataframe(ldf), rng(lrng), nrTrees(params.nrtrees.back()), mTry(
				params.mtry.back()), min_node(params.min_nodes), max_depth(
				params.max_depth), verbose_level(params.verbose), probability(
				params.probability), numjobs(params.numjobs), weight(
				params.weight), entropy_loss(params.entropy) {
	oob_loss = 0.0;
}

//void RandomForest::setParameters(DataFrame ldf_, int nrTrees_, int mTry_,
//		int min_node_, int max_depth_, int verbose_level_, int probability_,
//		int numjobs_) {
//	dataframe = ldf_;
//	nrTrees = nrTrees_;
//	mTry = mTry_;
//	min_node = min_node_;
//	max_depth = max_depth_;
//	verbose_level = verbose_level_;
//	probability = probability_;
//	numjobs = numjobs_;
//	;
//}
void RandomForest::setParameters(int nrTrees_, int mTry_, int min_node_,
		int max_depth_, int numjobs_, int verbose_level_) {
	nrTrees = nrTrees_;
	mTry = mTry_;
	min_node = min_node_;
	max_depth = max_depth_;
	numjobs = numjobs_;
	verbose_level = verbose_level_;
}

void RandomForest::setDataFrame(DataFrame &ldf_) {
	dataframe = ldf_;
}

//prints general info on RF
void RandomForest::printInfo() {
	cout << endl << "RandomForest" << endl;
	cout << left << setw(20) << "trees: " << nrTrees << endl;
	cout << setw(20) << "min. node size: " << min_node << endl;
	cout << setw(20) << "max depth: " << max_depth << endl;
	cout << setw(20) << "try_features: " << mTry << endl;
	if (dataframe.nrrows==0) {
	if (!dataframe.regression && (weight > 0.501 || weight < 0.499)) {
		cout << setw(20) << "weight class 1: " << weight << endl;
		cout << setw(20) << "weight class 0: " << (1.0 - weight) << endl;
	}
	if (entropy_loss && !dataframe.regression) {
		cout << setw(20) << "loss function: " << "entropy" << endl;
	} else if (!entropy_loss && !dataframe.regression) {
		cout << setw(20) << "loss function: " << "gini" << endl;
	} else if (dataframe.regression) {
		cout << setw(20) << "loss function: " << "RMSE" << endl;
	}
	cout << setw(20) << "data: " << dataframe.nrrows << "x" << dataframe.nrcols
			<< endl;
	if (probability && !dataframe.regression)
		cout << "returning probabilities." << endl;
	}
	cout << setw(20) << "verbose level: " << verbose_level << endl;
	if (numjobs > 1) {
		cout << setw(20) << "parallel jobs: " << numjobs << endl;
	}

	cout << endl;
}

string RandomForest::forest2string() {
	string tmp = "";
	for (int i = 0; i < nrTrees; i++) {
		tmp = tmp + trees[i].tree2string();
	}
	return tmp;
}

//predict, could also be parallized!
Eigen::VectorXd RandomForest::predict(DataFrame &testSet, const bool verbose) {
	Eigen::MatrixXd pall(testSet.nrrows, nrTrees);
	Eigen::VectorXd p;
	if (verbose) cout << "Prediction (" << trees.size() << " trees)" << endl;
    //dataframe.restoreOrder();
	for (int i = 0; i < nrTrees; i++) {

		p = trees[i].predict(testSet, false);
		pall.col(i) = p;
	}
	p = pall.rowwise().sum() / (double) nrTrees;
	if (!probability && !dataframe.regression)
		return LUtils::round(p);
	else
		return p;
}

//grows Forest
//@TODO we should control random seeds for sample function (used for bootstrap and feature selection in growTree)
void RandomForest::train() {
	int nthreads = 1;
	bool verbose = verbose_level>0;
#ifdef _OPENMP
	omp_set_num_threads(numjobs);
#pragma omp parallel
	nthreads = omp_get_num_threads();
	if (verbose) cout << "++Using OPENMP - NUMTHREADS:" << nthreads << "++\n";
#endif

	oobcounter = vector<int>(dataframe.nrrows);
	int nrFeat = mTry;
	if (nrFeat > dataframe.nrcols - 1) {
		cout << "ERROR: Illegal value for try_features, maximum value:"
				<< (dataframe.nrcols - 1) << endl;
		exit(1);
	}
	if (verbose)  cout << "Growing forest with " << nrFeat << " random features, using sampling with replacement."<< endl;

	Eigen::MatrixXd pall = Eigen::MatrixXd::Zero(dataframe.nrrows, nrTrees);

	//seeds and out of bag indices generated outside parallel loop to warrant reproducibility
	vector<int> tree_seeds = LUtils::sample(rng, nrTrees, 1000000, true);
	vector<vector<int> > inbagindices(nrTrees);
	for (int i = 0; i < nrTrees; i++) {
		inbagindices[i] = LUtils::sample(rng, dataframe.nrrows,
				dataframe.nrrows, true);
	}

#pragma omp parallel for shared(pall)
	for (int j = 0; j < nthreads; j++) {
		DataFrame localFrame = dataframe.copy();
		const int id = omp_get_thread_num();
		if (nrTrees % nthreads != 0) {
			cout
					<< "ERROR: Current restriction: the number of trees should be divisible by number of threads."
					<< endl;
			exit(1);
		}
		int nrTrees_local = nrTrees / nthreads;
		if (verbose) cout << "THREAD::" << id << " with " << nrTrees_local<< " trees started.\n";

		for (int i = 0; i < nrTrees_local; i++) {
			//serial index in all original trees
			int index = id * nrTrees_local + i;

			vector<int> featList(localFrame.nrcols - 1);
			for (unsigned j = 0; j < featList.size(); ++j) {
				featList[j] = j;
			}
			//vector<int> inbagidx = LUtils::sample(rng, localFrame.nrrows,
			//		localFrame.nrrows, true);
			vector<int> inbagidx = inbagindices[index];
			DataFrame inbagsample = localFrame.getRows(inbagidx, true);
			Tree myTree(min_node, probability, inbagsample.regression, weight,
					tree_seeds[index]);
			myTree.max_depth = max_depth;

			myTree.train(inbagsample, featList, nrFeat, verbose=false);
			//predict out-of-bag data
			vector<int> oobagidx = LUtils::complement(inbagidx,
					localFrame.nrrows);
			DataFrame oobagsample = localFrame.getRows(oobagidx, true);
			Eigen::VectorXd p_oob = myTree.predict(oobagsample, verbose=false);

			Eigen::VectorXd p = LUtils::fillPredictions(p_oob, oobagidx,
					localFrame.nrrows, oobcounter);

#pragma omp critical
			pall.col(index) = p;
			if (i > 0 && i % 25 == 0 &&  verbose ) {
				cout << "THREAD::" << id << " iteration: " << setw(5) << i
						<< " current tree size: " << setw(5) << myTree.tree_size
						<< endl;
			}

#pragma omp critical
			trees.push_back(myTree);
		}
		if (verbose) cout << "THREAD::" << id << " finished\n";
	}
	poob_all = averageOOB(pall);
	if (verbose) cout << "Training results (out-of-bag):" << endl;
	oob_loss = LUtils::evaluate(dataframe, poob_all, probability,
			verbose_level);

//
//	if (!dataframe.regression)
//		 LUtils::aucLoss(dataframe.matrix.col(dataframe.classCol), poob_all,
//		 true);
}

Eigen::VectorXd RandomForest::averageOOB(const Eigen::MatrixXd &pall) {
	Eigen::VectorXd p_local = pall.rowwise().sum();
	int leftout = 0;
	for (int i = 0; i < p_local.size(); ++i) {
		if (oobcounter[i] == 0) {
			leftout++;
		} else {
			p_local(i) = p_local(i) / (double) oobcounter[i];
		}
	}
	return p_local;
}

RandomForest::~RandomForest() {
	//TODO iterate over trees to delete them
}
