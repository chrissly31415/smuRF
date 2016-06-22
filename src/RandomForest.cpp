/*
 * RandomForest.cpp
 *
 *  Created on: Jan 2, 2013
 *      Author: Christoph Loschen
 */

#include <iostream>
#include <omp.h>

#include "RandomForest.h"
#include "LUtils.h"
#include "Tree.h"
#include "IOHelper.h"


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
	regression = true;
	oob_loss = 0.0;
	rng= RandomGen(42);
}

//RandomForest::RandomForest(DataFrame ldf, RandomGen lrng, Parameters params) :
//		dataframe(ldf), rng(lrng), nrTrees(params.nrtrees.back()), mTry(
//				params.mtry.back()), min_node(params.min_nodes), max_depth(
//				params.max_depth), verbose_level(params.verbose), probability(
//				params.probability), numjobs(params.numjobs), weight(
//				params.weight), entropy_loss(params.entropy) {
//	oob_loss = 0.0;
//}

//RandomForest::RandomForest(RandomGen lrng, Parameters params) :
//		rng(lrng), nrTrees(params.nrtrees.back()), mTry(params.mtry.back()), min_node(
//				params.min_nodes), max_depth(params.max_depth), verbose_level(
//				params.verbose), probability(params.probability), numjobs(
//				params.numjobs), weight(params.weight), entropy_loss(
//				params.entropy) {
//	oob_loss = 0.0;
//	regression = true;
//}

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
		int max_depth_, int numjobs_, int verbose_level_, bool regression_ =
				true) {
	nrTrees = nrTrees_;
	//cout<<"Setting parameters:"<<nrTrees<<"\n";
	mTry = mTry_;
	min_node = min_node_;
	max_depth = max_depth_;
	numjobs = numjobs_;
	verbose_level = verbose_level_;
	regression = regression_;
}

//void RandomForest::setDataFrame(DataFrame &ldf_) {
//	dataframe = ldf_;
//}

//prints general info on RF
void RandomForest::printInfo() {
	cout << endl << "RandomForest" << endl;
	cout << left << setw(20) << "trees: " << nrTrees << endl;
	cout << setw(20) << "min. node size: " << min_node << endl;
	cout << setw(20) << "max depth: " << max_depth << endl;
	cout << setw(20) << "try_features: " << mTry << endl;
	/*if (dataframe.nrrows==0) {
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
	 }*/
	if (!regression && (weight > 0.501 || weight < 0.499)) {
		cout << setw(20) << "weight class 1: " << weight << endl;
		cout << setw(20) << "weight class 0: " << (1.0 - weight) << endl;
	}
	if (entropy_loss && !regression) {
		cout << setw(20) << "loss function: " << "entropy" << endl;
	} else if (!entropy_loss && !regression) {
		cout << setw(20) << "loss function: " << "gini" << endl;
	} else if (regression) {
		cout << setw(20) << "loss function: " << "RMSE" << endl;
	}
	if (probability && !regression) {
		cout << "returning probabilities." << endl;
	}

	cout << setw(20) << "parallel jobs: " << numjobs << endl;
	cout << setw(20) << "verbose level: " << verbose_level << endl;

	//printf("Address of RF is %p\n", (void *) this);

	cout << endl;
}

string RandomForest::forest2string() {
	string tmp = "";
	for (int i = 0; i < nrTrees; i++) {
		tmp = tmp + trees[i].tree2string();
	}
	return tmp;
}

//predict, could also be parallized...
Eigen::VectorXd RandomForest::predict(DataFrame &testSet, const bool verbose) {
	Eigen::MatrixXd pall(testSet.nrrows, nrTrees);
	Eigen::VectorXd p;
	if (verbose)
		cout << "Prediction (" << trees.size() << " trees)" << endl;
	//dataframe.restoreOrder();
	for (int i = 0; i < nrTrees; i++) {
		p = trees[i].predict(testSet, false);
		pall.col(i) = p;
	}
	p = pall.rowwise().sum() / (double) nrTrees;
	if (!probability && !regression)
		return LUtils::round(p);
	else
		return p;
}

//grows Forest
//@TODO we should control random seeds for sample function (used for bootstrap and feature selection in growTree)
void RandomForest::train(DataFrame &dataframe) {
	int nthreads = 1;
	bool verbose = verbose_level > 0;
	//check if dataframe is there...
	if (dataframe.nrrows <= 0) {
		cout << "ERROR: Missing valid datafame!\n";
		exit(1);
	}
	//printf("RF training - address of RF is %p\n", (void *) this);
	//printf("RF training - address of DF is %p\n", (void *) &dataframe);

	oobcounter = vector<int>(dataframe.nrrows);
	int nrFeat = mTry;
	if (nrFeat > dataframe.nrcols - 1) {
		cout << "ERROR: Illegal value for try_features, maximum value:"
				<< (dataframe.nrcols - 1) << endl;
		exit(1);
	}
	if (verbose) {
		cout << "Growing forest with " << nrFeat
				<< " random features, using sampling with replacement." << endl;
	}
	Eigen::MatrixXd pall = Eigen::MatrixXd::Zero(dataframe.nrrows, nrTrees);
	//seeds and out of bag indices generated outside parallel loop to warrant reproducibility
	vector<int> tree_seeds = LUtils::sample(rng, nrTrees, 1000000, true);
	vector<vector<int> > inbagindices(nrTrees);
	for (int i = 0; i < nrTrees; i++) {
		inbagindices[i] = LUtils::sample(rng, dataframe.nrrows,
				dataframe.nrrows, true);
	}
	//create batches
	int n_base = nrTrees/numjobs;
	vector<int> tree_partition(numjobs,n_base);
	//put remaining trees into last batch
	tree_partition[numjobs-1] = tree_partition[numjobs-1] + nrTrees % numjobs;

#pragma omp parallel
	omp_set_num_threads(numjobs);
	int nProcessors=omp_get_max_threads();
	nthreads = numjobs;
	if (verbose) cout << "++Using OPENMP - NUMTHREADS:" << nthreads << "/"<< nProcessors<< "++\n";


#pragma omp parallel for shared(pall)
	for (int j = 0; j < nthreads; j++) {
		DataFrame localFrame = dataframe.copy(); //really???
		const int id = omp_get_thread_num();

		int nrTrees_local = 0;
#pragma omp critical
		nrTrees_local =	tree_partition[j];

		if (verbose)
			cout << "THREAD::" << id << " with " << nrTrees_local
					<< " trees started.\n";


		for (int i = 0; i < nrTrees_local; i++) {
			//serial index in all original trees
			//int index = id * nrTrees_local + i;
			int index = id * n_base + i;
			//cout<<"index:"<<index<<endl;

			vector<int> featList(localFrame.nrcols - 1);
			for (unsigned j = 0; j < featList.size(); ++j) {
				featList[j] = j;
			}
			vector<int> inbagidx = inbagindices[index];
			DataFrame inbagsample = localFrame.getRows(inbagidx, true);

			Tree myTree(min_node, probability, regression, weight,
					tree_seeds[index]);

			myTree.max_depth = max_depth;

			myTree.train(inbagsample, featList, nrFeat, verbose_level>1);

			//predict out-of-bag data
			vector<int> oobagidx = LUtils::complement(inbagidx,
					localFrame.nrrows);
			DataFrame oobagsample = localFrame.getRows(oobagidx, true);
			Eigen::VectorXd p_oob = myTree.predict(oobagsample, verbose_level>1);

			Eigen::VectorXd p = LUtils::fillPredictions(p_oob, oobagidx,
					localFrame.nrrows, oobcounter);

			if (i > 0 && i % 25 == 0 && verbose) {
							cout << "THREAD::" << id << " iteration: " << setw(5) << i
									<< " - current tree size: " << setw(5) << myTree.tree_size
									<< " - tree index:"<<index<<endl;
						}
#pragma omp critical
			pall.col(index) = p;


#pragma omp critical
			trees.push_back(myTree);
		}
		if (verbose_level>0)
			cout << "THREAD::" << id << " finished\n";
	}
#pragma omp barrier
	poob_all = averageOOB(pall);
	if (verbose_level>0) {
		cout << "Training results (out-of-bag):" << endl;
		oob_loss = LUtils::evaluate(dataframe, poob_all, probability,
				verbose_level);
	}
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
	//cout<<"RF destructor being called..."<<endl;
	//printf("Address of RF is %p\n", (void *) this);
}
