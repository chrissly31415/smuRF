/*
 * RandomForest.cpp
 *
 *  Created on: Jan 2, 2013
 *      Author: Christoph Loschen
 */

#include "RandomForest.h"
#include <omp.h>
#include "LUtils.h"
#include "Tree.h"
#include "IOHelper.h"
#include <iostream>

RandomForest::RandomForest(DataFrame ldf, RandomGen lrng, Parameters params) :
	dataframe(ldf), rng(lrng), nrTrees(params.nrtrees.back()),
			mTry(params.mtry.back()), min_node(params.min_nodes),
			max_depth(params.max_depth), verbose_level(params.verbose),probability(params.probability),numjobs(params.numjobs),
			weight(params.weight), entropy_loss(params.entropy)
			  {
	oobcounter = vector<int> (dataframe.nrrows);
	oob_loss = 0.0;
}


//prints general info on RF
void RandomForest::printInfo() {
	cout << endl << "RandomForest" << endl;
	cout << left << setw(20) << "trees: " << nrTrees << endl;
	cout << setw(20) << "min. node size: " << min_node << endl;
	cout << setw(20) << "max depth: " << max_depth << endl;
	cout << setw(20) << "try_features: " << mTry << endl;
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
	cout << setw(20) << "verbose level: " << verbose_level << endl;
	if (numjobs>1) {
		cout << setw(20) << "parallel jobs: " << numjobs << endl;
	}

	cout << endl;
}

//predict
Eigen::VectorXd RandomForest::predict(DataFrame &testSet, const bool verbose) {
	Eigen::MatrixXd pall(testSet.nrrows, nrTrees);
	Eigen::VectorXd p;
	cout << "Prediction (" << trees.size() << " trees)" << endl;
	for (int i = 0; i < nrTrees; i++) {
		dataframe.restoreOrder();
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
void RandomForest::growForest_parallel() {
	int nthreads = 1;
#ifdef OMP_H
	omp_set_num_threads(numjobs);
#pragma omp parallel
	nthreads = omp_get_num_threads();
	cout << "++Using OPENMP - NUMTHREADS:" << nthreads << "++\n";
#endif
	int nrFeat = mTry;
	if (nrFeat > dataframe.nrcols - 1) {
		cout << "ERROR: Illegal value for try_features, maximum value:"
				<< (dataframe.nrcols - 1) << endl;
		exit(1);
	}
	cout << "Growing forest with " << nrFeat
			<< " random features, using sampling with replacement.";
	cout << endl;
	Eigen::MatrixXd pall = Eigen::MatrixXd::Zero(dataframe.nrrows, nrTrees);

	//seeds and out of bag indices generated outside parallel loop to warrant reproducibility
	vector<int> tree_seeds = LUtils::sample(rng, nrTrees,
				1000000, true);
	vector<vector<int> > inbagindices(nrTrees);
	for (int i = 0; i < nrTrees; i++) {
		inbagindices[i] = LUtils::sample(rng, dataframe.nrrows,
				dataframe.nrrows, true);
	}

#pragma omp parallel for shared(pall)
	for (int j = 0; j < nthreads; j++) {
		DataFrame localFrame = dataframe.copy();
		const int id = omp_get_thread_num();
		if (nrTrees % nthreads !=0) {
			cout<<"ERROR: Current restriction: the number of trees should be divisible by number of threads."<<endl;
			exit(1);
		}
		int nrTrees_local = nrTrees / nthreads;
		cout << "Starting thread: " << id << " with " << nrTrees_local << " trees\n";

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
			Tree myTree(min_node, probability, inbagsample.regression, weight,tree_seeds[index]);
			myTree.max_depth = max_depth;
			bool verbose = false;
			myTree.growTree(inbagsample, featList, nrFeat, verbose);
			//predict out-of-bag data
			vector<int> oobagidx = LUtils::complement(inbagidx,
					localFrame.nrrows);
			DataFrame oobagsample = localFrame.getRows(oobagidx, true);
			Eigen::VectorXd p_oob = myTree.predict(oobagsample, verbose);
			Eigen::VectorXd p = LUtils::fillPredictions(p_oob, oobagidx,
					localFrame.nrrows, oobcounter);

#pragma omp critical
			pall.col(index) = p;
			if (i>0 && i % 25 == 0 &&  verbose_level > 0) {
				cout << "THREAD::" << id << " iteration: " << setw(5) << i
						<< " current tree size: " << setw(5) << myTree.tree_size
						<< endl;
			}

#pragma omp critical
			trees.push_back(myTree);
		}
		cout << "THREAD::" << id << " finished\n";
	}
	poob_all = averageOOB(pall);
//	cout << "Training results (out-of-bag):" << endl;
	oob_loss
			= LUtils::evaluate(dataframe, poob_all, probability, verbose_level);
//
//	if (!dataframe.regression)
//		 LUtils::aucLoss(dataframe.matrix.col(dataframe.classCol), poob_all,
//		 true);
}

//grows Forest seriel @deprecated
void RandomForest::growForest() {
	int nrFeat = mTry;
	bool bootstrap = true;
	bool verbose = false;
	vector<int> featfreq(dataframe.nrcols - 1);
	if (nrFeat > dataframe.nrcols - 1) {
		cout << "ERROR: Illegal value for try_features, maximum value:"
				<< (dataframe.nrcols - 1) << endl;
		exit(1);
	}
	cout << "Growing forest with " << nrFeat << " random features ";
	if (bootstrap) {
		cout << ", using sampling with replacement.";
	} else {
		cout << ", using Sampling WITHOUT replacement.";
	}
	cout << endl;
	Eigen::MatrixXd pall = Eigen::MatrixXd::Zero(dataframe.nrrows, nrTrees);
	//#pragma omp parallel for private(dataframe)
	for (int i = 0; i < nrTrees; i++) {
		//const int id = omp_get_thread_num();
		//cout<<"THREAD::"<<id<<"\n";
		vector<int> featList(dataframe.nrcols - 1);
		for (unsigned j = 0; j < featList.size(); ++j) {
			featList[j] = j;
		}
		DataFrame localFrame = dataframe;
		//sample with replacement
		vector<int> inbagidx;
		if (bootstrap) {
			//we try to get identical distributed sample, does not make a difference
			inbagidx = LUtils::sample(rng, localFrame.nrrows,
					localFrame.nrrows,
					localFrame.matrix.col(localFrame.classCol), true);
		} else {
			inbagidx = LUtils::sample(rng,
					localFrame.nrrows - localFrame.nrrows / 5,
					localFrame.nrrows, false);
		}
		DataFrame inbagsample = localFrame.getRows(inbagidx, true);
		Tree myTree(min_node, probability, inbagsample.regression, weight);
		myTree.max_depth = max_depth;
		myTree.growTree(inbagsample, featList, nrFeat, verbose);
		//predict out-of-bag data
		vector<int> oobagidx = LUtils::complement(inbagidx, localFrame.nrrows);
		//LUtils::print(oobagidx);
		DataFrame oobagsample = localFrame.getRows(oobagidx, true);
		Eigen::VectorXd p_oob = myTree.predict(oobagsample, verbose);

		Eigen::VectorXd p = LUtils::fillPredictions(p_oob, oobagidx,
				localFrame.nrrows, oobcounter);
		//#pragma omp critical
		pall.col(i) = p;

		//verbose=true;
		if (i % 25 == 0 && verbose_level > 0) {
			cout << "iteration:" << setw(5) << i << " current tree size:"
					<< setw(5) << myTree.tree_size;
			if (i > 20) {
				Eigen::VectorXd p_local = averageOOB(pall);
				double loss = LUtils::evaluate(localFrame, p_local,
						probability, 0);
				cout << fixed << setprecision(3) << "out-of-bag loss: " << loss
						<< endl;
			} else {
				cout << endl;
			}
			cout.flush();
		} else if (i % 25 == 0) {
			cout << "iteration:: " << setw(5) << i;
			cout << endl;
		}
		trees.push_back(myTree);
	}
	cout << endl;
	poob_all = averageOOB(pall);
	cout << "Training results (out-of-bag):" << endl;
	oob_loss
			= LUtils::evaluate(dataframe, poob_all, probability, verbose_level);
	if (!dataframe.regression)
		LUtils::aucLoss(dataframe.matrix.col(dataframe.classCol), poob_all,
				true);
	//iohelper->writePred2CSV("oob.csv", dataframe, poob_all, true, true);
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
