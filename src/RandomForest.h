/*
 * RandomForest.h
 *
 *  Created on: Jan 2, 2013
 *   Author: Christoph Loschen
 */

#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include "DataFrame.h"
#include "Tree.h"
#include "RandomGen.h"
#include "Parameters.h"

using namespace std;

struct RandomForest {
	RandomForest(DataFrame ldf, RandomGen rng, Parameters params);

	~RandomForest();

	DataFrame dataframe;
	RandomGen rng;

	int nrTrees,mTry,min_node,max_depth,verbose_level,probability,numjobs;
	double weight,oob_loss;
	bool entropy_loss;

	vector<Tree> trees;
	vector<int> histogram;
	vector<int> oobcounter;

	Eigen::VectorXd poob_all;
	Eigen::VectorXd averageOOB(const Eigen::MatrixXd &pall);
	Eigen::VectorXd predict(DataFrame &testSet, const bool verbose = false);

	void growForest();
	void growForest_parallel();
	void growShuffledForest();
	void printInfo();

};

#endif /* RANDOMFOREST_H_ */
