/*
 * RandomForest.h
 *
 *  Created on: Jan 2, 2013
 *      Author: Christoph Loschen
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
	RandomForest(DataFrame ldf, RandomGen rng, int m, bool p=false, double w=0.5, bool l=false, int v=1);
	RandomForest(DataFrame ldf, RandomGen rng,  int m, int a, int b, int c=100, bool p=false, double w=0.5, bool l=false, int v=1);

	~RandomForest();
	DataFrame dataframe;
	RandomGen rng;
	int nrTrees;
	int mTry;
	int min_node;
	int max_depth;
	bool probability;
	double weight;
	bool entropy_loss;
	int verbose_level;
	vector<Tree> trees;
	vector<int> histogram;
	Eigen::VectorXd poob_all;
	double oob_loss;
	vector<int> oobcounter;
	void growForest();
	void growShuffledForest();
	void printInfo();
	Eigen::VectorXd averageOOB(const Eigen::MatrixXd &pall);
	Eigen::VectorXd predict(DataFrame &testSet, const bool verbose = false);
};

#endif /* RANDOMFOREST_H_ */
