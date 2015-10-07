/*
 * RandomForest.h
 *
 *  Created on: Jan 2, 2013
 *   Author: Christoph Loschen
 */

#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include "MLModel.h"
#include "DataFrame.h"
#include "Tree.h"
#include "RandomGen.h"
#include "Parameters.h"

using namespace std;

struct RandomForest {
	RandomForest();
	RandomForest(DataFrame ldf, RandomGen rng, Parameters params);
	~RandomForest();

	DataFrame dataframe;
	RandomGen rng;

	int nrTrees,mTry,min_node,max_depth,verbose_level,probability,numjobs;
	double weight,oob_loss;
	bool entropy_loss;

	vector<Tree> trees;
	vector<int> oobcounter;
	Eigen::VectorXd poob_all;

	void setParameters(DataFrame ldf, int nrTrees, int mTry, int min_node, int max_depth, int verbose_level, int probability,int numjobs);
	void setParameters(int nrTrees, int mTry, int min_node, int max_depth, int numjobs);
	void setDataFrame(DataFrame &ldf);

	void train();
	Eigen::VectorXd predict(DataFrame &testSet, const bool verbose = false);

	Eigen::VectorXd averageOOB(const Eigen::MatrixXd &pall);
	void printInfo();
	string forest2string();


};

#endif /* RANDOMFOREST_H_ */
