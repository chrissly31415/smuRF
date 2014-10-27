/*
 * Tree.h
 *
 *  Created on: Sep 20, 2012
 *      Author: Chris
 */

#ifndef TREE_H_
#define TREE_H_

#include "DataFrame.h"
#include "Node.h"


using namespace std;

struct Tree {
	Tree (int n, bool p = false, bool r = false, double w=0.5);
	virtual ~Tree();
	//Node *root;
	boost::shared_ptr<Node> root;
	int min_node;
	bool probability;
	bool regression;
	int tree_size;
	int max_depth;
	bool entropy_loss;
	double w1;

	void printInfo();
	vector<vector<double> > subdata;
	typedef struct pair<DataFrame, DataFrame> DoubleDF;
	void growTree(DataFrame &dataframe,const vector<int> &featList, const int featNr,  bool verbose = false);
	void growTree(DataFrame &dataframe, bool verbose = false);
	void showTree(const bool verbose=false);
	void showFeatureFreq(const DataFrame &dataframe);
	void createBranch(boost::shared_ptr<Node> parentNode,
			DataFrame &dfsplit, const int nrFeat, const bool verbose = false);
	double makePrediction(const DataFrame &testset, const boost::shared_ptr<Node> parentNode, const int observation,
			const bool verbose = false);
	Eigen::VectorXd
			predict(DataFrame &testSet, const bool verbose = false);

	int tnodecount;
	vector<int> histogram;
};

#endif /* TREE_H_ */
