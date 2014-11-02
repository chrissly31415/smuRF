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
	boost::shared_ptr<Node> root;
	int min_node,tnodecount,tree_size,max_depth;
	bool probability,regression,entropy_loss;
	double w1;

	void printInfo();
	typedef struct pair<DataFrame, DataFrame> DoubleDF;
	void growTree(DataFrame &dataframe,const vector<int> &featList, const int featNr,  bool verbose = false);
	void growTree(DataFrame &dataframe, bool verbose = false);
	void showTree(const bool verbose=false);
	void createBranch(boost::shared_ptr<Node> parentNode,
			DataFrame &dfsplit, const int nrFeat, const bool verbose = false);
	double makePrediction(const DataFrame &testset, const boost::shared_ptr<Node> parentNode, const int observation,
			const bool verbose = false);
	Eigen::VectorXd
			predict(DataFrame &testSet, const bool verbose = false);


};

#endif /* TREE_H_ */
