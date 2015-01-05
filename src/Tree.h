/*
 * Tree.h
 *
 *  Created on: Sep 20, 2012
 *      Author: Chris
 */

#ifndef TREE_H_
#define TREE_H_

#include "MLModel.h"
#include "DataFrame.h"
#include "Node.h"
#include "RandomGen.h"


using namespace std;

class Tree  {

public:

	Tree (int n, bool p = false, bool r = false, double w=0.5, int seed=42);
	virtual ~Tree();
	boost::shared_ptr<Node> root;
	int min_node,tnodecount,tree_size,max_depth;
	bool probability,regression,entropy_loss;
	double w1;
	int rseed;
	RandomGen rng;

	void printInfo();
	string tree2string();
	typedef struct pair<DataFrame, DataFrame> DoubleDF;

	void train(DataFrame &dataframe, const bool verbose = false);
	void train(DataFrame &dataframe,const vector<int> &featList, const int featNr,  bool verbose = false);

	void showTree(const bool verbose=false);
	void createBranch(boost::shared_ptr<Node> parentNode,
			DataFrame &dfsplit, const int nrFeat, const bool verbose = false);
	double makePrediction(const DataFrame &testset, const boost::shared_ptr<Node> parentNode, const int observation,
			const bool verbose = false);
	virtual Eigen::VectorXd
			predict(DataFrame &testSet, const bool verbose = false);


	//using interface mlmodel in Future:
//	virtual void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const bool verbose = false);
//	virtual Eigen::VectorXd predict(const Eigen::MatrixXd &Xtest, const bool verbose = false);

};

#endif /* TREE_H_ */
