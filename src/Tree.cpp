/*
 * Tree.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: Chris
 *
 *   TODO: pruning of tree
 */

#include <iostream>
#include "LUtils.h"
#include "IOHelper.h"
#include "Tree.h"
#include "Node.h"
#include <boost/make_shared.hpp>

//Binary tree
Tree::Tree(int n, bool p, bool r, double w, int seed) :
	min_node(n), probability(p), regression(r), w1(w), rseed(seed) {
	root=boost::make_shared<Node>(0, 0.0);
	tree_size = 0;
	max_depth = 100;
	//Node::NODECOUNTER = 0;
	tnodecount = 0;
	entropy_loss = false;
}

//most important information
void Tree::printInfo() {
	cout << endl;
	if (regression) {
		cout << "REGRESSION Tree";
	} else {
		cout << "CLASSIFICATION Tree";
	}
	if (probability)
		cout << " - Returning class probabilities";
	cout << " - Minimum node size: " << min_node << endl;
}

// creates recursively the tree, uses subset of features
void Tree::growTree(DataFrame &dataframe, const vector<int> &featList, const int nrFeat,
		const bool verbose) {
	//select features randomly
	vector<int> featsubset = LUtils::sample(nrFeat ,
					dataframe.nrcols - 1, false);
	DataFrame::FeatureResult featResult = dataframe.findBestFeature(featsubset,entropy_loss);
	//Create a new root node
	root->feature = featResult.opt_feat;
	root->splitvalue = featResult.opt_split;
	root->impurity = featResult.loss;
	root->nameFeature = dataframe.header.at(featResult.opt_feat);
	root->nrsamples = dataframe.nrrows;
	createBranch(root, dataframe, nrFeat,verbose);
}

void Tree::growTree(DataFrame &dataframe,
		const bool verbose) {
	vector<int> featList(dataframe.nrcols - 1);
	for (unsigned i = 0; i < featList.size(); ++i) {
			featList[i] = i;
		}
	growTree(dataframe, featList, featList.size(), verbose);
}

//recursively insert nodes
void Tree::createBranch(boost::shared_ptr<Node> parentNode,
		DataFrame &dfsplit, const int nrFeat, bool verbose) {
	vector<int> featsubset = LUtils::sample(nrFeat ,
			dfsplit.nrcols - 1, false);
	DataFrame leftDF;
	DataFrame rightDF;
	dfsplit.splitFrame(parentNode->splitvalue, parentNode->feature, leftDF,
			rightDF);
	tree_size++;
	//LEFT BRANCH
	if (verbose && leftDF.nrrows > 0) {
		cout << "...Creating left branch: Feature: "<<dfsplit.header[parentNode->feature]<<" Value:" <<parentNode->splitvalue<<" n:"<<leftDF.nrrows<<" with prediction:" << leftDF.cm
						<< endl;
		leftDF.printSummary();
	}
	if (leftDF.nrrows == 0) {
		//happens if one of the nodes is "practically" pure
		if (verbose) {
			cout << "No data in left node, right node:"<<rightDF.nrrows << endl;
			cout << "Left node: Parent node is terminal." << endl;
		}
		parentNode->isTerminal = true;
		tnodecount++;
		return;
	} else if (leftDF.nrrows <= min_node || parentNode->depth + 1 > max_depth
			|| leftDF.distinct[leftDF.classCol] < 2) {
		if (verbose)
			cout << "Terminal node, cm: " << leftDF.cm << endl;
		//Node *left = new Node(parentNode.depth + 1, leftDF.cm);
		boost::shared_ptr<Node> left=boost::make_shared<Node>(parentNode->depth + 1, leftDF.cm);
		left->isTerminal = true;

		left->nrsamples = leftDF.nrrows;
		parentNode->left = left;
		tnodecount++;
	} else {
		DataFrame::FeatureResult featResulta = leftDF.findBestFeature(featsubset,entropy_loss);
		boost::shared_ptr<Node> left=boost::make_shared<Node>(featResulta.opt_feat, featResulta.opt_split,
				featResulta.loss, parentNode->depth + 1, leftDF.header[
						featResulta.opt_feat], leftDF.nrrows, leftDF.cm);
		parentNode->left = left;
		createBranch(left, leftDF, nrFeat, verbose);
	}

	//RIGHT BRANCH
	if (verbose && rightDF.nrrows > 0) {
		cout << "...Creating right branch: Feature: "<<dfsplit.header[parentNode->feature]<<" Value:" <<parentNode->splitvalue<<" n:"<<rightDF.nrrows<<" with prediction:" << rightDF.cm
						<< endl;
		rightDF.printSummary();
	}
	if (rightDF.nrrows == 0) {
		//happens if one of the nodes is "practically" pure
		if (verbose) {
			cout << "No data in right node,  left node:"<<leftDF.nrrows<< endl;
			cout << "Right node: Parent node is terminal." << endl;
		}
		parentNode->isTerminal = true;
		tnodecount++;
		return;
	} else if (rightDF.nrrows <= min_node || parentNode->depth + 1 > max_depth
			|| rightDF.distinct[rightDF.classCol] < 2) {
		if (verbose)
			cout << "Terminal node, cm: " << rightDF.cm << endl;
		boost::shared_ptr<Node> right=boost::make_shared<Node>(parentNode->depth + 1, rightDF.cm);
		right->isTerminal = true;

		right->nrsamples = rightDF.nrrows;
		parentNode->right = right;
		tnodecount++;
	} else {
		DataFrame::FeatureResult featResultb =
				rightDF.findBestFeature(featsubset,entropy_loss);
		if (verbose)
			cout << "Terminal node, cm: " << rightDF.cm << endl;
		boost::shared_ptr<Node> right=boost::make_shared<Node>(featResultb.opt_feat, featResultb.opt_split,
				featResultb.loss, parentNode->depth + 1, rightDF.header[
						featResultb.opt_feat], rightDF.nrrows, rightDF.cm);
		parentNode->right = right;
		createBranch(right, rightDF, nrFeat, verbose);

	}
	//if we reach this point, we should return
	return;
}

void Tree::showTree(const bool verbose) {
	cout << endl << "###Printing Tree Structure..." << endl;
	root->showChildren(verbose);
	cout << "Number of terminal nodes:" << tnodecount << endl;
}


//prediction for external data
Eigen::VectorXd Tree::predict(DataFrame &testSet, const bool verbose) {
	//cout<<"Tree:predict"<<endl;
	//testSet.printData();
	//cout<<"left"<<flush;
	//starting at root level, then call makePredictions recursively
	double pi = 0.0;
	Eigen::VectorXd p(testSet.nrrows);
	for (int obs = 0; obs < testSet.nrrows; obs++) {
		//in some rare case we do have a terminal node at the 0th level
		if(root->isTerminal) {
			pi=root->cm;
		}
		else if (testSet.matrix(obs, root->feature) < root->splitvalue) {
			//makePrediction leftDF
			pi = makePrediction(testSet, root->left, obs, verbose);
		} else {
			//makePrediction rightDF
			pi = makePrediction(testSet, root->right, obs, verbose);
		}
		if (!probability && !regression) {
			//cout<<"Rounding "<<pi<<" to "<<LUtils::round(pi)<<endl;
			pi = LUtils::round(pi);
		}
		p(obs) = pi;
		if (verbose) {
			cout << "p(" << obs << "): " << p(obs) << " order:"
					<< testSet.order[obs] << endl;
		}

	}
	//original order for p
	Eigen::VectorXd tmp(testSet.nrrows);
	int idx_orig = 0;
	for (int i = 0; i < testSet.nrrows; i++) {
		idx_orig = testSet.order.at(i);
		tmp(idx_orig) = p(i);
	}
	for (int i = 0; i < testSet.nrrows; i++) {
		p(i) = tmp(i);
	}
	//after prediction we have to re-establish the original order
	testSet.restoreOrder();
	if (verbose)
		cout << "###Tree size:" << tree_size + 1 << " nodes." << endl;
	return p;

}


//make prediction for a certain row, recursive until terminal node is reached
double Tree::makePrediction(const DataFrame &testset, const boost::shared_ptr<Node> localNode,
		const int obs, const bool verbose) {
	double t=0.0;
	if (localNode->isTerminal) {
		if (verbose) {
			cout << " cm(Node" << localNode->nodeID << "):";
		}
		return localNode->cm;
	}
	if (testset.matrix(obs, localNode->feature) < localNode->splitvalue) {
		//makePrediction leftDF
		  t = makePrediction(testset, localNode->left, obs, verbose);
	} else {
		//makePrediction rightDF
		  t =makePrediction(testset, localNode->right, obs, verbose);
	}
	//if we reach this end we go upwards again...
	return t;
}

Tree::~Tree() {
	//cout<<"###Deleting tree:"<<flush<<endl;
	//cerr<<"ROOT:"<<root->NODECOUNTER<<endl;
	//root->showChildren();
	//delete root->left;
	//delete root->right;
	//delete root;
}
