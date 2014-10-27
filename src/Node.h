/*
 * Node.h
 *
 *  Created on: Sep 25, 2012
 *      Author: Christoph Loschen
 */

#ifndef NODE_H_
#define NODE_H_

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

using namespace std;

struct Node {
	Node(int a, double b);
	Node(int a, double b, double c, int d, string e, int f, double h);
	~Node();
	int nodeID;
	int feature;
	double splitvalue;
	double impurity;
	int depth;
	string nameFeature;
	int nrsamples;
	double cm;

	boost::shared_ptr<Node> left;
	boost::shared_ptr<Node> right;
	//Node *left;
	//Node *right;
	bool isTerminal;
	void createChildren(int maxdepth, bool verbose=false);
	void showChildren(const bool verbose=false);
	string printChildren(const bool verbose=false);
	//should be tree variable...
	static int NODECOUNTER;

};



#endif /* NODE_H_ */
