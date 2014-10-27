/*
 * Node.cpp
 *
 *  Created on: Sep 25, 2012
 *      Author: jars
 */

#include "Node.h"
#include "Tree.h"
#include "LUtils.h"
#include <iostream>
#include <boost/make_shared.hpp>

Node::Node(int a, double b) :
	depth(a), cm(b) {
	nodeID = NODECOUNTER;
	NODECOUNTER++;
	isTerminal = false;


}

Node::Node(int a, double b, double c, int d, string e, int f, double g) :
	feature(a), splitvalue(b), impurity(c), depth(d), nameFeature(e),
			nrsamples(f), cm(g) {
	nodeID = NODECOUNTER;
	NODECOUNTER++;
	isTerminal = false;
}

int Node::NODECOUNTER = 0;

//creates children nodes recursivly
void Node::createChildren(int maxdepth, bool verbose) {
	if (depth >= maxdepth) {
		if (verbose)
			cout << "TERMINAL NODE, MAXDEPTH reached...NO children for NODE"
					<< nodeID << "[DEPTH: " << depth << "]" << endl;
		isTerminal = true;
		return;
	} else {
		if (verbose)
			cout << "Creating children for NODE" << nodeID << "[DEPTH: "
					<< depth << "]" << endl;
		if (verbose)
			cout << "LEFTNODE for NODE" << nodeID << " [DEPTH: " << depth
					<< "]: ";
		//left = new Node(depth + 1, 0.0);
		left=boost::make_shared<Node>(depth + 1, 0.0);

		left->createChildren(maxdepth, verbose);
		if (verbose)
			cout << "RIGHTNODE for NODE" << nodeID << " [DEPTH: " << depth
					<< "]: ";
		//right = new Node(depth + 1, 0.0);
		right=boost::make_shared<Node>(depth + 1, 0.0);
		right->createChildren(maxdepth, verbose);
	}
}

//shows children nodes recursively
void Node::showChildren(const bool verbose) {
	string spacer = "";
	for (int i = 0; i < depth; ++i) {
		spacer = spacer + "\t";
	}
	if (isTerminal == false) {
		if (verbose) {
			cout << spacer << "NODEID:" << nodeID << " DEPTH: " << depth
					<< " Feature: " << nameFeature << "[" << nrsamples
					<< "] Splitvalue: " << splitvalue << " Loss: " << impurity;
		} else {
			cout << spacer << nameFeature << " Splitvalue: " << splitvalue;
		}
		cout << endl;
	} else if (isTerminal == true) {
		if (verbose) {
			cout << spacer << ">>NODEID:" << nodeID << "<< DEPTH: " << depth;
			cout << " TERMINAL" << " [" << nrsamples << "] y(x): " << cm;
			cout << endl;
		} else {
			cout << spacer << ">>TERMINAL : " << cm << endl;
		}
		return;
	}

	left->showChildren();
	right->showChildren();
}

//prints children nodes concisely into string stream
string Node::printChildren(const bool verbose) {
	stringstream ss;


	return ss.str();
}



Node::~Node() {

}

