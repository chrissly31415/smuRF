/*
 * Parameters.h
 *
 *  Created on: Feb 9, 2013
 *      Author: jars
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <vector>

using namespace std;

struct Split {
			double splitvalue;
			int splitcolumn;
		};

struct Parameters {
		vector<string> dataset;
		vector<string> testset;
		vector<string> protocol;
		vector<int> mtry;
		vector<int> nrtrees;
		vector<int> remove;
		//int nrtrees;
		//int mtry;
		int min_nodes;
		int max_depth;
		int iter;
		int verbose;
		int numjobs;
		unsigned int seed;
		Split splitinfo;
		bool entropy;
		bool probability;
		bool regression;
		double weight;
	};




#endif /* PARAMETERS_H_ */
