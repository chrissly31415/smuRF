/*
 * DataFrame.h
 *
 *  Created on: Sep 18, 2012
 *  Author: Christoph Loschen
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>

using namespace std;

struct DataFrame {
	int nrrows;
	int nrcols;
	int classCol;
	vector<string> header;
	typedef struct vector<pair<double, int> > VecPairDoubInt;
	//row major storing saves some time during ordering of the datasets
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
			Eigen::ColMajor> MatrixXdcm;

	MatrixXdcm matrix;
	Eigen::VectorXd y;

	vector<int> order;
	vector<string> type;
	vector<int> distinct;

	double cm;
	bool regression;

	//determines overall speed because it decides to  switch for split value search from a ktile subset -> to all features
	static const int factorlimit = 2;
	//how many qtiles to get
	static const int ktile_nr = 12;
	//when to recompute distinct values
	static const int distinct_switch = 50;
	//Constructor
	DataFrame(int a, int b, int c, bool r);
	DataFrame();
	virtual ~DataFrame();

	//Setter methods
	void setMatrix(const MatrixXdcm &matrix);
	void setHeader(const vector<string> &header);
	void setOrder(const vector<int> &order);
	void setType(const vector<string> &type);
	void setDistinct(const vector<int> &distinct);

	int nrDistinctValues(int column);
	set<double> distinctValues(int column);

	void setSortOrder();

	void analyze();
	void quick_analyze();
	void categoricalStatistics();

	void printData();
	void printSummary();
	void printSummary(int j,bool single=false);

	//data anaylsis & manipulation
	DataFrame getColumns(const vector<int> &cols);
	DataFrame getRows(const vector<int> &rows, const bool uniqueRows = true);
	DataFrame removeColumn(const int colnr);
	DataFrame removeColumn(const vector<int> &cols);

	DataFrame renameCategoricals();
	DataFrame shuffleCategoricals();
	int containsFeature(string feature);
	void sortRowsOnColumn(int colnr);
	void sortRowsOnClassOutcome(int colnr);

	void restoreOrder();

	typedef struct pair<DataFrame, DataFrame> DoubleDF;
	void splitFrame(double split, int colnr, DataFrame &dfa, DataFrame &dfb);

	vector<double> getKTile(int k, int colnr, bool sort = false);
	vector<double> getSplits(int k, int colnr, bool sort = false);

	struct FeatureResult {
		int opt_feat;
		double opt_split;
		double loss;
	};
	FeatureResult findBestFeature(const vector<int> &featList,const bool entropy_loss, const double w1=0.5);

	typedef struct pair<double, double> SplitResult;
	SplitResult findBestSplit_ktile(const int colnr, const int k, const bool entropy_loss, const double w1=0.5, bool verbose =
			false);
	SplitResult findBestSplit_all(const int colnr,const bool entropy_loss, const double w1=0.5, bool verbose = false);

	Eigen::VectorXd ave_y(const int colnr, const int ynr, const double split,
			bool verbose = false);
	Eigen::VectorXd class_prop(const int colnr, const int ynr,
			const double split, bool verbose = false);

	double rmse_loss(const Eigen::VectorXd &v,
				const Eigen::VectorXd &y, const double split, bool verbose = false);

	//STATIC FUNCTIONS
	//this is improving speed, short form
	static inline double rmse_loss_direct(const Eigen::VectorXd &v,
			const Eigen::VectorXd &y, const double split, bool verbose = false) {
		int nrrows = v.size();
		int splitindex = 0;
		double se_left = 0.0;
		double se_right = 0.0;
		//do only one loop for sum of sqaures and average
		for (int i = 0; i < nrrows; ++i) {
			if (v(i) < split) {
				se_left = se_left + y(i);
				splitindex++;
			} else {
				se_right = se_right + y(i);
			}
		}

		if (splitindex > 0) {
			se_left = (pow(se_left,2) / (double) splitindex);
		}
		if ((nrrows - splitindex) > 0) {
			se_right = (pow(se_right,2) / (double) (nrrows-splitindex));
		}
		return  -1.0/nrrows*(se_left + se_right);
	}

	//improving speed by inlining
	static inline double gini_loss_direct(const Eigen::VectorXd &v,
			const Eigen::VectorXd &y, const double split, bool entropy = true, double w1=0.5,
			bool verbose = false) {
		double  w0=1.0-w1;
		int nrrows = v.size();
		int splitindex = 0;
		double ysplit = 0.5;
		double pa0 = 0.0, pa1 = 0.0, pb0 = 0.0, pb1 = 0.0;
		//create predict vectors, GET UNSORTED VECTOR
		for (int i = 0; i < nrrows; ++i) {
			//left side
			if (v(i) < split) {
				//assign 0
				if (y(i) < ysplit) {
					pa0 = pa0 + 1.0;
					//assign 1
				} else {
					pa1 = pa1 + 1.0;
				}
				splitindex++;
			//right side, v(i)>=split
			} else {
				//assign 0
				if (y(i) < ysplit) {
					pb0 = pb0 + 1.0;
					//assign 1
				} else {
					pb1 = pb1 + 1.0;
				}
			}
		}
		if (splitindex > 0) {
			pa0 = w0* pa0 / splitindex;
			pa1 = w1* pa1 / splitindex;
		}
		if ((nrrows - splitindex) > 0) {
			pb0 = w0* pb0 / (nrrows - splitindex);
			pb1 = w1* pb1 / (nrrows - splitindex);
		}
		double wa = splitindex / (double) (nrrows);
		double loss = 0.0;
		if (!entropy) {
			loss = pa0 * (1 - pa0) * wa + pb1 * (1 - pb1) * (1.0 - wa);
		} else {
			//entropy
			loss = -pa0 * log(pa0 + 1.0E-15) * wa - pb0 * log(pb0 + 1.0E-15)
					* (1.0 - wa);
		}
		return loss;
	}

};

#endif /* DATAFRAME_H_ */
