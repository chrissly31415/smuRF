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
	// FORTRAN STYLE row major storing saves some time during ordering of the datasets
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
			Eigen::RowMajor> MatrixXdcm;

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
	static const int ktilenr = 12;
	//when to recompute distinct values
	static const int distinct_switch = 50;
	//Constructor
	DataFrame(int a, int b, int c, bool r);
	DataFrame();
	virtual ~DataFrame();

	//Setter methods
	void setParameters();
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
	DataFrame copy();

	int containsFeature(string feature);
	void sortRowsOnColumn(int colnr);
	void sortRowsOnClassOutcome(int colnr);

	void restoreOrder();

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

};

#endif /* DATAFRAME_H_ */
