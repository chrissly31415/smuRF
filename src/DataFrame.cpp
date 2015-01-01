/*
 * DataFrame.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: jars
 */

#include "DataFrame.h"
#include <algorithm>
#include <vector>
#include <set>
#include <iterator>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>

#include "LUtils.h"

using namespace std;

//Constructor
DataFrame::DataFrame(int a, int b, int c, bool r) :
	nrrows(a), nrcols(b), classCol(c), regression(r) {
	header.resize(nrcols);
	matrix.resize(nrrows, nrcols);
	order.resize(nrrows);
	type.resize(nrcols);
	distinct.resize(nrcols);
	y.resize(nrrows);
	cm = 0.0;

}

//empty constructor
DataFrame::DataFrame() {
}

void DataFrame::printData() {
	cout << setw(5) << "";
	for (int j = 0; j < nrcols; j++) {
		cout << setw(20) << header[j] << " ";
	}
	cout << endl;
	for (int i = 0; i < nrrows; i++) {
		cout << setw(5) << i;
		for (int j = 0; j < nrcols; j++) {
			cout << setw(20) << fixed << setprecision(2) << matrix(i, j) << " ";
		}
		cout << endl;
	}
}

void DataFrame::analyze() {
	for (int j = 0; j < nrcols; j++) {
		distinct[j] = nrDistinctValues(j);
		if (distinct[j] > factorlimit) {
			type[j] = "numeric";
		} else {
			//we set factorlimit=2
			type[j] = "binary";
		}
	}
	cm = matrix.col(classCol).sum();
	cm = cm / nrrows;
}

void DataFrame::quick_analyze() {
	cm = matrix.col(classCol).sum();
	cm = cm / nrrows;
}

DataFrame DataFrame::copy() {
	DataFrame *newDF = new DataFrame(nrrows, nrcols, classCol, regression);
	newDF->setHeader(header);
	newDF->setMatrix(matrix);
	newDF->setOrder(order);
	newDF->setType(type);
	newDF->setDistinct(distinct);
	if (newDF->nrrows < distinct_switch) {
		newDF->analyze();
	} else {
		newDF->quick_analyze();
	}
	return *newDF;
}

void DataFrame::setSortOrder() {
	for (int colnr = 0; colnr < nrcols; colnr++) {
		Eigen::VectorXd v = matrix.col(colnr);
		//we need a vector of pairs
		VecPairDoubInt vec(nrrows);
		//we make the pair to keep track of sorted indices
		for (int i = 0; i < nrrows; i++) {
			vec[i] = std::make_pair(v(i), i);
		}
		sort(vec.begin(), vec.end(), LUtils::pairComparator);
		int oldidx = 0;
		vector<int> tmpidx = order;
		MatrixXdcm tmp = matrix;
		for (int i = 0; i < nrrows; i++) {
			oldidx = vec[i].second;
			//save new order in temporary vector
			matrix.row(i) = tmp.row(oldidx);
			order[i] = tmpidx[oldidx];
		}
		printData();
	}
}

int DataFrame::nrDistinctValues(int column) {
	return distinctValues(column).size();
}

set<double> DataFrame::distinctValues(int column) {
	vector<double> v = LUtils::getColumn(matrix, column);
	set<double> uniqueSet(v.begin(), v.end());
	return uniqueSet;
}

void DataFrame::printSummary(int j, bool single) {
	if (single) {
		analyze();
		cout << "\nDataframe: Observations:" << nrrows << " Features:"
				<< nrcols << endl;
		cout << setw(34) << left << "Feature" << setw(10) << "MAX" << setw(10)
				<< "MIN" << setw(10) << "DISTINCT" << setw(10) << "MEDIAN"
				<< setw(10) << "Q25%" << setw(10) << "Q75%" << setw(10)
				<< "IQR" << setw(10) << "AVG" << setw(10) << "STDEV"
				<< setw(10) << "TYPE" << endl;
	}
	cout << setw(34) << left << header[j];
	cout << setprecision(2);
	cout << setw(10) << matrix.col(j).maxCoeff();
	cout << setw(10) << matrix.col(j).minCoeff();
	cout << setw(10) << distinct[j];
	//only with sufficient values
	vector<double> ktiles = getKTile(3, j, true);
	double q25 = ktiles[0];
	double median = ktiles[1];
	double q75 = ktiles[2];
	//interquartile range
	double iqr = q75 - q25;
	cout << setw(10) << median;
	cout << setw(10) << q25;
	cout << setw(10) << q75;
	cout << setw(10) << iqr;
	double sum = 0.0;
	for (int i = 0; i < matrix.rows(); ++i) {
		sum = sum + matrix(i, j);
	}
	double avg = sum / nrrows;
	cout << setw(10) << avg;
	//quick fix
	vector<double> v = LUtils::getColumn(matrix, j);
	double stdev = LUtils::calcStdev(v);
	cout << setw(10) << stdev;
	cout << setw(10) << type[j] << endl;
	cout << setprecision(2);
}

void DataFrame::printSummary() {
	if (nrrows < 1) {
		cout << "No data in dataframe." << endl;
		return;
	}
	//colwise
	for (int j = 0; j < nrcols; j++) {
		if (j == 0)
			printSummary(j, true);
		if (j > 0)
			printSummary(j, false);
	}
	//getKtile shuffles data.
	restoreOrder();
}

vector<double> DataFrame::getSplits(int k, int colnr, bool sort) {
	vector<double> ktile = DataFrame::getKTile(k, colnr, sort);
	//get max & min as well
	//Eigen::VectorXd v = matrix.col(colnr);
	//ktile.push_back(v.maxCoeff());
	//ktile.push_back(v.minCoeff());
	return ktile;
}

int DataFrame::containsFeature(string feat) {
	for (int i = 0; i < nrcols; ++i) {
		unsigned found = header[i].find(feat);
		if (found != std::string::npos)
			return i;
	}
	return -1;
}

//Calculates arbitrary quantiles
vector<double> DataFrame::getKTile(int k, int colnr, bool sort) {
	vector<double> v;
	if (sort) {
		v = LUtils::getSortedColumn(matrix, colnr);
	} else {
		v = LUtils::getColumn(matrix, colnr);
	}
	vector<double> ktile(k, 0.0);
	int index = 0;
	if (nrrows > k) {
		for (int j = 1; j <= k; j++) {
			index = j * nrrows / (k + 1);
			if (nrrows % 2 == 0) {
				ktile[j - 1] = 0.5 * (v[index] + v[index - 1]);
			} else {
				ktile[j - 1] = v[index];
			}
		}
		//now we know nrrows<=k, i.e. all fits in ktile, ktile.size>=v.size
	} else {
		for (int j = 0; j < nrrows; j++) {
			ktile[j] = v[j];
		}
	}
	return ktile;
}

void DataFrame::setHeader(const vector<string> &header) {
	this->header = header;
}

void DataFrame::setOrder(const vector<int> &order) {
	this->order = order;
}

void DataFrame::setType(const vector<string> &type) {
	this->type = type;
}

void DataFrame::setDistinct(const vector<int> &distinct) {
	this->distinct = distinct;
}

//splits dataframe by changing function arguments
void DataFrame::splitFrame(double split, int colnr, DataFrame &dfa,
		DataFrame &dfb) {
	//first determine matrix sizes
	int uc = 0, lc = 0;
	for (int i = 0; i < nrrows; i++) {
		if (matrix(i, colnr) < split) {
			uc++;
		} else {
			lc++;
		}
	}
	MatrixXdcm upper(uc, nrcols);
	MatrixXdcm lower(lc, nrcols);
	vector<int> orderu(uc);
	vector<int> orderl(lc);
	//now fill matrices
	uc = 0, lc = 0;
	for (int i = 0; i < nrrows; i++) {
		if (matrix(i, colnr) < split) {
			upper.row(uc) = this->matrix.row(i);
			orderu[uc] = uc;
			uc++;
		} else {
			lower.row(lc) << this->matrix.row(i);
			orderl[lc] = lc;
			lc++;
		}
	}
	DataFrame *upperDF = new DataFrame(uc, nrcols, classCol, regression);
	DataFrame *lowerDF = new DataFrame(lc, nrcols, classCol, regression);

	upperDF->setHeader(header);
	upperDF->setMatrix(upper);
	upperDF->setOrder(orderu);
	upperDF->setType(type);
	upperDF->setDistinct(distinct);
	if (upperDF->nrrows < distinct_switch) {
		upperDF->analyze();
	} else {
		upperDF->quick_analyze();
	}
	dfa = *upperDF;

	lowerDF->setHeader(header);
	lowerDF->setMatrix(lower);
	lowerDF->setOrder(orderl);
	lowerDF->setType(type);
	lowerDF->setDistinct(distinct);
	if (lowerDF->nrrows < distinct_switch) {
		lowerDF->analyze();
	} else {
		lowerDF->quick_analyze();
	}
	dfb = *lowerDF;
	delete upperDF;
	delete lowerDF;
}

//removes single column, e.g. train indicator
DataFrame DataFrame::removeColumn(const int col) {
	vector<int> cols(nrcols - 2);
	int oldref = 0;
	for (int i = 0; i < nrcols - 1; i++) {
		if (col == oldref)
			oldref++;
		if (oldref == classCol)
			continue;
		cols[i] = oldref;
		oldref++;
	}
	DataFrame df_reduced = DataFrame::getColumns(cols);
	return df_reduced;
}

//removes single column, e.g. train indicator
DataFrame DataFrame::removeColumn(const vector<int> &removecols) {
	vector<int> cols(nrcols - 1 - removecols.size());
	int oldref = 0;
	for (int i = 0; i < nrcols - 1; i++) {
		for (int j = 0; j < (int) removecols.size(); j++) {
			if (removecols[j] == oldref)
				oldref++;
		}
		if (oldref == classCol)
			continue;
		cols[i] = oldref;
		oldref++;
	}
	DataFrame df_reduced = DataFrame::getColumns(cols);
	return df_reduced;
}

//returns subset of features, return by value, class column is kept
DataFrame DataFrame::getColumns(const vector<int> &cols) {
	DataFrame df_reduced(nrrows, cols.size() + 1, cols.size(), regression);
	vector<string> header_reduced;
	MatrixXdcm data_reduced(nrrows, cols.size() + 1);
	for (unsigned int i = 0; i < cols.size(); i++) {
		if (cols[i] > (nrcols - 1)) {
			cerr << "ERROR: Could not create subset, column " << cols[i]
					<< " does not exists in dataframe with 0.." << this->nrcols
					<< " columns." << endl;
			exit(1);
		}
		if (cols[i] == classCol) {
			cerr << "ERROR: Class column can not be within subset." << endl;
			exit(1);
		}
		header_reduced.push_back(header[cols[i]]);
		for (int j = 0; j < nrrows; j++) {
			data_reduced(j, i) = matrix(j, cols[i]);
		}
	}
	//create class column
	header_reduced.push_back(header[classCol]);
	for (int j = 0; j < nrrows; j++) {
		data_reduced(j, cols.size()) = matrix(j, classCol);
	}
	df_reduced.setHeader(header_reduced);
	df_reduced.setMatrix(data_reduced);
	df_reduced.setOrder(order);
	df_reduced.setType(type);
	df_reduced.setDistinct(distinct);
	df_reduced.quick_analyze();
	return df_reduced;
}

//returns subset of features, return by value, we may optimize it later
//if sampled with replacement we have to reset ordering variable
DataFrame DataFrame::getRows(const vector<int> &rows, const bool resetOrder) {
	DataFrame df_reduced(rows.size(), nrcols, classCol, regression);
	MatrixXdcm data_reduced(rows.size(), nrcols);
	vector<int> order_red(rows.size());
	for (unsigned int i = 0; i < rows.size(); i++) {
		if (rows[i] > (nrrows - 1)) {
			cerr << "ERROR: Could not create subset, row " << rows[i]
					<< " does not exists in dataframe with 0.." << this->nrcols
					<< " rows." << endl;
			exit(1);
		}
		data_reduced.row(i) = matrix.row(rows[i]);
		if (!resetOrder) {
			order_red[i] = order[rows[i]];
			//old order makes no sense anymore if we have duplicate rows
		} else {
			order_red[i] = i;
		}

	}
	df_reduced.setHeader(header);
	df_reduced.setMatrix(data_reduced);
	df_reduced.setOrder(order_red);
	df_reduced.setType(type);
	df_reduced.setDistinct(distinct);
	df_reduced.quick_analyze();
	return df_reduced;
}

//sorting ALL rows according to proportion in propability in colnr
void DataFrame::sortRowsOnClassOutcome(int colnr) {
	//create vector with class proportion for each entry
}

//sorting ALL rows according to variables in colnr
void DataFrame::sortRowsOnColumn(int colnr) {
	Eigen::VectorXd v = matrix.col(colnr);
	//we need a vector of pairs
	VecPairDoubInt vec(nrrows);
	//we make the pair to keep track of sorted indices
	for (int i = 0; i < nrrows; i++) {
		vec[i] = std::make_pair(v(i), i);
	}
	sort(vec.begin(), vec.end(), LUtils::pairComparator);
	int oldidx = 0;
	vector<int> tmpidx = order;
	MatrixXdcm tmp = matrix;
	for (int i = 0; i < nrrows; i++) {
		oldidx = vec[i].second;
		//save new order in temporary vector
		matrix.row(i) = tmp.row(oldidx);
		order[i] = tmpidx[oldidx];
	}
}

//restores initial order of dataframe which may be shuffled to to variable ordering
void DataFrame::restoreOrder() {
	MatrixXdcm tmp(nrrows, nrcols);
	int idx_orig = 0;
	vector<int> tmporder(nrrows);
	for (int i = 0; i < nrrows; i++) {
		idx_orig = order[i];
		tmp.row(idx_orig) = matrix.row(i);
		tmporder[idx_orig] = i;
	}
	matrix = tmp;
	order = tmporder;
	//	for (int i = 0; i < nrrows; i++) {
	//		matrix.row(i) = tmp.row(i);
	//		order[i] = i;
	//	}

}

void DataFrame::setMatrix(const MatrixXdcm &matrix) {
	this->matrix = matrix;
}

//finds best feature
DataFrame::FeatureResult DataFrame::findBestFeature(
		const vector<int> &featList, const bool entropy_loss, double w1) {
	SplitResult best;
	int colnr = 0;
	double opt_loss = 0.0;
	double opt_split = 0.0;
	vector<double> loss;
	vector<string> feats;
	int opt_feat = 0;
	int distValues = 0;
	int ties = 0;
	bool skipped = true;
	//cout<<"Findbestfeature:"<<endl;
	for (unsigned i = 0; i < featList.size(); ++i) {
		colnr = featList[i];
		feats.push_back(header[colnr]);
		//cout<<"Feature: "<<header[colnr]<<endl;
		if (colnr == classCol) {
			continue;
		}
		if (nrrows > distinct_switch) {
			distValues = distinct[colnr];
		} else {
			distValues = nrDistinctValues(colnr);
		}
		if (distValues == 1) {
			best.first = matrix(0, colnr);
			best.second = 0.0;

		} else if (distValues > factorlimit) {
			best = findBestSplit_ktile(colnr, min(DataFrame::ktile_nr, nrrows),
					entropy_loss, w1);
			skipped = false;
		} else {
			best = findBestSplit_all(colnr, entropy_loss, w1);
			skipped = false;
		}

		loss.push_back(best.second);
		if (best.second <= opt_loss || i == 0) {
			opt_loss = best.second;
			opt_feat = colnr;
			opt_split = best.first;
		}
		if (best.second == opt_loss) {
			ties++;
		}

	}
	if (skipped == true) {
		//cout<<"No distinct values in variable: "<< header[colnr]<<" ..."<<endl;
		//exit(1);
	}

	FeatureResult retValue;
	retValue.opt_feat = opt_feat;
	retValue.opt_split = opt_split;
	retValue.loss = opt_loss;
	return retValue;
}

//uses ktile method of Chickering et al., for continous variables
DataFrame::SplitResult DataFrame::findBestSplit_ktile(const int colnr,
		const int k, const bool entropy_loss, double w1, bool verbose) {
	int classcol = classCol;
	//try different splits
	double opt_loss = 0.0;
	double opt_split = 0.0;
	double loss = 0.0;
	Eigen::VectorXd v = matrix.col(colnr);
	Eigen::VectorXd y = matrix.col(classcol);
	vector<double> ktiles = getSplits(k, colnr, false);
	double split = ktiles[0];
	for (unsigned i = 0; i < ktiles.size(); i++) {
		split = ktiles[i];
		if (regression) {
			loss = DataFrame::rmse_loss_direct(v, y, split, verbose);
		} else {
			loss = DataFrame::gini_loss_direct(v, y, split, entropy_loss, w1,
					verbose);
		}
		if (i < 1) {
			opt_loss = loss;
			opt_split = split;
		}
		if (loss < opt_loss) {
			opt_loss = loss;
			opt_split = split;
			if (verbose)
				cout << "New RMSE,opt: " << opt_loss << ", split feature: "
						<< header.at(colnr) << ", split value: " << opt_split
						<< endl;
		}
	}
	//we should return best_split und opt_idx
	SplitResult retValue = make_pair(opt_split, opt_loss);
	return retValue;
}

DataFrame::SplitResult DataFrame::findBestSplit_all(const int colnr,
		const bool entropy_loss, double w1, bool verbose) {
	int classcol = classCol;
	//feature values
	Eigen::VectorXd v = matrix.col(colnr);
	//class values
	Eigen::VectorXd y = matrix.col(classcol);
	//iterate over all unique! variables as possible split
	set<double> uniqueSet = distinctValues(colnr);
	double split = *uniqueSet.begin();
	double opt_loss = 10e15;
	double opt_split = split;
	double loss = 0.0;
	for (std::set<double>::const_iterator it = uniqueSet.begin(); it
			!= uniqueSet.end();) {
		if (it != uniqueSet.end() && std::next(it) != uniqueSet.end()) {
			split = (*it + *next(it)) / 2.0;
			//cout<<" SPLIT:"<<split<<endl;
		} else if (it == uniqueSet.begin()) {
			++it;
			continue;
		} else {
			break;
		}
		if (regression) {
			loss = DataFrame::rmse_loss_direct(v, y, split, verbose);
		} else {
			loss = DataFrame::gini_loss_direct(v, y, split, entropy_loss, w1,
					verbose);
		}
		if (loss < opt_loss || split == *boost::next(uniqueSet.begin())) {
			//verbose=true;
			opt_loss = loss;
			opt_split = split;
			if (verbose)
				cout << "New RMSE,opt: " << opt_loss << ", split feature: "
						<< header.at(colnr) << ", split value: " << opt_split
						<< endl;
		}
		++it;
	}
	SplitResult retValue = make_pair(opt_split, opt_loss);
	return retValue;
}

//finds average y according to binary split on continious variable
//prediction is basically just the average of the splitted frame
Eigen::VectorXd DataFrame::ave_y(int colnr, int ynr, double split, bool verbose) {
	double ylow = 0.0;
	double yhigh = 0.0;
	int splitindex = 0;
	Eigen::VectorXd v = matrix.col(colnr);
	//create predict vectors, GET UNSORTED VECTOR
	Eigen::VectorXd pred(v.size());
	Eigen::VectorXd y = matrix.col(ynr);
	for (int i = 0; i < v.size(); i++) {
		if (v(i) < split) {
			ylow = ylow + y(i);
			splitindex++;
		} else {
			yhigh = yhigh + y(i);
		}
	}
	ylow = ylow / double(splitindex);
	yhigh = yhigh / double(v.size() - splitindex);
	//assign predictions
	for (int i = 0; i < pred.size(); i++) {
		if (v(i) < split) {
			pred(i) = ylow;
		} else {
			pred(i) = yhigh;
		}
	}
	return pred;
}

//Assigns prediction according to split value for factor variable
Eigen::VectorXd DataFrame::class_prop(int colnr, int ynr, double split,
		bool verbose) {
	int splitindex = 0;
	double ysplit = 0.5;
	double pa0 = 0.0, pa1 = 0.0, pb0 = 0.0, pb1 = 0.0;
	//create predict vectors, GET UNSORTED VECTOR
	Eigen::VectorXd v = matrix.col(colnr);
	Eigen::VectorXd y = matrix.col(ynr);
	Eigen::VectorXd pred(nrrows);
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
	pa0 = pa0 / splitindex;
	pa1 = pa1 / splitindex;
	pb0 = pb0 / (nrrows - splitindex);
	pb1 = pb1 / (nrrows - splitindex);

	for (int i = 0; i < nrrows; ++i) {
		//region a
		if (v(i) < split) {
			//region a more probable for 1
			if (pa1 > pb1) {
				pred(i) = 1.0;
			} else {
				pred(i) = 0.0;
			}
			//region b
		} else {
			//region b most probable for 1
			if (pb1 > pa1) {
				pred(i) = 1.0;
			} else {
				pred(i) = 0.0;
			}
		}
	}
	return pred;
}

DataFrame::~DataFrame() {
}

