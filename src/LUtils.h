/*
 * LUtils.h
 *
 *  Created on: Sep 18, 2012
 *      Author: Christoph Loschen
 */

#ifndef LUTILS_H_
#define LUTILS_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <cstdlib>
#include <set>
#include <algorithm>
#include "DataFrame.h"
#include "RandomForest.h"
#include "RandomGen.h"
#include "Parameters.h"
#include <iostream>

using namespace std;

struct LUtils {

	typedef struct vector<pair<double, int> > VecPairDoubInt;

	template<class T>
	static void print(T& c) {
		unsigned count = 0;
		for (typename T::iterator i = c.begin(); i != c.end(); i++) {
			std::cout << "[" << count << "]" << *i << endl;
			count++;
		}
	}

	template<class T>
	static void print(T& c, T& d) {
		for (unsigned i = 0; i < c.size(); i++) {
			std::cout << "[" << i << "] " << setw(10) << c[i] << "-  [" << i
					<< "] " << setw(10) << d[i] << endl;
		}
	}

	//creates stratified samples
	static vector<vector<int> > strata_sample(const int k, const DataFrame &df,
			const int colnr, RandomGen rng) {
		vector<vector<int> > strata(k);
		//sampling create vector with indices and remove them one by one....
		vector<int> zero_idx;
		vector<int> ones_idx;
		//fill vector
		for (int i = 0; i < df.nrrows; ++i) {
			if (df.matrix(i, colnr) == 1) {
				ones_idx.push_back(i);
			} else {
				zero_idx.push_back(i);
			}
		}
		if (!df.regression) {
			cout << "Total fraction of     1: " << ones_idx.size()
					/ (double) df.nrrows << " [" << ones_idx.size() << "/"
					<< df.nrrows << "]" << endl;
			cout << "Total fraction of other: " << zero_idx.size()
					/ (double) df.nrrows << " [" << zero_idx.size() << "/"
					<< df.nrrows << "]" << endl;
		} else {
			cout << "Regression: Skipping strata creation." << endl;
		}
		//fill with 1 values
		while (ones_idx.size() > 0) {
			for (int i = 0; i < k; ++i) {
				if (ones_idx.size() == 0)
					continue;
				int tmp = rng.getRandomNumber(ones_idx.size());
				strata.at(i).push_back(ones_idx[tmp]);
				//cout<<strata.at(k)
				//swap value with last one
				int tmp2 = ones_idx.back();
				ones_idx.back() = ones_idx[tmp];
				ones_idx[tmp] = tmp2;
				//go trough each strata
				ones_idx.pop_back();
			}
		}
		//fill with 0 values
		while (zero_idx.size() > 0) {
			for (int i = 0; i < k; ++i) {
				if (zero_idx.size() == 0)
					continue;
				int tmp = rng.getRandomNumber(zero_idx.size());
				strata.at(i).push_back(zero_idx[tmp]);
				//cout<<strata.at(k)
				//swap value with last one
				int tmp2 = zero_idx.back();
				zero_idx.back() = zero_idx[tmp];
				zero_idx[tmp] = tmp2;
				//go trough each strata
				zero_idx.pop_back();
			}
		}
		//summarize percentage of y in each stratum
		return strata;
	}

	//create indices with/without replacement, keep distribution
	//n: number of return indices, bound: upper limit of indices
	static vector<int> sample(RandomGen &rng, int n, int bound,
			const Eigen::VectorXd &y, bool replacement) {
		//create histogram
		double minc = y.minCoeff();
		int nrsteps = 10;
		double step = (y.maxCoeff() - minc) / (double) nrsteps;
		//cout << "Intervall:" << step << endl;
		vector<vector<int> > hist(nrsteps);
		double low = minc;
		double high = minc + step;
		for (int j = 0; j < nrsteps; ++j) {
			for (int i = 0; i < y.rows(); ++i) {
				if (y(i) >= low && y(i) < high && j < nrsteps - 1) {
					hist.at(j).push_back(i);
					//closing last intervall, gets additional 10%
				} else if (y(i) >= low && y(i) <= high + step / 10.0 && j
						== nrsteps - 1) {
					hist.at(j).push_back(i);
				}
			}
			low = low + step;
			high = high + step;

		}
		vector<int> vec(n);
		for (int i = 0; i < n; ++i) {
			int tmp = rng.getRandomNumber(y.size());
			//cout<<"tmp:"<<tmp<<endl;
			int c = 0;
			for (unsigned j = 0; j < hist.size(); ++j) {
				c = c + hist.at(j).size();
				//cout << "Intervall " << j << " " << hist.at(j).size()<< " sum:"<<c<<endl;
				if (tmp < c) {
					tmp = rng.getRandomNumber(hist.at(j).size());
					//tmp=max(0,tmp);
					//cout<<"tmp:"<<tmp<<endl;
					//cout<<"Sampling from intervall:"<<j<<" index: "<<tmp-1<<" value:"<<hist.at(j).at(tmp)<<endl;
					vec[i] = hist.at(j).at(tmp);
					break;
				}

			}
		}
		return vec;
	}

	//create indices with/without replacement
	//n: number of return indices, bound: upper limit of indices
	static vector<int> sample(RandomGen &rng, int n, int bound,
			bool replacement) {
		if (n > bound && replacement == false) {
			cerr
					<< "Error: For sampling without replacement number of samples should not exceed bound."
					<< endl;
			exit(1);
		}
		vector<int> vec(n);
		int tmp = 0;
		bool exists;
		for (int i = 0; i < n; ++i) {
			do {
				//tmp = getRandomNumber(bound);
				tmp = rng.getRandomNumber(bound);
				if (replacement)
					break;
				exists = false;
				for (int j = 0; j < i; ++j) {
					if (tmp == vec[j]) {
						exists = true;
						break;
					}
				}
			} while (exists == true);
			vec[i] = tmp;
		}
		return vec;
	}

	static double getRandomNumber(int bound) {
		return rand() % bound;
	}

	//create indices with/without replacement, no rng object
	//n: number of return indices, bound: upper limit of indices
	static vector<int> sample(int n, int bound, bool replacement) {
		if (n > bound && replacement == false) {
			cerr
					<< "Error: For sampling without replacement number of samples should not exceed bound."
					<< endl;
			exit(1);
		}
		vector<int> vec(n);
		int tmp = 0;
		bool exists;
		for (int i = 0; i < n; ++i) {
			do {
				tmp = getRandomNumber(bound);
				if (replacement)
					break;
				exists = false;
				for (int j = 0; j < i; ++j) {
					if (tmp == vec[j]) {
						exists = true;
						break;
					}
				}
			} while (exists == true);
			vec[i] = tmp;
		}
		return vec;
	}

	//gets missing indices to create out of bag sample
	static vector<int> complement(vector<int> &inbag, int bound) {
		if ((int) inbag.size() > bound) {
			cerr
					<< "Error: For finding complement indices number of samples should not exceed bound."
					<< endl;
			exit(1);
		}
		set<int> uniqueSet(inbag.begin(), inbag.end());
		vector<int> vec(bound - uniqueSet.size());
		int index = 0;
		for (int i = 0; i < (int) vec.size(); ++i) {
			bool exists;
			do {
				exists = false;
				for (int j = 0; j < (int) inbag.size(); ++j) {
					if (index == inbag[j]) {
						exists = true;
						index++;
						continue;
					}
				}
			} while (exists == true);
			vec[i] = index;
			index++;
		}
		return vec;
	}

	static bool checkForDouble(std::string const& s) {
		std::istringstream ss(s);
		double d;
		return (ss >> d) && (ss >> std::ws).eof();
	}

	static Eigen::VectorXd fillPredictions(const Eigen::VectorXd &p,
			const vector<int> &indices, int size, vector<int> &oobcounter) {
		Eigen::VectorXd r(size);
		for (int i = 0; i < r.size(); ++i) {
			r(i) = 0.0;
		}
		for (int j = 0; j < p.size(); ++j) {
			r(indices.at(j)) = p(j);
			oobcounter[indices.at(j)]++;
		}
		return r;
	}

	//gets column of a matrix in STL format
	static vector<double> getColumn(const vector<vector<double> > &matrix,
			int colnr) {
		unsigned int nrows = matrix.size();
		vector<double> col(nrows);
		if (colnr + 1 > (int) matrix[0].size()) {
			cout << "ERROR: Colnr " << colnr
					<< " exceeding maximum column number." << endl;
			exit(1);
		}
		for (unsigned i = 0; i < nrows; i++) {
			col[i] = matrix[i][colnr];
		}
		return col;
	}

	static vector<double> getSortedColumn(const DataFrame::MatrixXdcm &matrix,
			int colnr) {
		//we need a vector of pairs
		vector<double> vec = LUtils::getColumn(matrix, colnr);
		sort(vec.begin(), vec.end());
		return vec;
	}

	static vector<double> getColumn(const DataFrame::MatrixXdcm &matrix,
			int colnr) {
		int nrows = matrix.rows();
		vector<double> col(nrows);
		if (colnr + 1 > (int) matrix.cols() + 1) {
			cout << "ERROR: Colnr " << colnr
					<< " exceeding maximum column number." << endl;
			exit(1);
		}
		for (int i = 0; i < nrows; i++) {
			col[i] = matrix(i, colnr);
		}
		return col;
	}

	static double calcStdev(vector<double> &v) {
		double stdev = 0.0;
		//get mean
		double sum = 0.0;
		for (vector<double>::iterator k = v.begin(); k != v.end(); ++k) {
			sum += *k;
		}
		double mean = sum / v.size();
		sum = 0.0;
		for (unsigned i = 0; i < v.size(); i++) {
			sum = sum + pow(v[i] - mean, 2);
		}
		stdev = sqrt(sum / v.size());
		return stdev;
	}

	//verbosity
	//level=0-2 (o minimum, 2 maximum)
	static double evaluate(const DataFrame &df, const Eigen::VectorXd &p,
			bool probability = false, int verbose = 0) {
		//final metric
		double loss = 0.0;
		Eigen::VectorXd y = df.matrix.col(df.classCol);
		if (y.size() != p.size()) {
			cout << "ERROR: Vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		if (df.regression) {
			if (verbose == 2) {
				for (int i = 0; i < y.size(); ++i) {
					cout << "T:" << y(i) << " P:" << p(i) << " SE:" << pow(
							y(i) - p(i), 2) << endl;
				}
			}
			loss = LUtils::loss_rmse2(y, p);
			if (verbose > 0) {
				double rsq = LUtils::loss_corrcoeff(y, p);
				printf("RMSE: %8.3f\n", loss);
				printf("MSE : %8.3f\n", pow(loss, 2));
				printf("R^2 : %8.3f\n", rsq);
			}
		} else {
			//confusion matrix
			int false_negativ = 0;
			int false_positiv = 0;
			int right_negativ = 0;
			int right_positiv = 0;
			for (int i = 0; i < y.size(); ++i) {
				if (verbose == 2) {
					if (probability)
						cout << "T:" << round(y(i)) << " P:" << p(i);
					else
						cout << "T:" << round(y(i)) << " P:" << round(p(i));
				}
				//false_negativ
				if (round(p(i)) == 0 && round(y(i)) == 1) {
					loss = loss + 1.0;
					false_negativ++;
					if (verbose == 2)
						cout << " #FN#" << endl;
					//false_positiv
				} else if (round(p(i)) == 1 && round(y(i)) == 0) {
					loss = loss + 1.0;
					false_positiv++;
					if (verbose == 2)
						cout << " #FP#" << endl;
					//right positiv
				} else if (round(p(i)) == 1 && round(y(i)) == 1) {
					right_positiv++;
					if (verbose == 2)
						cout << endl;
					//right negativ
				} else if (round(p(i)) == 0 && round(y(i)) == 0) {
					right_negativ++;
					if (verbose == 2)
						cout << endl;
				} else {
					cout << "ERROR in classification metric:" << "p:" << p(i)
							<< endl;
					exit(1);
				}
			}

			if (verbose >= 1) {
				cout << "Correctly classified:" << (y.size() - loss)
						<< " - false classified:" << loss << " (" << y.size()
						<< ")" << endl;
				cout << setw(16) << "Right positiv:" << setw(8)
						<< right_positiv;
				cout << setw(16) << "False positiv:" << setw(8)
						<< false_positiv << endl;
				cout << setw(16) << "False negativ:" << setw(8)
						<< false_negativ;
				cout << setw(16) << "Right negativ:" << setw(8)
						<< right_negativ << endl;

			}
			if (verbose > 1) {
				cout << setw(28) << "Correctly classified:" << setw(5)
						<< setprecision(2) << (1.0 - loss / y.size()) * 100.0
						<< "% (" << y.size() - loss << ")" << endl;
				cout << setw(28) << "Misclassification loss:" << loss
						/ y.size() * 100.0 << "% (" << loss << ") " << endl;

			}
			loss = loss / y.size();
		}
		return loss;
	}

	//simple round to next integer function
	static inline int round(double r) {
		return (int) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
	}

	//round vector
	static Eigen::VectorXd round(const Eigen::VectorXd &p) {
		Eigen::VectorXd r(p.size());
		for (int i = 0; i < p.size(); i++) {
			r(i) = round(p(i));
		}
		return r;
	}

	//gini loss
	static inline double gini_loss(const Eigen::VectorXd &y,
			const Eigen::VectorXd &p, bool verbose) {
		int n = p.size();
		if (y.size() != n) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		double p0 = 0.0, p1 = 0.0;
		int c0 = 0, ptemp = 0.0, ytemp = 0.0;
		for (int i = 0; i < n; i++) {
			ptemp = round(p(i));
			ytemp = round(y(i));
			//cout<<"p:"<<ptemp<<" y:"<<ytemp<<"\n";
			//class = 0
			if (ptemp == 0 && ytemp == 0) {
				p0 = p0 + 1.0;
			}
			//class = 1
			if (ptemp == 1 && ytemp == 1) {
				p1 = p1 + 1.0;
			}
			if (ptemp == 0) {
				c0++;
			}
		}
		if (c0 > 0)
			p0 = p0 / c0;
		if ((n - c0) > 0)
			p1 = p1 / (n - c0);
		double w0 = c0 / (double) (n);
		return (p0 * (1 - p0) * w0 + p1 * (1 - p1) * (1.0 - w0));
	}

	static inline double aucLoss(const Eigen::VectorXd &y,
			const Eigen::VectorXd &p, bool verbose = false) {
		int n = p.size();
		//make ranking first
		if (y.size() != n) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		int npos = (int) y.sum();
		int nneg = n - npos;
		VecPairDoubInt vec(n);
		//we make the pair to keep track of sorted indices
		for (int i = 0; i < n; i++) {
			vec[i] = make_pair(p(i), i);
		}
		sort(vec.begin(), vec.end(), LUtils::pairComparatorInv);
		//sensitivity true positive rate
		//specificity true negative rate = 1-false positive rate
		Eigen::VectorXd tpr(n);
		Eigen::VectorXd fpr(n);
		int pos = 0;
		int neg = 0;
		int ytemp = 0;
		double integ = 0.0;
		double bin = 0.0;
		for (int i = 0; i < n; i++) {
			ytemp = round(y(vec[i].second));
			if (ytemp == 1.0)
				pos++;
			if (ytemp == 0.0)
				neg++;
			tpr(i) = pos / (double) npos;
			fpr(i) = neg / (double) nneg;
			//simple trapezoidal like integration
			if (i > 1) {
				bin = fpr(i) - fpr(i - 1);
				if (bin > 1e-10) {
					integ = integ + bin * 0.5 * (tpr(i) + tpr(i - 1));
					//cout << "bin: " << bin << " int:" << integ << endl;
				}
			}

		}
		//		mean_tpr = mean_tpr / n;
		//		sq_tpr = sq_tpr / n;
		//		cout << "mean:" << mean_tpr << " +/-:" << sqrt(mean_tpr - sq_tpr) / n
		//				<< endl;
		if (verbose) {
			cout << setprecision(4) << "AUC:" << integ << endl;
			writeColumns("auc.csv", fpr, tpr, ";");
		}
		return integ;
	}

	static void writeColumns(string filename, const Eigen::VectorXd &x,
			const Eigen::VectorXd &y, string sep) {
		int n = min(x.size(), y.size());
		ofstream f;
		f.open(filename, ios::ate);
		f << "x" << sep << "y" << endl;
		for (int i = 0; i < n; i++) {
			f << x(i) << sep << y(i) << endl;
		}
		f.close();

	}

	//entropy loss
	static double entropy_loss(const Eigen::VectorXd &y,
			const Eigen::VectorXd &p, bool verbose) {
		int n = p.size();
		if (y.size() != n) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		double p0 = 0.0, p1 = 0.0;
		int c0 = 0, c1 = 0, ptemp = 0, ytemp = 0;
		double entropy = 0.0;
		for (int i = 0; i < n; i++) {
			ptemp = round(p(i));
			ytemp = round(y(i));
			//class = 0
			if (ptemp == 0 && ytemp == 0) {
				p0 = p0 + 1.0;
			}
			//class = 1
			if (ptemp == 1 && ytemp == 1) {
				p1 = p1 + 1.0;
			}
			if (ptemp == 0) {
				c0++;
			} else {
				c1++;
			}
		}
		if (c0 > 0) {
			p0 = p0 / (double) c0;
		}
		if (c1 > 0) {
			p1 = p1 / (double) c1;
		}
		double w0 = c0 / (double) (n);
		double w1 = c1 / (double) (n);
		entropy = -1.0
				* (p0 * log(p0 + 1e-15) * w0 + p1 * log(p1 + 1e-15) * w1);
		return entropy;
	}

	//misclassification loss
	static double missclass_loss(const Eigen::VectorXd &y,
			const Eigen::VectorXd &p, bool verbose) {
		if (y.size() != p.size()) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		double loss = 0.0;
		for (int i = 0; i < y.size(); ++i) {
			//false_negativ
			if (round(p(i)) == 0 && round(y(i)) == 1) {
				loss = loss + 1.0;
				//false_positiv
			} else if (round(p(i)) == 1 && round(y(i)) == 0) {
				loss = loss + 1.0;
				//right positiv
			} else if (round(p(i)) == 1 && round(y(i)) == 1) {
				//right negativ
			} else if (round(p(i)) == 0 && round(y(i)) == 0) {
			} else {
				cout << "ERROR in miss classification metric!" << endl;
				exit(1);
			}
		}
		return loss / y.size();
	}

	static double loss_rmse2(const Eigen::VectorXd &y, const Eigen::VectorXd &p) {
		if (y.size() != p.size()) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		double loss = 0.0;
		for (int i = 0; i < y.size(); ++i) {

			//	cout << "p(" << i << "): " << p(i) << " " << "y(" << i << "): "<< y(i) << endl;

			loss = loss + pow((y(i) - p(i)), 2);

		}
		loss = sqrt(loss / (double) y.size());
		return loss;
	}

	static double loss_corrcoeff(const Eigen::VectorXd &y,
			const Eigen::VectorXd &p) {
		if (y.size() != p.size()) {
			cout << "ERROR: vectors in error function of unequal length!"
					<< endl;
			exit(1);
		}
		//computing median and sdev
		double sumy_q = 0.0;
		double sump_q = 0.0;
		double sum = 0.0;
		double meany = y.sum() / (double) y.rows();
		double meanp = p.sum() / (double) p.rows();
		for (int i = 0; i < y.size(); ++i) {
			sum = sum + (y(i) - meany) * (p(i) - meanp);
			sumy_q = sumy_q + pow((y(i) - meany), 2);
			sump_q = sump_q + pow((p(i) - meanp), 2);
		}
		return sum / sqrt(sumy_q * sump_q);
	}

	struct column_comparer {
		int column_num;
		column_comparer(int c) :
			column_num(c) {
		}
		bool operator()(const vector<double> & lhs, const vector<double> & rhs) const {
			return lhs[column_num] < rhs[column_num];
		}
	};

	typedef std::pair<double, int> MyPair;

	static inline bool pairComparator(const MyPair& l, const MyPair& r) {
		return l.first < r.first;
	}

	static inline bool pairComparatorInv(const MyPair& l, const MyPair& r) {
		return l.first > r.first;
	}

	static void printDebug(string msg, bool debug) {
		if (debug == true) {
			cout << msg;
		}
	}

	static void printVerbose(string msg, bool verbose) {
		if (verbose == true) {
			cout << msg;
		}
	}

	//Function prints timing
	static void printTiming(timeval &start, timeval &end) {
		int difsec;
		int difusec;
		difsec = end.tv_sec - start.tv_sec;
		difusec = end.tv_usec - start.tv_usec;
		//cout << start.tv_sec << ':' << start.tv_usec << endl;
		//cout << end.tv_sec << ':' << end.tv_usec << endl;
		//cout <<"Simulation ended after: " << setprecision(6) << dif << " sec.";
		if (difusec < 0) {
			difsec--;
			difusec = 1000000 + difusec;
			//cout <<"modifying";
		}
		cout << fixed << setprecision(2);
		cout << endl << "RUN TIME: ";
		cout << difsec << "." << difusec << " sec" << " [" << difsec / 60.0
				<< " min]" << endl;
	}

	static void Xvalidation(int k, DataFrame df, RandomGen rng,
			Parameters params) {
		df.printSummary();
		vector<vector<int> > strata = LUtils::strata_sample(k, df, df.classCol,
				rng);
		vector<DataFrame> xvaldfs;
		double loss = 0.0;
		for (int i = 0; i < k; ++i) {
			DataFrame testDF = df.getRows(strata.at(i), true);
			//testDF.printSummary();
			vector<int> trainidx = LUtils::complement(strata.at(i), df.nrrows);
			DataFrame trainDF = df.getRows(trainidx, true);
			trainDF.printSummary(trainDF.classCol, true);
			RandomForest myRF(trainDF, rng, params);
			//myRF->printInfo();
			cout << "Fold " << i + 1 << ": ";
			myRF.growForest();
			Eigen::VectorXd p = myRF.predict(testDF);
			testDF.printSummary(testDF.classCol, true);
			loss = loss + LUtils::evaluate(testDF, p, false, false);
		}
		cout << endl << "XValdiation Summary:" << endl;
		if (!df.regression) {
			cout << setw(28) << "Correctly classified   :"
					<< (1.0 - (loss / k)) * 100 << "%" << endl;
			cout << setw(28) << "Missclassification loss:" << loss / k * 100
					<< "%" << endl;
		} else {
			cout << setw(8) << "<RMSE> :" << (loss / k) << "" << endl;
		}

	}

};

#endif /* LUTILS_H_ */
