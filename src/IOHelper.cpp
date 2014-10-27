/*
 * IOHelper.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: Christoph Loschen
 */

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>

#include <vector>
#include <set>
#include <algorithm>
#include "LUtils.h"
#include "IOHelper.h"
#include "DataFrame.h"

Parameters IOHelper::parseParameters(string filename) {
	cout << "Parsing SETUP file" << endl;
	Parameters params;
	//setting default
	//params.nrtrees.push_back(100);
	//params.mtry.push_back(5);
	params.max_depth = 100;
	params.min_nodes = 5;
	params.probability = false;
	params.seed = 42;
	params.weight = 0.5;
	params.splitinfo.splitcolumn = -1;
	params.splitinfo.splitvalue = 0.5;
	params.verbose = 1;

	ifstream myfile;
	myfile.open(filename);
	string line;
	int tmpi;
	double tmpd;
	bool data_exists = false;
	//keywords
	boost::regex dataset("dataset\\s*=\\s*(.*)", boost::regex::icase);
	boost::regex remove("remove\\s*=\\s*([0-9]{0,})", boost::regex::icase);
	boost::regex split(
			"split\\s*=\\s*([0-9]{1,})\\s*,\\s*(-?[0-9]{0,}\\.[0-9]{0,})",
			boost::regex::icase);
	boost::regex protocol("job\\s*=\\s*(.*)", boost::regex::icase);
	boost::regex nrtrees("nrtrees\\s*=\\s*([0-9]{0,})", boost::regex::icase);
	boost::regex min_node("min_node\\s*=\\s*([0-9]{0,})", boost::regex::icase);
	boost::regex mtry("mtry\\s*=\\s*([0-9]{0,})", boost::regex::icase);
	boost::regex maxdepth("max_depth\\s*=\\s*([0-9]{0,})", boost::regex::icase);
	boost::regex probability("probability", boost::regex::icase);
	boost::regex seed("seed\\s*=\\s*([0-9]{0,9})", boost::regex::icase);
	boost::regex iter("iter\\s*=\\s*([0-9]{0,9})", boost::regex::icase);
	boost::regex loss("loss\\s*=\\s*(.*)", boost::regex::icase);
	boost::regex weight("weight1?\\s*=\\s*([0-9]{0,}\\.[0-9]{0,})",
			boost::regex::icase);
	boost::regex verbose("verbose\\s*=\\s*([0-9]{0,9})", boost::regex::icase);
	boost::regex comment("^#");
	boost::smatch matches;
	if (myfile.is_open()) {
		while (myfile.good()) {
			getline(myfile, line);
			if (boost::regex_search(line, matches, comment)) {
				continue;
			} else if (boost::regex_search(line, matches, dataset)) {
				params.dataset.push_back(matches[1]);
				data_exists = true;
			} else if (boost::regex_search(line, matches, loss)) {
				string tmp = matches[1];
				std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
				if (tmp.find("entropy") != std::string::npos) {
					params.entropy = true;
				} else {
					params.entropy = false;
				}
			} else if (boost::regex_search(line, matches, mtry)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.mtry.push_back(tmpi);
			} else if (boost::regex_search(line, matches, nrtrees)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.nrtrees.push_back(tmpi);
			} else if (boost::regex_search(line, matches, iter)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.iter = tmpi;
			} else if (boost::regex_search(line, matches, protocol)) {
				params.protocol.push_back(matches[1]);
			} else if (boost::regex_search(line, matches, probability)) {
				params.probability = true;
			} else if (boost::regex_search(line, matches, min_node)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.min_nodes = tmpi;
			} else if (boost::regex_search(line, matches, remove)) {
				//empty comment
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.remove.push_back(tmpi);
			} else if (boost::regex_search(line, matches, split)) {
				stringstream Str, Str2;
				Str << matches[1];
				Str >> tmpi;
				params.splitinfo.splitcolumn = tmpi;
				Str2 << matches[2];
				Str2 >> tmpd;
				params.splitinfo.splitvalue = tmpd;
			} else if (boost::regex_search(line, matches, weight)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpd;
				params.weight = tmpd;
			} else if (boost::regex_search(line, matches, maxdepth)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.max_depth = tmpi;
			} else if (boost::regex_search(line, matches, seed)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.seed = tmpi;
			} else if (boost::regex_search(line, matches, verbose)) {
				stringstream Str;
				Str << matches[1];
				Str >> tmpi;
				params.verbose = tmpi;
			}
		}
	}
	//default values
	if (!params.mtry.size() > 0) {
		params.mtry.push_back(5);
	}
	if (!params.nrtrees.size() > 0) {
		params.nrtrees.push_back(500);
	}
	myfile.close();
	if (data_exists == false) {
		cout << "ERROR: Please specify dataset!" << endl;
		exit(1);
	}
	return params;
}

void IOHelper::writePredictions(char* filename, const Eigen::VectorXd &p) {
	ofstream f;
	f.open(filename, ios::ate);
	for (int i = 0; i < p.size(); i++) {
		f << p(i) << endl;
	}
	f.close();
	cout << "Predictions written to: " << filename << endl;
}

void IOHelper::writePred2CSV(string filename, const DataFrame &df,
		const Eigen::VectorXd &p, bool header, bool truth) {
	Eigen::VectorXd y = df.matrix.col(df.classCol);
	ofstream f;
	f.open(filename, ios::ate);
	if (header && truth)
		f << "truth,predicted" << endl;
	if (header && !truth)
		f << "predicted" << endl;
	for (int i = 0; i < p.size(); i++) {
		if (truth)
			f << y(i) << "," << p(i) << endl;
		if (!truth)
			f << p(i) << endl;
	}
	f.close();
	cout << "Predictions in .csv format written to: " << filename << endl;
}

void IOHelper::writeColumns(string filename, const Eigen::VectorXd &a,
		const Eigen::VectorXd &b, string sep) {
	int n = min(a.size(), b.size());
	ofstream f;
	f.open(filename, ios::ate);
	f << "a,b" << endl;
	for (int i = 0; i < n; i++) {
		f << a(i) << sep << b(i) << endl;
	}
	f.close();

}

void IOHelper::writeDataframe2CSV(string filename, const DataFrame &df,
		const string separator, const bool header) {
	ofstream f;
	f.open(filename, ios::ate);
	if (header) {
		for (int i = 0; i < df.nrcols - 1; ++i) {
			f << df.header[i] << separator;
		}
		f << df.header[df.nrcols - 1] << endl;
	}

	for (int i = 0; i < df.nrrows; ++i) {
		for (int j = 0; j < df.nrcols - 1; ++j) {
			f << df.matrix(i, j) << separator;
		}
		f << df.matrix(i, df.nrcols - 1) << endl;
	}
	f.close();
	cout << "Dataframe in .csv format (separator: " << separator
			<< ")  written to: " << filename << endl;
}

void IOHelper::expandCategoricals(string infile, string outfile, bool lasty) {
	cout << endl << "###To do" << endl;
	ifstream myfile;
	string line;
	myfile.open(infile);
	if (myfile.is_open()) {
		getline(myfile, line);
		while (myfile.good()) {

		}
	}
	myfile.close();
}

//transforms categorical string variables into numeric ones, needs header, asumes class to be last column
void IOHelper::transformCSVfile(string infile, string outfile, bool lasty,
		bool verbose) {
	cout << endl << "###Substituting string variables with numerical values..."
			<< endl;
	ifstream myfile;
	stringstream sstr;
	bool hasHeader = true;
	bool headerParsed = false;
	int lcount = 0;
	string line;
	vector<string> fields;
	vector<set<string> > non_doubles;
	vector<map<string, double> > nonDoublesMap;
	vector<bool> isNonDoubleField;
	myfile.open(infile);
	if (!myfile) {
		cout << "ERROR: Could not find file " << infile << endl;
		exit(1);
	}
	//STEP 1 scan file & identify non doubles (identical to readcsv)
	if (myfile.is_open()) {
		getline(myfile, line);
		int nd_count = 0;
		while (myfile.good()) {
			boost::split_regex(fields, line, boost::regex(","));
			//check for string variables
			nd_count = 0;
			for (unsigned var = 0; var < fields.size(); ++var) {
				//now we are creating a set for each non double column
				if (lcount == 1 && !LUtils::checkForDouble(fields[var])) {
					isNonDoubleField.push_back(true);
					set<string> tmp;
					tmp.insert(fields[var]);
					non_doubles.push_back(tmp);
					//cout << "y:" << fields[fields.size() - 1] << endl;
					nd_count++;
				} else if (lcount == 1 && LUtils::checkForDouble(fields[var])) {
					isNonDoubleField.push_back(false);
				}
				//parse subsequent rows
				if (lcount > 1 && !LUtils::checkForDouble(fields[var])) {
					if (!isNonDoubleField[var]) {
						cout << "Error parsing dataset for value:"
								<< fields[var] << "..." << endl;
						exit(1);
					}
					//cout << "y:" << fields[fields.size() - 1] << endl;
					non_doubles.at(nd_count).insert(fields[var]);
					nd_count++;
				}
			}
			lcount++;
			getline(myfile, line);
		}
	}
	if (hasHeader == true) {
		lcount = lcount - 1;
		cout << "Parsing .csv file: Assuming header exists:";
	}
	cout << " Data: NROWS=" << lcount << " NCOLS=" << fields.size() << endl;
	//STEP 2: analyses of non double fields
	if (verbose)
		cout << "Transformation Legend: " << non_doubles.size()
				<< " non-numerical fields." << endl;
	int col = 0;
	for (vector<set<string> >::const_iterator it = non_doubles.begin(); it
			!= non_doubles.end(); ++it) {
		int unique_id = 0;
		//contains unique categories
		set<string> non_double = *it;
		//maps categories to doubles
		map<string, double> local_map;
		if (verbose)
			cout << left << setw(32) << "Category" << setw(10) << "Identifier"
					<< endl;
		for (set<string>::const_iterator it2 = non_double.begin(); it2
				!= non_double.end(); ++it2) {
			if (verbose)
				cout << left << setw(max(32, (int) it2->size())) << *it2
						<< setw(10) << unique_id << endl;
			//create map in first step
			if (col == 0) {
				map<string, double> local_map;
				local_map.insert(std::make_pair(*it2, unique_id));
				nonDoublesMap.push_back(local_map);
			} else {
				nonDoublesMap.at(col).insert(std::make_pair(*it2, unique_id));
			}
			unique_id++;
		}
		col++;
	}
	myfile.close();

	//STEP 3: parse file finally, also needed in readcsv
	DataFrame df(lcount, fields.size(), fields.size() - 1, true);
	myfile.open(infile);
	lcount = 0;
	double tmp;
	//parsing file row-wise
	if (myfile.is_open()) {
		getline(myfile, line);
		while (!myfile.eof()) {
			boost::split_regex(fields, line, boost::regex(","));
			if (hasHeader == true && headerParsed == false) {
				df.setHeader(fields);
				getline(myfile, line);
				headerParsed = true;
				continue;
			}
			//parsing data rowwise
			for (unsigned var = 0; var < fields.size(); ++var) {
				stringstream Str;
				Str << fields[var];
				Str >> tmp;
				//if is a non-double field convert to number
				if (isNonDoubleField[var]) {
					//...find right identifier
					for (vector<map<string, double> >::const_iterator it =
							nonDoublesMap.begin(); it != nonDoublesMap.end(); ++it) {
						map<string, double> local_map = *it;
						for (map<string, double>::const_iterator it2 =
								local_map.begin(); it2 != local_map.end(); ++it2) {
							if (it2->first.compare(fields[var]) == 0) {
								tmp = it2->second;
							}
						}
					}
				}
				df.matrix(lcount, var) = tmp;
			}
			df.order.at(lcount) = lcount;
			lcount++;
			getline(myfile, line);
		}
	}
	myfile.close();
	//determine if regression or classification
	vector<double> v = LUtils::getColumn(df.matrix, fields.size() - 1);
	set<double> uniqueSet(v.begin(), v.end());
	if (uniqueSet.size() > 2) {
		df.regression = true;
	} else {
		df.regression = false;
	}
	df.analyze();
	this->writeDataframe2CSV(outfile, df);
}

//return by value, we can optimize it later
DataFrame IOHelper::readCSVfile(string filename, bool entropy_loss, bool lasty) {
	ifstream myfile;
	stringstream sstr;
	bool hasHeader = true;
	bool headerParsed = false;
	bool string_found = false;
	int lcount = 0;
	string line;
	vector<string> fields;
	myfile.open(filename);
	if (!myfile) {
		cout << "ERROR: Could not find file " << filename << endl;
		exit(1);
	}
	//scan file & identify non doubles
	if (myfile.is_open()) {
		getline(myfile, line);
		while (myfile.good()) {
			boost::split_regex(fields, line, boost::regex(","));
			//check for string variables
			for (unsigned var = 0; var < fields.size(); ++var) {
				//now we are creating a set for each non double column
				if (lcount > 1 && !LUtils::checkForDouble(fields[var])
						&& string_found == false) {
					cout << line << endl;
					cout << "Row:" << lcount << " Columm:" << var << " Value->"
							<< fields[var] << "<-" << endl;
					cout
							<< "Warning: Dataset not mere numeric. Transformation of strings to doubles needed!"
							<< endl;
					string_found = true;
					//exit(1);
				}
				//parse subsequent rows
			}
			lcount++;
			getline(myfile, line);
		}
	}
	myfile.close();
	cout << endl;

	if (hasHeader == true) {
		lcount = lcount - 1;
		cout << "Parsing .csv file:" << filename << ". Assuming header exists:";
	}
	cout << " Data: NROWS=" << lcount << " NCOLS=" << fields.size() << endl;
	int targetcol = fields.size() - 1;
	DataFrame df(lcount, fields.size(), targetcol, true);

	//parse file finally
	myfile.open(filename);
	lcount = 0;
	double tmp;
	//parsing file row-wise
	if (myfile.is_open()) {
		getline(myfile, line);
		while (!myfile.eof()) {
			boost::split_regex(fields, line, boost::regex(","));
			if (hasHeader == true && headerParsed == false) {
				df.setHeader(fields);
				getline(myfile, line);
				headerParsed = true;
				continue;
			}
			//parsing data rowwise
			for (unsigned var = 0; var < fields.size(); ++var) {
				stringstream Str;
				Str << fields[var];
				Str >> tmp;
				df.matrix(lcount, var) = tmp;
				//we save data additionally in separate field
				if ((int) var == targetcol) {
					df.y(lcount) = tmp;
				}
			}
			//last column is target
			df.order.at(lcount) = lcount;
			lcount++;
			getline(myfile, line);
		}
	}
	myfile.close();
	//determine if regression or classification
	vector<double> v = LUtils::getColumn(df.matrix, fields.size() - 1);
	set<double> uniqueSet(v.begin(), v.end());
	if (uniqueSet.size() > 2) {
		df.regression = true;
	} else {
		df.regression = false;
	}
	df.analyze();
	return df;
}
