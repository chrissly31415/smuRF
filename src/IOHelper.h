/*
 * IOHelper.h
 *
 *  Created on: Sep 16, 2012
 *      Author: Christoph Loschen
 */

#ifndef IOHELPER_H_
#define IOHELPER_H__H_

//#include "DataFrame.h"
//#include "Parameters.h"

using namespace std;

struct IOHelper {
	IOHelper() {
	}
	DataFrame readCSVfile(string filename);
	Parameters parseParameters(string filename);
	void transformCSVfile(string infile, string outfile, bool lasty=true, bool verbose=false);
	void expandCategoricals(string infile, string outfile, bool lasty=true);
	void writePred2CSV(string filename,const DataFrame &df, const Eigen::VectorXd &p, bool header=true, bool truth=false);
	void writePredictions(string filename, const Eigen::VectorXd &p);
	void writeDataframe2CSV(string filename,const DataFrame &df, const string separator=",", const bool header=true);
	void writeColumns(string filename,const Eigen::VectorXd &a, const Eigen::VectorXd &b, string sep);
	//Destruktor
	~IOHelper() {
	}
	//static DataFrame loadCSVfile(string filename, bool lasty=true);
};

#endif /* IOHELPER_H_ */
