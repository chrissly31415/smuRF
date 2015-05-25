/*
 * OnlineSGD.cpp
 *
 *  Created on: Jan 2, 2015
 *      Author: loschen
 */

#include "OnlineSGD.h"
#include <iostream>

OnlineSGD::OnlineSGD(){}

void OnlineSGD::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const bool verbose){
	cout<<"We train!!"<<endl;
}

void OnlineSGD::train(const DataFrame &df, const bool verbose){
	cout<<"We train!!"<<endl;
}

Eigen::VectorXd OnlineSGD::predict(const Eigen::MatrixXd &test, const bool verbose) {
	Eigen::VectorXd p;
	cout<<"We predict!!"<<endl;
	return p;
}

Eigen::VectorXd OnlineSGD::predict(const DataFrame &df, const bool verbose) {
	Eigen::VectorXd p;
	cout<<"We predict!!"<<endl;
	return p;
}

OnlineSGD::~OnlineSGD(){}
