
#ifndef ONLINESGD_H_
#define ONLINESGD_H_

#include "MLModel.h"

using namespace std;

class OnlineSGD : public MLModel
{
public:
	OnlineSGD();
	~OnlineSGD();
	virtual void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const bool verbose = false);
	virtual void train(const DataFrame &df, const bool verbose = false);

	virtual Eigen::VectorXd predict(const Eigen::MatrixXd &Xtest, const bool verbose = false);
	virtual Eigen::VectorXd predict(const DataFrame &df, const bool verbose = false);
};

#endif
