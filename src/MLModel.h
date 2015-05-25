//simple interface
//http://www.cplusplus.com/forum/general/107753/
#ifndef MLMODEL_H_
#define MLMODEL_H_

#include <Eigen/Core>
#include "DataFrame.h"

class MLModel
{
    public:
		//virtual MLModel();
        virtual ~MLModel();
        virtual void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const bool verbose = false)=0;
        virtual void train(const DataFrame &df, const bool verbose = false)=0;

        virtual Eigen::VectorXd predict(const Eigen::MatrixXd &Xtest, const bool verbose = false)=0;
        virtual Eigen::VectorXd predict(const DataFrame &df, const bool verbose = false)=0;
};

#endif
