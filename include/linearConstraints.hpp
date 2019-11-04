#ifndef LINEAR_CONSTRAINTS_HPP
#define LINEAR_CONSTRAINTS_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/4/2019
 */
using namespace Eigen;

class LinearConstraints
{
    /*
     * A class that implements a set of linear constraints among a set
     * of variables, A * x >= b.
     */
    private:
        unsigned D;    // Number of variables
        unsigned N;    // Number of constraints
        MatrixXd A;    // Matrix of constraint coefficients
        VectorXd b;    // Matrix of constraint values

    public:
        LinearConstraints()
        {
            /* 
             * Empty constructor.
             */
            this->A = MatrixXd::Zero(0, 0);
            this->b = VectorXd::Zero(0);
            this->N = 0;
            this->D = 0;
        }

        LinearConstraints(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Constructor with supplied matrix and vector.
             */
            this->A = A;
            this->b = b;
            this->N = this->A.rows();
            this->D = this->A.cols();
            if (this->b.size() != this->N)
            {
                std::stringstream ss;
                ss << "Dimensions of A and b do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << this->b.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
        }

        ~LinearConstraints()
        {
            /*
             * Empty destructor.
             */
        }

        template <typename Derived>
        bool check(const MatrixBase<Derived>& x)
        {
            /* 
             * Return true if no constraints were satisfied or, otherwise,
             * if this->A * x >= this->b.
             */
            if (this->N == 0) return true;
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() >= (this->b).array()).all();
        }
};

#endif
