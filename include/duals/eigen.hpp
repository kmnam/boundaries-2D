#ifndef DUAL_NUMBER_EIGEN_SUPPORT_HPP
#define DUAL_NUMBER_EIGEN_SUPPORT_HPP
#include <functional>
#include <Eigen/Dense>
#include "duals.hpp"

/*
 * Eigen support header file for the Duals::DualNumber class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */

namespace Eigen {

template<>
struct NumTraits<Duals::DualNumber> : NumTraits<double>
{
    typedef Duals::DualNumber Real;
    typedef Duals::DualNumber NonInteger;
    typedef Duals::DualNumber Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

// Convenience typedefs for DualNumber vectors and matrices
typedef Matrix<Duals::DualNumber, Dynamic, Dynamic> MatrixXDual;
typedef Matrix<Duals::DualNumber, Dynamic, 1>       VectorXDual;

}   // namespace Eigen

namespace Duals {

using namespace Eigen;

VectorXd gradient(std::function<DualNumber(const Ref<const VectorXDual>&)> func,
                  const Ref<const VectorXDual>& x, DualNumber& v)
{
    /*
     * Compute the gradient of the given multivariable function at the
     * given vector of values.  
     */
    v = func(x);    // Evaluate func at x and store result in v

    unsigned dim = x.size();
    VectorXDual xi(dim);
    VectorXd grad(dim);
    for (unsigned i = 0; i < dim; ++i)
    {
        // Set the nilpotent part to 1 for the i-th coordinate and 0 elsewhere
        for (unsigned j = 0; j < dim; ++j) xi(j) = x(j).x();
        xi(i) = DualNumber(x(i).x(), 1.0);

        // Compute the partial derivative w.r.t. the i-th coordinate
        grad(i) = func(xi).d();
    }
    return grad; 
}

}   // namespace Duals

#endif
