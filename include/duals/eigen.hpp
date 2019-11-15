#ifndef DUAL_NUMBER_EIGEN_SUPPORT_HPP
#define DUAL_NUMBER_EIGEN_SUPPORT_HPP
#include <Eigen/Dense>
#include "duals.hpp"

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

}   // namespace Eigen

#endif
