#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "../../include/boundaryFinder.hpp"
#include "../../include/linearConstraints.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/10/2019
 */
using namespace Eigen;

int coin_toss(boost::random::mt19937& rng)
{
    boost::random::uniform_int_distribution<> dist(0, 1);
    return dist(rng);
}

// -------------------------------------------------------- //
//                      TEST FUNCTIONS                      //
// -------------------------------------------------------- //
template <typename T>
Matrix<T, Dynamic, 1> project(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    if (x.size() < 2) return Matrix<T, Dynamic, 1>::Zero(2);
    Matrix<T, Dynamic, 1> y(2);
    y << x(0), x(1);
    return y;
}

// -------------------------------------------------------- //
//                     MUTATION FUNCTIONS                   //
// -------------------------------------------------------- //
template <typename T>
Matrix<T, Dynamic, 1> add_delta(const Ref<const Matrix<T, Dynamic, 1> >& x, boost::random::mt19937& rng,
                                LinearConstraints* constraints)
{
    Matrix<T, Dynamic, 1> y(x.size());
    T delta = 0.01;
    for (unsigned i = 0; i < x.size(); ++i)
    {
        int toss = coin_toss(rng);
        if (!toss) y(i) = x(i) + delta;
        else       y(i) = x(i) - delta;
    }
    return constraints->nearestL2(y.template cast<double>()).template cast<T>();
}

int main()
{
    const unsigned seed = 1234567890;

    // Example 1: projection onto the first two coordinates out of six,
    // constrained to the 6-D cube of side length 2
    MatrixXd A(12, 6);
    VectorXd b(12);
    A <<  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  1.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  1.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  1.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  1.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  1.0,
         -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, -1.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, -1.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, -1.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, -1.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, -1.0;
    b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0;
    unsigned max_iter = 20;    // Maximum of 20 iterations 
    double tol = 1e-5;         // Tolerance for convergence
    unsigned n_init = 100;     // Initial number of parameter points
    BoundaryFinder finder(6, tol, max_iter, 1234567890, A, b);
    MatrixXd params = MatrixXd::Ones(n_init, 6) + MatrixXd::Random(n_init, 6);
    std::function<VectorXvar(const Ref<const VectorXvar>&)> func = project<var>;
    std::function<VectorXvar(const Ref<const VectorXvar>&, boost::random::mt19937&, LinearConstraints*)> mutate = add_delta<var>; 
    finder.run(func, mutate, params, false, true, "project");
    return 0;
}
