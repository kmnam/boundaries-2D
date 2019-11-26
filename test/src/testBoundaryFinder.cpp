#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <random>
#include <Eigen/Dense>
#include <autodiff/reverse/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include "../../include/boundaryFinder.hpp"
#include "../../include/linearConstraints.hpp"

/*
 * Test module for the BoundaryFinder class. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/25/2019
 */
using namespace Eigen;

std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(std::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
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
Matrix<T, Dynamic, 1> add_delta(const Ref<const Matrix<T, Dynamic, 1> >& x, std::mt19937& rng)
{
    Matrix<T, Dynamic, 1> y(x.size());
    T delta = 0.1;
    for (unsigned i = 0; i < x.size(); ++i)
    {
        int toss = coin_toss(rng);
        y(i) = (!toss ? x(i) + delta : x(i) - delta);
    }
    return y;
}

// -------------------------------------------------------- //
//                 BOUNDARY FINDER FUNCTIONS                //
// -------------------------------------------------------- //
void testBoundaryFinderProject()
{
    /*
     * Example 1A: projection onto the first two coordinates out of six,
     * constrained to the 6-D cube of side length 2
     */
    const unsigned seed = 1234567890;
    std::mt19937 rng(seed);
    const unsigned max_iter = 10;    // Maximum of 10 iterations 
    const double tol = 1e-5;         // Tolerance for convergence
    const unsigned n_init = 100;     // Initial number of parameter points

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
    BoundaryFinder<autodiff::var> finder(6, tol, rng, A, b);
    MatrixXd params = MatrixXd::Ones(n_init, 6) + MatrixXd::Random(n_init, 6);
    finder.run(
        project<autodiff::var>, add_delta<autodiff::var>, params,
        10,    // Maximum of 10 mutation iterations
        5,     // Maximum of 5 pulling iterations
        false, // Do not simplify the boundary
        true,  // Verbose output
        10,    // Maximum of 10 quadratic programs per SQP iteration
        1e-4,  // SQP convergence tolerance
        false, // Suppress SQP output
        "/n/groups/gunawardena/chris_nam/boundaries/test/project"
    );
}

void testBoundaryFinderProjectSimplified()
{
    /*
     * Example 1B: projection onto the first two coordinates out of six,
     * constrained to the 6-D cube of side length 2, with simplification
     */
    const unsigned seed = 1234567890;
    std::mt19937 rng(seed);
    const unsigned max_iter = 10;    // Maximum of 10 iterations 
    const double tol = 1e-5;         // Tolerance for convergence
    const unsigned n_init = 100;     // Initial number of parameter points

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
    BoundaryFinder<autodiff::var> finder(6, tol, rng, A, b);
    MatrixXd params = MatrixXd::Ones(n_init, 6) + MatrixXd::Random(n_init, 6);
    finder.run(
        project<autodiff::var>, add_delta<autodiff::var>, params,
        10,    // Maximum of 10 mutation iterations
        5,     // Maximum of 5 pulling iterations
        true,  // Simplify the boundary
        true,  // Verbose output
        10,    // Maximum of 10 quadratic programs per SQP iteration
        1e-4,  // SQP convergence tolerance
        false, // Suppress SQP output
        "/n/groups/gunawardena/chris_nam/boundaries/test/project-simplified"
    );
}

int main()
{
    testBoundaryFinderProject();
    testBoundaryFinderProjectSimplified();
    return 0;
}
