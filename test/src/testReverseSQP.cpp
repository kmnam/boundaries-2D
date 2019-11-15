#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <autodiff/reverse/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include <boost/test/included/unit_test.hpp>
#include "../../include/SQP.hpp"

/*
 * Test module for the SQPOptimizer class with reverse-mode variables.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */
using namespace Eigen;

autodiff::var f1(const Ref<const VectorXvar>& x)
{
    /*
     * Nocedal and Wright, Example 16.4 (a convex quadratic program).
     */
    return autodiff::pow(x(0) - 1.0, 2) + autodiff::pow(x(1) - 2.5, 2);
}

autodiff::var f2(const Ref<const VectorXvar>& x)
{
    /*
     * Nocedal and Wright, Example 12.6.
     */
    return autodiff::pow(x(0) - 1.5, 2) + autodiff::pow(x(1) - 0.5, 4);
}

autodiff::var f3(const Ref<const VectorXvar>& x)
{
    /*
     * 2-D Rosenbrock function.
     */
    return autodiff::pow(1.0 - x(0), 2) + 100.0 * autodiff::pow(x(1) - autodiff::pow(x(0), 2), 2);
}

BOOST_AUTO_TEST_CASE(test_f1)
{
    /*
     * Run the optimizer on f1().
     */
    using std::abs;
    const double tol = 1e-8;

    MatrixXd A(5, 2);
    A <<  1.0, -2.0,
         -1.0, -2.0,
         -1.0,  2.0,
          1.0,  0.0,
          0.0,  1.0;
    VectorXd b(5);
    b << -2.0, -6.0, -2.0, 0.0, 0.0;
    SQPOptimizer<autodiff::var>* opt = new SQPOptimizer<autodiff::var>(2, 5, A, b);
    VectorXvar xl(7);
    xl << 2.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0;
    VectorXd solution = opt->run(f1, xl, 10, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.4) < tol);
    BOOST_TEST(abs(solution(1) - 1.7) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_f2)
{
    /*
     * Run the optimizer on f2().
     */
    using std::abs;
    const double tol = 1e-8;

    MatrixXd A(4, 2);
    A << -1.0, -1.0,
         -1.0,  1.0,
          1.0, -1.0,
          1.0,  1.0;
    VectorXd b(4);
    b << -1.0, -1.0, -1.0, -1.0;
    SQPOptimizer<autodiff::var>* opt = new SQPOptimizer<autodiff::var>(2, 4, A, b);
    VectorXvar xl(6);
    xl << 0.8, 0.0, 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(f2, xl, 10, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_f3)
{
    /*
     * Run the optimizer on f3().
     */
    using std::abs;
    const double tol = 1e-8;

    MatrixXd A(4, 2);
    A <<  1.0,  0.0,
          0.0,  1.0,
         -1.0,  0.0,
          0.0, -1.0;
    VectorXd b(4);
    b << 0.0, 0.0, -2.0, -2.0;
    SQPOptimizer<autodiff::var>* opt = new SQPOptimizer<autodiff::var>(2, 4, A, b);
    VectorXvar xl(6);
    xl << 0.5, 0.5, 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(f3, xl, 100, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}
