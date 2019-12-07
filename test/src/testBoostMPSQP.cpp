#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/test/included/unit_test.hpp>
#include "../../include/SQP.hpp"

/*
 * Test module for the SQPOptimizer class with boost::multiprecision::mpfr scalars.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/6/2019
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::et_off;
typedef number<mpfr_float_backend<50>, et_off> mpfr_50_noet;
typedef Matrix<mpfr_50_noet, Dynamic, Dynamic> MatrixX50;
typedef Matrix<mpfr_50_noet, Dynamic, 1> VectorX50;

mpfr_50_noet f1(const Ref<const VectorX50>& x)
{
    /*
     * Nocedal and Wright, Example 16.4 (a convex quadratic program).
     */
    using boost::multiprecision::pow;
    return pow(x(0) - 1.0, 2) + pow(x(1) - 2.5, 2);
}

mpfr_50_noet f2(const Ref<const VectorX50>& x)
{
    /*
     * Nocedal and Wright, Example 12.6.
     */
    using boost::multiprecision::pow;
    return pow(x(0) - 1.5, 2) + pow(x(1) - 0.5, 4);
}

mpfr_50_noet f3(const Ref<const VectorX50>& x)
{
    /*
     * 2-D Rosenbrock function.
     */
    using boost::multiprecision::pow;
    return pow(1.0 - x(0), 2) + 100.0 * pow(x(1) - pow(x(0), 2), 2);
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
    SQPOptimizer<mpfr_50_noet>* opt = new SQPOptimizer<mpfr_50_noet>(2, 5, A, b);
    VectorX50 xl(7);
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
    SQPOptimizer<mpfr_50_noet>* opt = new SQPOptimizer<mpfr_50_noet>(2, 4, A, b);
    VectorX50 xl(6);
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
    SQPOptimizer<mpfr_50_noet>* opt = new SQPOptimizer<mpfr_50_noet>(2, 4, A, b);
    VectorX50 xl(6);
    xl << 0.5, 0.5, 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(f3, xl, 100, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}
