#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/SQP.hpp"

/**
 * Test module for the `SQPOptimizer<double>` and `LineSearchSQPOptimizer<double>`
 * classes.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     10/15/2022
 */
using namespace Eigen;
double pi = 3.1415926539;

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
double F1(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.0, 2) + std::pow(x(1) - 2.5, 2);
}

/**
 * Nocedal and Wright, Example 12.6.
 */
double F2(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.5, 2) + std::pow(x(1) - 0.5, 4); 
}

/**
 * Nocedal and Wright, problem at top of page 467. 
 */
double F3(const Ref<const VectorXd>& x)
{
    return std::pow(x(0), 2) + std::pow(x(1) + 1.0, 2);
}

/**
 * 2-D Rosenbrock function.
 */
double Rosenbrock2D(const Ref<const VectorXd>& x)
{
    return std::pow(1.0 - x(0), 2) + 100.0 * std::pow(x(1) - std::pow(x(0), 2), 2);
}

/**
 * 3-D Rosenbrock function. 
 */
double Rosenbrock3D(const Ref<const VectorXd>& x)
{
    return std::pow(1.0 - x(0), 2) + std::pow(1.0 - x(1), 2) + 100.0 * (
        std::pow(x(1) - std::pow(x(0), 2), 2) + std::pow(x(2) - std::pow(x(1), 2), 2)
    ); 
}

BOOST_AUTO_TEST_CASE(TEST_SQP_F1)
{
    /**
     * Run the default SQP optimizer on F1().
     */
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const double c1 = 1e-4;
    const double c2 = 0.9;

    // Constraints in Nocedal and Wright, Example 16.4
    MatrixXd A(5, 2);
    A <<  1.0, -2.0,
         -1.0, -2.0,
         -1.0,  2.0,
          1.0,  0.0,
          0.0,  1.0;
    VectorXd b(5);
    b << -2.0, -6.0, -2.0, 0.0, 0.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 5, A, b);

    // Start at (2, 0); constraints 3 and 5 are active, with Lagrange  
    // multipliers -2 and -1
    VectorXd x(2);
    VectorXd l(5); 
    x << 2.0, 0.0; 
    l << 0.0, 0.0, -2.0, 0.0, -1.0;
    VectorXd solution = opt->run(
        F1, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 1.4) < tol);
    BOOST_TEST(abs(solution(1) - 1.7) < tol); 
    delete opt;

    std::cout << "Passed: TEST_SQP_F1" << std::endl; 
}

BOOST_AUTO_TEST_CASE(TEST_SQP_F2)
{
    /**
     * Run the default SQP optimizer on F2().
     */
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const int hessian_modify_max_iter = 1000;
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;

    // Constraints in Nocedal and Wright, Example 12.6
    MatrixXd A(4, 2);
    A << -1.0, -1.0,
         -1.0,  1.0,
          1.0, -1.0,
          1.0,  1.0;
    VectorXd b(4);
    b << -1.0, -1.0, -1.0, -1.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 4, A, b);

    // Start at (0.8, 0); all constraints are active 
    VectorXd x(2);
    VectorXd l(4);
    x << 0.8, 0.0;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        F2, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(TEST_SQP_F3)
{
    /**
     * Run the default SQP optimizer on F3().
     */
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const int hessian_modify_max_iter = 1000;
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;

    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2); 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 0.0, 0.0; 
    VectorXd solution = opt->run(
        F3, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(TEST_SQP_ROSENBROCK2D)
{
    /**
     * Run the default SQP optimizer on the 2-D Rosenbrock function.
     */
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const int hessian_modify_max_iter = 1000;
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;

    MatrixXd A(4, 2);
    A <<  1.0,  0.0,
          0.0,  1.0,
         -1.0,  0.0,
          0.0, -1.0;
    VectorXd b(4);
    b << 0.0, 0.0, -2.0, -2.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 4, A, b);
    VectorXd x(2); 
    VectorXd l(4);
    x << 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        Rosenbrock2D, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0,
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(TEST_SQP_ROSENBROCK3D)
{
    /**
     * Run the default SQP optimizer on the 3-D Rosenbrock function.
     */
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const int hessian_modify_max_iter = 1000;
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;

    MatrixXd A(6, 3);
    A <<  1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0,
         -1.0,  0.0,  0.0, 
          0.0, -1.0,  0.0,
          0.0,  0.0, -1.0;
    VectorXd b(6);
    b << 0.0, 0.0, 0.0, -2.0, -2.0, -2.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(3, 6, A, b);
    VectorXd x(3); 
    VectorXd l(6);
    x << 0.5, 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        Rosenbrock3D, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0,
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol);
    BOOST_TEST(abs(solution(2) - 1.0) < tol);  
    delete opt;
}
