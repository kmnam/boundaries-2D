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
 *     10/30/2022
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
 * Nocedal and Wright, Example 12.10.
 */
double F3(const Ref<const VectorXd>& x)
{
    return 0.5 * (x(0) * x(0) + x(1) * x(1)); 
}

/**
 * Nocedal and Wright, problem at top of page 467. 
 */
double F4(const Ref<const VectorXd>& x)
{
    return std::pow(x(0), 2) + std::pow(x(1) + 1.0, 2);
}

/**
 * Nocedal and Wright, Example 19.1.
 */ 
double F5(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) + 0.5, 2) + std::pow(x(1) - 0.5, 2); 
}

/**
 * N-dimensional Rosenbrock function.
 */
double RosenbrockND(const Ref<const VectorXd>& x)
{
    ArrayXd arr1 = (VectorXd::Ones(x.size() - 1) - x(Eigen::seqN(0, x.size() - 1))).array().pow(2);
    ArrayXd arr2 = (x(Eigen::seqN(1, x.size() - 1)).array() - x(Eigen::seqN(0, x.size() - 1)).array().pow(2)).pow(2);
    return arr1.sum() + 100 * arr2.sum();
}

/**
 * Himmelblau function.
 */
double Himmelblau(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) * x(0) + x(1) - 11, 2) + std::pow(x(0) + x(1) * x(1) - 7, 2);
}

/**
 * Run the default SQP optimizer on F1().
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_F1)
{
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

    // Start at (2, 0); constraints 3 and 5 are active (with Lagrange  
    // multipliers -2 and -1)
    VectorXd x(2);
    VectorXd l(5); 
    x << 2.0, 0.0; 
    l << 1.0, 1.0, 0.0, 1.0, 0.0;
    
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

/**
 * Run the default SQP optimizer on F2().
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_F2)
{
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

    // Start at (0, 1); first & third constraints are active, second & fourth 
    // constraints are not
    VectorXd x(2);
    VectorXd l(4);
    x << 0.0, 1.0;
    l << 0.0, 1.0, 0.0, 1.0;
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

/**
 * Run the default SQP optimizer on F3().
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_F3)
{
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

    // Constraint in Nocedal and Wright, Example 12.10, plus non-negativity for x(1)
    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b(2); 
    b << 1, 0;        // x(0) >= 1, x(1) >= 0
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2);    // Start at (1, 2); first constraint is active, second is not
    VectorXd l(2); 
    x << 1.0, 2.0;
    l << 0.0, 1.0; 
    VectorXd solution = opt->run(
        F4, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F4().
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_F4)
{
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

    // Non-negativity constraints for both variables
    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2);    // Start at (1, 1); all constraints are inactive 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 1.0, 1.0; 
    VectorXd solution = opt->run(
        F4, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F5().
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_F5)
{
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

    // Both variables are constrained to lie in [0, 1]
    MatrixXd A(4, 2);
    VectorXd b(4);
    A <<  1.0,  0.0,
          0.0,  1.0,
         -1.0,  0.0,
          0.0, -1.0;
    b << 0.0, 0.0, -1.0, -1.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 4, A, b);
    VectorXd x(2);    // Start at (1, 1); constraints 3 and 4 are active 
    VectorXd l(4); 
    x << 1.0, 1.0;
    l << 1.0, 1.0, 0.0, 0.0; 
    VectorXd solution = opt->run(
        F5, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0, 
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.5) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on the 2-D, ..., 10-D Rosenbrock functions.
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_ROSENBROCK_MULTIDIM)
{
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

    // All variables are constrained to lie in [0, 2]
    for (int D = 2; D <= 10; ++D)
    {
        MatrixXd A(2 * D, D);
        A(Eigen::seqN(0, D), Eigen::all) = MatrixXd::Identity(D, D); 
        A(Eigen::seqN(D, D), Eigen::all) = -MatrixXd::Identity(D, D); 
        VectorXd b = VectorXd::Zero(2 * D);
        b(Eigen::seqN(D, D)) = -2 * VectorXd::Ones(D); 
        SQPOptimizer<double>* opt = new SQPOptimizer<double>(D, 2 * D, A, b);
        VectorXd x = 0.5 * VectorXd::Ones(D);    // Start at (0.5, ..., 0.5); all constraints are inactive
        VectorXd l = VectorXd::Ones(2 * D);
        VectorXd solution = opt->run(
            RosenbrockND, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
            QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0,
            hessian_modify_max_iter, c1, c2, line_search_max_iter,
            zoom_max_iter, true, true, true
        );
        for (int i = 0; i < D; ++i)
            BOOST_TEST(abs(solution(i) - 1.0) < tol);
        delete opt;
    }
}

/**
 * Run the default SQP optimizer on the Himmelblau function, with the input 
 * constrained to a box containing the origin in the first quadrant.
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_HIMMELBLAU_POSQUADRANT)
{
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

    // Non-negativity constraints for both variables 
    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x = 5 * VectorXd::Ones(2);   // Start at (5, 5); both constraints are inactive
    VectorXd l = VectorXd::Ones(2);
    VectorXd solution = opt->run(
        Himmelblau, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0,
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - 3.0) < tol);
    BOOST_TEST(abs(solution(1) - 2.0) < tol);
    delete opt;
}

/**
 * Run the default SQP optimizer on the Himmelblau function, with the input 
 * constrained to a box containing the origin in the third quadrant.
 */
BOOST_AUTO_TEST_CASE(TEST_SQP_HIMMELBLAU_NEGQUADRANT)
{
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

    // Non-positivity constraints for both variables 
    MatrixXd A = -MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x = -5 * VectorXd::Ones(2);    // Start at (-5, -5); all constraints are inactive 
    VectorXd l = VectorXd::Ones(2);
    VectorXd solution = opt->run(
        Himmelblau, x, l, delta, beta, min_stepsize, max_iter, tol, x_tol,
        QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, 0,
        hessian_modify_max_iter, c1, c2, line_search_max_iter,
        zoom_max_iter, true, true, true
    );
    BOOST_TEST(abs(solution(0) - (-3.7793102533777469)) < tol);
    BOOST_TEST(abs(solution(1) - (-3.2831859912861694)) < tol);
    delete opt;
}
