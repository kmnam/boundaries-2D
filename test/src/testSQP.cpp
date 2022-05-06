#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/duals.hpp"
#include "../../include/SQP.hpp"

/**
 * Test module for the `SQPOptimizer<double>` and `LineSearchSQPOptimizer<double>`
 * classes.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     5/6/2022
 */
using namespace Eigen;
double pi = 3.1415926539;

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
double f1(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.0, 2) + std::pow(x(1) - 2.5, 2);
}

/**
 * Nocedal and Wright, Example 16.4, in terms of dual doubles.
 */
Dual<double> f1d(const Ref<const Matrix<Dual<double>, Dynamic, 1> >& x)
{
    return pow(x(0) - 1.0, 2) + pow(x(1) - 2.5, 2); 
}

/**
 * Nocedal and Wright, Example 12.6.
 */
double f2(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.5, 2) + std::pow(x(1) - 0.5, 4); 
}

/**
 * Nocedal and Wright, Example 12.6, in terms of dual doubles.
 */
Dual<double> f2d(const Ref<const Matrix<Dual<double>, Dynamic, 1> >& x)
{
    return pow(x(0) - 1.5, 2) + pow(x(1) - 0.5, 4); 
}

/**
 * Nocedal and Wright, problem at top of page 467. 
 */
double f3(const Ref<const VectorXd>& x)
{
    return std::pow(x(0), 2) + std::pow(x(1) + 1.0, 2);
}

/**
 * Nocedal and Wright, problem at top of page 467, in terms of dual doubles. 
 */
Dual<double> f3d(const Ref<const Matrix<Dual<double>, Dynamic, 1> >& x)
{
    return pow(x(0), 2) + pow(x(1) + 1.0, 2);
}

/**
 * 2-D Rosenbrock function.
 */
double rosenbrock2d(const Ref<const VectorXd>& x)
{
    return std::pow(1.0 - x(0), 2) + 100.0 * std::pow(x(1) - std::pow(x(0), 2), 2);
}

/**
 * 2-D Rosenbrock function in terms of dual doubles.
 */
Dual<double> rosenbrock2d_(const Ref<const Matrix<Dual<double>, Dynamic, 1> >& x)
{
    return pow(1.0 - x(0), 2) + 100.0 * pow(x(1) - pow(x(0), 2), 2);
}

/**
 * 3-D Rosenbrock function. 
 */
double rosenbrock3d(const Ref<const VectorXd>& x)
{
    return std::pow(1.0 - x(0), 2) + std::pow(1.0 - x(1), 2) + 100.0 * (
        std::pow(x(1) - std::pow(x(0), 2), 2) + std::pow(x(2) - std::pow(x(1), 2), 2)
    ); 
}

/**
 * 3-D Rosenbrock function in terms of dual doubles. 
 */
Dual<double> rosenbrock3d_(const Ref<const Matrix<Dual<double>, Dynamic, 1> >& x)
{
    return pow(1.0 - x(0), 2) + pow(1.0 - x(1), 2) + 100.0 * (
        pow(x(1) - pow(x(0), 2), 2) + pow(x(2) - pow(x(1), 2), 2)
    ); 
}

BOOST_AUTO_TEST_CASE(TEST_SQP_f1)
{
    /**
     * Run the default SQP optimizer on f1().
     */
    using std::abs;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

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
        f1, x, l, tau, delta, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.4) < tol);
    BOOST_TEST(abs(solution(1) - 1.7) < tol); 
    delete opt;

    std::cout << "Passed: TEST_SQP_f1" << std::endl; 
}

BOOST_AUTO_TEST_CASE(test_forward_autodiff_SQP_f1d)
{
    /**
     * Run the forward-mode automatic differentiation SQP optimizer on f1d().
     */
    using std::abs;
    const double tau = 0.5;
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    // Constraints in Nocedal and Wright, Example 16.4
    MatrixXd A(5, 2);
    A <<  1.0, -2.0,
         -1.0, -2.0,
         -1.0,  2.0,
          1.0,  0.0,
          0.0,  1.0;
    VectorXd b(5);
    b << -2.0, -6.0, -2.0, 0.0, 0.0;
    ForwardAutoDiffSQPOptimizer<double>* opt = new ForwardAutoDiffSQPOptimizer<double>(2, 5, A, b);

    // Start at (2, 0); constraints 3 and 5 are active, with Lagrange  
    // multipliers -2 and -1
    VectorXd x(2);
    VectorXd l(5); 
    x << 2.0, 0.0; 
    l << 0.0, 0.0, -2.0, 0.0, -1.0;
    VectorXd solution = opt->run(
        f1d, x, l, tau, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.4) < tol);
    BOOST_TEST(abs(solution(1) - 1.7) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_SQP_f2)
{
    /**
     * Run the default SQP optimizer on f2().
     */
    using std::abs;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

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
        f2, x, l, tau, delta, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_forward_autodiff_SQP_f2d)
{
    /**
     * Run the forward-mode automatic differentiation SQP optimizer on f2d().
     */
    using std::abs;
    const double tau = 0.5;
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    // Constraints in Nocedal and Wright, Example 12.6
    MatrixXd A(4, 2);
    A << -1.0, -1.0,
         -1.0,  1.0,
          1.0, -1.0,
          1.0,  1.0;
    VectorXd b(4);
    b << -1.0, -1.0, -1.0, -1.0;
    ForwardAutoDiffSQPOptimizer<double>* opt = new ForwardAutoDiffSQPOptimizer<double>(2, 4, A, b);

    // Start at (0.8, 0); all constraints are active 
    VectorXd x(2);
    VectorXd l(4);
    x << 0.8, 0.0;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        f2d, x, l, tau, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_SQP_f3)
{
    /*
     * Run the default SQP optimizer on f3().
     */
    using std::abs;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2); 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 0.0, 0.0; 
    VectorXd solution = opt->run(
        f3, x, l, tau, delta, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_forward_autodiff_SQP_f3d)
{
    /*
     * Run the forward-mode automatic differentiation SQP optimizer on f3d().
     */
    using std::abs;
    const double tau = 0.5;
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    ForwardAutoDiffSQPOptimizer<double>* opt = new ForwardAutoDiffSQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2); 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 0.0, 0.0; 
    VectorXd solution = opt->run(
        f3d, x, l, tau, beta, max_iter, tol, BFGS, use_strong_wolfe,
        hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_SQP_rosenbrock2d)
{
    /*
     * Run the default SQP optimizer on the 2-D Rosenbrock function.
     */
    using std::abs;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

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
        rosenbrock2d, x, l, tau, delta, beta, max_iter, tol, BFGS,
        use_strong_wolfe, hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_forward_autodiff_SQP_rosenbrock2d)
{
    /*
     * Run the forward-mode automatic differentiation SQP optimizer on the
     * 2-D Rosenbrock function.
     */
    using std::abs;
    const double tau = 0.5;
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    MatrixXd A(4, 2);
    A <<  1.0,  0.0,
          0.0,  1.0,
         -1.0,  0.0,
          0.0, -1.0;
    VectorXd b(4);
    b << 0.0, 0.0, -2.0, -2.0;
    ForwardAutoDiffSQPOptimizer<double>* opt = new ForwardAutoDiffSQPOptimizer<double>(2, 4, A, b);
    VectorXd x(2); 
    VectorXd l(4);
    x << 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        rosenbrock2d_, x, l, tau, beta, max_iter, tol, BFGS,
        use_strong_wolfe, hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_SQP_rosenbrock3d)
{
    /*
     * Run the default SQP optimizer on the 3-D Rosenbrock function.
     */
    using std::abs;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

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
        rosenbrock3d, x, l, tau, delta, beta, max_iter, tol, BFGS, 
        use_strong_wolfe, hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol);
    BOOST_TEST(abs(solution(2) - 1.0) < tol);  
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_forward_autodiff_SQP_rosenbrock3d)
{
    /*
     * Run the forward-mode automatic differentiation SQP optimizer on the
     * 3-D Rosenbrock function.
     */
    using std::abs;
    const double tau = 0.5;
    const double beta = 1e-4;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const int hessian_modify_max_iter = 1000;
    const bool use_strong_wolfe = false;

    MatrixXd A(6, 3);
    A <<  1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0,
         -1.0,  0.0,  0.0, 
          0.0, -1.0,  0.0,
          0.0,  0.0, -1.0;
    VectorXd b(6);
    b << 0.0, 0.0, 0.0, -2.0, -2.0, -2.0;
    ForwardAutoDiffSQPOptimizer<double>* opt = new ForwardAutoDiffSQPOptimizer<double>(3, 6, A, b);
    VectorXd x(3); 
    VectorXd l(6);
    x << 0.5, 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(
        rosenbrock3d_, x, l, tau, beta, max_iter, tol, BFGS,
        use_strong_wolfe, hessian_modify_max_iter, true
    );
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol);
    BOOST_TEST(abs(solution(2) - 1.0) < tol);  
    delete opt;
}
