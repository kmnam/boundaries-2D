#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/SQP.hpp"

/**
 * Test module for the `SQPOptimizer<double>` class.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     4/11/2022
 */
using namespace Eigen;

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
double f1(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.0, 2) + std::pow(x(1) - 2.5, 2);
}

/**
 * Nocedal and Wright, Example 12.6.
 */
double f2(const Ref<const VectorXd>& x)
{
    return std::pow(x(0) - 1.5, 2) + std::pow(x(1) - 0.5, 4); 
}

/**
 * Nocedal and Wright, problem at top of page 467. 
 */
double f3(const Ref<const VectorXd>& x)
{
    return std::pow(x(0), 2) + std::pow(x(1) + 1, 2);
}

/**
 * 2-D Rosenbrock function.
 */
double rosenbrock(const Ref<const VectorXd>& x)
{
    return std::pow(1.0 - x(0), 2) + 100.0 * std::pow(x(1) - std::pow(x(0), 2), 2);
}

BOOST_AUTO_TEST_CASE(test_SQP_f1)
{
    /**
     * Run the default SQP optimizer on f1().
     */
    using std::abs;
    const double tol = 1e-8;

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
    VectorXd solution = opt->run(f1, x, l, 10, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.4) < tol);
    BOOST_TEST(abs(solution(1) - 1.7) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_linesearch_SQP_f1)
{
    /**
     * Run the line-search SQP optimizer on f1().
     */
    using std::abs;
    const double tol = 1e-8;

    // Constraints in Nocedal and Wright, Example 16.4
    MatrixXd A(5, 2);
    A <<  1.0, -2.0,
         -1.0, -2.0,
         -1.0,  2.0,
          1.0,  0.0,
          0.0,  1.0;
    VectorXd b(5);
    b << -2.0, -6.0, -2.0, 0.0, 0.0;
    LineSearchSQPOptimizer<double>* opt = new LineSearchSQPOptimizer<double>(2, 5, A, b);

    // Start at (2, 0); constraints 3 and 5 are active, with Lagrange  
    // multipliers -2 and -1
    VectorXd x(2); 
    VectorXd l(5);
    x << 2.0, 0.0;
    l << 0.0, 0.0, -2.0, 0.0, -1.0;
    VectorXd solution = opt->run(f1, x, l, 0.25, 0.5, 10, 1e-8, BFGS, true);
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
    const double tol = 1e-8;

    // Constraints in Nocedal and Wright, Example 12.6
    MatrixXd A(4, 2);
    A << -1.0, -1.0,
         -1.0,  1.0,
          1.0, -1.0,
          1.0,  1.0;
    VectorXd b(4);
    b << -1.0, -1.0, -1.0, -1.0;
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 4, A, b);

    // Start at (0.8, 0); all constraints are active, 
    VectorXd x(2);
    VectorXd l(4);
    x << 0.8, 0.0;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(f2, x, l, 10, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_linesearch_SQP_f2)
{
    /**
     * Run the line-search SQP optimizer on f2().
     */
    using std::abs;
    const double tol = 1e-8;

    // Constraints in Nocedal and Wright, Example 12.6
    MatrixXd A(4, 2);
    A << -1.0, -1.0,
         -1.0,  1.0,
          1.0, -1.0,
          1.0,  1.0;
    VectorXd b(4);
    b << -1.0, -1.0, -1.0, -1.0;
    LineSearchSQPOptimizer<double>* opt = new LineSearchSQPOptimizer<double>(2, 4, A, b);

    // Start at (0.8, 0); all constraints are active, 
    VectorXd x(2);
    VectorXd l(4);
    x << 0.8, 0.0;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(f2, x, l, 0.25, 0.5, 10, 1e-8, BFGS, true);
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
    const double tol = 1e-8;

    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2); 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 0.0, 0.0; 
    VectorXd solution = opt->run(f3, x, l, 10, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 0.0) < tol);
    BOOST_TEST(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_SQP_rosenbrock)
{
    /*
     * Run the default SQP optimizer on the 2-D Rosenbrock function.
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
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 4, A, b);
    VectorXd x(2); 
    VectorXd l(4);
    x << 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(rosenbrock, x, l, 100, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}

BOOST_AUTO_TEST_CASE(test_linesearch_SQP_rosenbrock)
{
    /*
     * Run the line-search SQP optimizer on the 2-D Rosenbrock function.
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
    LineSearchSQPOptimizer<double>* opt = new LineSearchSQPOptimizer<double>(2, 4, A, b);
    VectorXd x(2); 
    VectorXd l(4);
    x << 0.5, 0.5;
    l << 1.0, 1.0, 1.0, 1.0;
    VectorXd solution = opt->run(rosenbrock, x, l, 0.25, 0.5, 100, 1e-8, BFGS, true);
    BOOST_TEST(abs(solution(0) - 1.0) < tol);
    BOOST_TEST(abs(solution(1) - 1.0) < tol); 
    delete opt;
}
