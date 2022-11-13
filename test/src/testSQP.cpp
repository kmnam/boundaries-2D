#define BOOST_TEST_MODULE testSQP
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../../include/SQP.hpp"

/**
 * Test module for the `SQPOptimizer<double>` class.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/13/2022
 */
using namespace Eigen;
using std::pow; 
using boost::multiprecision::pow;
using std::abs;
using boost::multiprecision::abs; 
double pi = 3.1415926539;

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
template <typename T>
T F1(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    return pow(x(0) - 1, 2) + pow(x(1) - T(5) / T(2), 2); 
}

/**
 * Nocedal and Wright, Example 12.6.
 */
template <typename T>
T F2(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    return pow(x(0) - T(3) / T(2), 2) + pow(x(1) - T(1) / T(2), 4); 
}

/**
 * Nocedal and Wright, Example 12.10.
 */
template <typename T>
T F3(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    return (x(0) * x(0) + x(1) * x(1)) / T(2); 
}

/**
 * Nocedal and Wright, problem at top of page 467. 
 */
template <typename T>
T F4(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    return pow(x(0), 2) + pow(x(1) + 1, 2);
}

/**
 * Nocedal and Wright, Example 19.1.
 */ 
template <typename T>
T F5(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    T c = T(1) / T(2);
    return std::pow(x(0) + c, 2) + std::pow(x(1) - c, 2); 
}

/**
 * N-dimensional Rosenbrock function.
 */
template <typename T>
T RosenbrockND(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    Array<T, Dynamic, 1> arr1 = (Matrix<T, Dynamic, 1>::Ones(x.size() - 1) - x(Eigen::seqN(0, x.size() - 1))).array().pow(2);
    Array<T, Dynamic, 1> arr2 = (x(Eigen::seqN(1, x.size() - 1)).array() - x(Eigen::seqN(0, x.size() - 1)).array().pow(2)).pow(2);
    return arr1.sum() + 100 * arr2.sum();
}

/**
 * Himmelblau function.
 */
template <typename T>
T Himmelblau(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    return pow(x(0) * x(0) + x(1) - 11, 2) + pow(x(0) + x(1) * x(1) - 7, 2);
}

/**
 * Run the default SQP optimizer on F1() with double scalars.
 */
TEST_CASE("optimize F1() with doubles")
{
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
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
        F1<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 1.4) < tol);
    REQUIRE(abs(solution(1) - 1.7) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F2().
 */
TEST_CASE("optimize F2() with doubles")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

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
        F2<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 1.0) < tol);
    REQUIRE(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F3().
 */
TEST_CASE("optimize F3() with doubles")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

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
        F3<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 1.0) < tol);
    REQUIRE(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F4().
 */
TEST_CASE("optimize F4() with doubles")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

    // Non-negativity constraints for both variables
    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x(2);    // Start at (1, 1); all constraints are inactive 
    VectorXd l(2); 
    x << 1.0, 1.0;
    l << 1.0, 1.0; 
    VectorXd solution = opt->run(
        F4<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 0.0) < tol);
    REQUIRE(abs(solution(1) - 0.0) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on F5().
 */
TEST_CASE("optimize F5() with doubles")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

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
        F5<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 0.0) < tol);
    REQUIRE(abs(solution(1) - 0.5) < tol); 
    delete opt;
}

/**
 * Run the default SQP optimizer on the 2-D, ..., 10-D Rosenbrock functions.
 */
TEST_CASE("optimize multi-dimensional Rosenbrock function with doubles")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

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
            RosenbrockND<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
            VectorXd::Zero(D), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
            x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
            hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
            qp_max_iter, true, true, true
        );
        for (int i = 0; i < D; ++i)
            REQUIRE(abs(solution(i) - 1.0) < tol);
        delete opt;
    }
}

/**
 * Run the default SQP optimizer on the Himmelblau function, with the input 
 * constrained to a box containing the origin in the first quadrant.
 */
TEST_CASE("optimize Himmelblau function within box in first quadrant")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

    // Non-negativity constraints for both variables 
    MatrixXd A = MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x = 5 * VectorXd::Ones(2);   // Start at (5, 5); both constraints are inactive
    VectorXd l = VectorXd::Ones(2);
    VectorXd solution = opt->run(
        Himmelblau<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - 3.0) < tol);
    REQUIRE(abs(solution(1) - 2.0) < tol);
    delete opt;
}

/**
 * Run the default SQP optimizer on the Himmelblau function, with the input 
 * constrained to a box containing the origin in the third quadrant.
 */
TEST_CASE("optimize Himmelblau function within box in third quadrant")
{
    using std::abs;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const double min_stepsize = 1e-8;
    const int max_iter = 1000; 
    const double tol = 1e-8;
    const double x_tol = 1e-10;
    const double qp_stepsize_tol = 1e-12;
    const int hessian_modify_max_iter = 1000;
    const int line_search_max_iter = 10;
    const int zoom_max_iter = 10;
    const int qp_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;

    // Non-positivity constraints for both variables 
    MatrixXd A = -MatrixXd::Identity(2, 2);
    VectorXd b = VectorXd::Zero(2);
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(2, 2, A, b);
    VectorXd x = -5 * VectorXd::Ones(2);    // Start at (-5, -5); all constraints are inactive 
    VectorXd l = VectorXd::Ones(2);
    VectorXd solution = opt->run(
        Himmelblau<double>, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG,
        VectorXd::Zero(2), QuadraticProgramSolveMethod::USE_CUSTOM_SOLVER,
        x, l, delta, beta, min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, true, true, true
    );
    REQUIRE(abs(solution(0) - (-3.7793102533777469)) < tol);
    REQUIRE(abs(solution(1) - (-3.2831859912861694)) < tol);
    delete opt;
}
