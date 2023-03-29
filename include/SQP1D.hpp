/**
 * A wrapper for a 1-D `SQPOptimizer` class instance. 
 * 
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     3/29/2023
 */

#ifndef SQP_OPTIMIZER_1D_HPP
#define SQP_OPTIMIZER_1D_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include "SQP.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

template <typename T>
class SQPOptimizer1D : public SQPOptimizer<T>
{
    public:
        /**
         * Trivial constructor. 
         * 
		 * The variable is constrained to be greater than or equal to zero.
         */
        SQPOptimizer1D() : SQPOptimizer<T>(1)
        {
        }

        /**
         * Constructor with bounds specified and inequality set to greater-
		 * than-or-equal-to.
		 *
         * @param lower Lower bound.
         * @param upper Upper bound.
         */
        SQPOptimizer1D(const T lower, const T upper) : SQPOptimizer<T>(1)
        {
            this->D = 1;
            this->N = 2;
			Matrix<T, Dynamic, Dynamic> A(2, 1);
		    A << 1, -1;
		    Matrix<T, Dynamic, 1> b(2);
		    b << lower, -upper;	
            this->A = A;
            this->b = b;
            this->constraints = new Polytopes::LinearConstraints(
                Polytopes::InequalityType::GreaterThanOrEqualTo,
                A.template cast<mpq_rational>(),
                b.template cast<mpq_rational>()
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = true; 
        }

        /**
         * Constructor with bounds and inequality type specified.
         *
         * @param D    Number of variables.
         * @param N    Number of constraints.
         * @param type Inequality type.
         * @param A    Constraint matrix.
         * @param b    Constraint vector.
         * @throws std::invalid_argument If the dimensions of `A` or `b` do not
         *                               match those implied by `D` and `N`.
         */
        SQPOptimizer1D(const T lower, const T upper, const Polytopes::InequalityType type)
			: SQPOptimizer<T>(1)
        {
            this->D = 1;
            this->N = 2;
			Matrix<T, Dynamic, Dynamic> A(2, 1);
			Matrix<T, Dynamic, 1> b(2);
			if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
			{
				A << 1, -1;
				b << lower, -upper;
			}
			else
			{
				A << 1, -1;
				b << upper, -lower;
			}
			this->A = A;
			this->b = b;
            this->constraints = new Polytopes::LinearConstraints(
                type,
                A.template cast<mpq_rational>(),
                b.template cast<mpq_rational>()
            ); 
            
			// Note that the inequality type for the internal quadratic program
            // should always be greater-than-or-equal-to 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = true; 
        }

        /**
         * Update the stored linear constraints.
         *
         * @param lower Lower bound.
         * @param upper Upper bound.
         */
        void setConstraints(const mpq_rational lower, const mpq_rational upper)
        {
			Matrix<mpq_rational, Dynamic, Dynamic> A(2, 1);
			Matrix<mpq_rational, Dynamic, 1> b(2);
			if (this->constraints->getInequalityType() == Polytopes::InequalityType::GreaterThanOrEqualTo)
			{
				A << 1, -1;
				b << lower, -upper;
			}
			else
			{
				A << 1, -1;
				b << upper, -lower;
			}
            this->constraints->setAb(A, b);
			this->A = A.template cast<T>();
			this->b = b.template cast<T>();
        }

        /**
         * Run the optimization with the given objective function, initial
         * vector for the objective function, initial vector of Lagrange 
         * multipliers, and additional settings.
         *
         * @param func                    Objective function (with scalar input).
         * @param quasi_newton            Quasi-Newton method.
         * @param regularize              Regularization method.
         * @param regularize_bases        Regularization base value.
         * @param regularize_weights      Regularization weight.
         * @param qp_solve_method         Quadratic program solution method.
         * @param x_init                  Initial iterate. 
         * @param l_init                  Initial vector of Lagrange multipliers.
         * @param delta                   Increment for finite difference 
         *                                approximation.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semidefiniteness).
         * @param min_stepsize            Minimum stepsize.
         * @param max_iter                Maximum number of steps.
         * @param tol                     Tolerance for change in objective value.
         * @param x_tol                   Tolerance for change in input vector 
         *                                (L2 norm between successive iterates).
         * @param qp_stepsize_tol         Tolerance for assessing whether a 
         *                                stepsize during each QP is zero. 
         * @param hessian_modify_max_iter Maximum number of Hessian modifications.
         * @param c1                      Constant multiplier in Armijo condition.
         * @param c2                      Constant multiplier in strong curvature
         *                                condition.
         * @param line_search_max_iter    Maximum number of iterations during 
         *                                line search.
         * @param zoom_max_iter           Maximum number of iterations in `zoom()`.
         * @param qp_max_iter             Maximum number of iterations during 
         *                                each QP. 
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param search_verbose          If true, output intermittent messages 
         *                                to `stdout` from `lineSearch()`.
         * @param zoom_verbose            If true, output intermittent messages 
         *                                to `stdout` from `zoom()`.
         * @returns Minimizing input value (as a scalar). 
         */
        T run(std::function<T(T)> func, const QuasiNewtonMethod quasi_newton,
			  const RegularizationMethod regularize, const T regularize_base,
			  const T regularize_weight, const QuadraticProgramSolveMethod qp_solve_method,
			  const T x_init, const Ref<const Matrix<T, Dynamic, 1> >& l_init,
			  const T delta, const T beta, const T min_stepsize, const int max_iter,
			  const T tol, const T x_tol, const T qp_stepsize_tol, 
			  const int hessian_modify_max_iter, const T c1, const T c2,
			  const int line_search_max_iter, const int zoom_max_iter,
			  const int qp_max_iter, const bool verbose = false,
			  const bool search_verbose = false, const bool zoom_verbose = false)
        {
			// Define vector versions of func, x_init, regularize_base, and
			// regularize_weights
			std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func_vec
				= [&func](const Ref<const Matrix<T, Dynamic, 1> >& x) -> T
				{
					return func(x(0));
				};
			Matrix<T, Dynamic, 1> x_init_vec(1); 
			x_init_vec << x_init;
			Matrix<T, Dynamic, 1> regularize_bases(1); 
			regularize_bases << regularize_base;
			Matrix<T, Dynamic, 1> regularize_weights(1);
			regularize_weights << regularize_weight;

			return this->SQPOptimizer<T>::run(
				func_vec, quasi_newton, regularize, regularize_bases, 
				regularize_weights, qp_solve_method, x_init_vec, l_init,
				delta, beta, min_stepsize, max_iter, tol, x_tol,
				qp_stepsize_tol, hessian_modify_max_iter, c1, c2,
				line_search_max_iter, zoom_max_iter, qp_max_iter, 
				verbose, search_verbose, zoom_verbose
			)(0);
        }
};

#endif 
