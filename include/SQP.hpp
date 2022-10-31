/**
 * An implementation of a line-search nonlinear optimizer with respect to
 * linear constraints with sequential quadratic programming (SQP) with finite-
 * difference approximations or automatic differentiation for gradient
 * calculations, and quasi-Newton methods for Hessian approximations.
 *
 * Note that this implementation only deals with problems with *linear*
 * constraints (which define *convex* polytopes in the domain space).
 * This means that:
 *
 * - merit functions need not be used, since any candidate solution of the 
 *   form x_k + a_k * p_k, with stepsize a_k < 1 and direction p_k, satisfies
 *   all constraints if x_k + p_k does; and 
 * - inconsistent linearizations in the style of Nocedal and Wright, Eq. 18.12
 *   are not encountered.
 *
 * Stepsizes are determined to satisfy the (strong) Wolfe conditions, given by
 * Eqs. 3.6 and 3.7 in Nocedal and Wright.
 *
 * L1, L2, or elastic-net regularization may be incorporated into the
 * objective function if desired.  
 *
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     10/31/2022
 */

#ifndef SQP_OPTIMIZER_HPP
#define SQP_OPTIMIZER_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <linearConstraints.hpp>
#include "quasiNewton.hpp"
#include "linesearch.hpp"   // Includes Eigen/Dense, CGAL/QP_*, Boost.Multiprecision, boostMultiprecisionEigen.hpp, etc.
#include "duals.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

enum QuasiNewtonMethod
{
    NONE,
    BFGS,
    SR1,
};

template <typename T>
struct StepData
{
    public:
        Matrix<T, Dynamic, 1> xl;
        T f; 
        Matrix<T, Dynamic, 1> grad_f;
        Matrix<T, Dynamic, 1> grad_lagrangian;
        Matrix<T, Dynamic, Dynamic> hessian_lagrangian;
        T mu; 

        StepData()
        {
            /*
             * Empty constructor.
             */
        }

        ~StepData()
        {
            /*
             * Empty destructor.
             */
        }
};

/**
 * Following the prescription of Nocedal and Wright (Algorithm 3.3, p.51), 
 * add successive multiples of the identity until the given matrix is 
 * positive-definite.
 *
 * The input matrix is assumed to be symmetric.
 *
 * Note that this function is not necessary with the damped BFGS update 
 * proposed in Procedure 18.2.
 *
 * @param A        Input matrix.
 * @param max_iter Maximum number of modifications. 
 * @param beta     Multiple of the identity to add per modification.
 * @returns        Final modified matrix. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> modify(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                   int max_iter, T beta)
{
    // Check that A is positive-definite with the Cholesky decomposition
    LLT<Matrix<T, Dynamic, Dynamic> > dec(A);
    bool posdef = (dec.info() == Success);
    if (posdef)
        return A; 

    // If A is not positive-definite ...
    Matrix<T, Dynamic, Dynamic> B(A);
    T tau = 0.0;
    for (int i = 0; i < max_iter; ++i)
    {
        // Add beta to the diagonal ...
        if (tau == 0.0)
            tau = B.cwiseAbs().diagonal().minCoeff() + beta;
        else
            tau *= 2.0;
        B += tau * Matrix<T, Dynamic, Dynamic>::Identity(B.rows(), B.cols());
        dec.compute(B);

        // ... until the matrix is positive-definite
        posdef = (dec.info() == Success);

        if (posdef)
            break;    // If positive-definite, then we are done 
    }

    return B;
}

/**
 * An implementation of sequential quadratic programming for nonlinear
 * optimization on convex polytopes (i.e., linear inequality constraints). 
 */
template <typename T>
class SQPOptimizer
{
    protected:
        int D;                                                      /** Dimension of input space.      */
        int N;                                                      /** Number of constraints.         */
        Polytopes::LinearConstraints<mpq_rational>* constraints;    /** Linear inequality constraints. */
        Program* program;                                           /** Internal quadratic program to be solved at each step. */
        bool deallocate_constraints;                                /** Whether to deallocate constraints upon destruction.   */ 

        /**
         * Compute the gradient of the Lagrangian of an implicitly known objective
         * function at the given vector, with a pre-computed gradient for the 
         * objective.
         *
         * Note that this method does *not* require the function itself, as the
         * calculation only requires the gradient vector and the constraints. 
         *
         * This method can only be called internally. 
         *
         * @param x     Input vector for the implicitly known objective.
         * @param l     Input vector for additional Lagrangian variables (one
         *              for each constraint).
         * @param grad  Pre-computed gradient vector of the objective at `x`.
         * @returns     Gradient approximation.
         */
        Matrix<T, Dynamic, 1> lagrangianGradient(const Ref<const Matrix<T, Dynamic, 1> >& x,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& l,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& grad)
        {
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>();   // Convert from rationals to T
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();         // Convert from rationals to T
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            T sign = (type == Polytopes::InequalityType::GreaterThanOrEqualTo ? -1 : 1); 

            // Initialize the gradient vector
            Matrix<T, Dynamic, 1> grad_lagrangian = Matrix<T, Dynamic, 1>::Zero(this->D + this->N);
            grad_lagrangian.head(this->D) = grad;

            // Incorporate the contributions of the *linear* constraints to 
            // each partial derivative of the Lagrangian 
            for (int i = 0; i < this->D; ++i)
                grad_lagrangian(i) += sign * A.col(i).dot(l);
            for (int i = 0; i < this->N; ++i)
                grad_lagrangian(this->D + i) = sign * (A.row(i).dot(x) - b(i));  
            
            return grad_lagrangian;
        }

    public:
        /**
         * Straightforward constructor with `D` variables.
         *
         * Each variable is constrained to be greater than or equal to zero.
         *
         * @param D Number of variables.
         */
        SQPOptimizer(const int D)
        {
            this->D = D;
            this->N = D;    // One constraint for each variable
            Matrix<mpq_rational, Dynamic, Dynamic> A = Matrix<mpq_rational, Dynamic, Dynamic>::Identity(D, D); 
            Matrix<mpq_rational, Dynamic, 1> b = Matrix<mpq_rational, Dynamic, 1>::Zero(D); 
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = true; 
        }

        /**
         * Constructor with constraint matrix and vector specified, and
         * inequality set to greater-than-or-equal-to.
         *
         * @param D Number of variables.
         * @param N Number of constraints.
         * @param A Constraint matrix.
         * @param b Constraint vector.
         * @throws std::invalid_argument If the dimensions of `A` or `b` do not
         *                               match those implied by `D` and `N`.
         */
        SQPOptimizer(const int D, const int N,
                     const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                     const Ref<const Matrix<T, Dynamic, 1> >& b)
        {
            this->D = D;
            this->N = N;
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                Polytopes::InequalityType::GreaterThanOrEqualTo,
                A.template cast<mpq_rational>(),
                b.template cast<mpq_rational>()
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = true; 
        }

        /**
         * Constructor with constraint matrix, vector, and inequality type
         * specified.
         *
         * @param D    Number of variables.
         * @param N    Number of constraints.
         * @param type Inequality type.
         * @param A    Constraint matrix.
         * @param b    Constraint vector.
         * @throws std::invalid_argument If the dimensions of `A` or `b` do not
         *                               match those implied by `D` and `N`.
         */
        SQPOptimizer(const int D, const int N, const Polytopes::InequalityType type, 
                     const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                     const Ref<const Matrix<T, Dynamic, 1> >& b)
        {
            this->D = D;
            this->N = N;
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                type, A.template cast<mpq_rational>(), b.template cast<mpq_rational>()
            ); 

            // Note that the inequality type for the internal quadratic program
            // should always be greater-than-or-equal-to 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = true; 
        }

        /**
         * Constructor with linear constraints specified via a `LinearConstraints`
         * instance.
         *
         * @param constraints Pointer to `LinearConstraints` instance containing
         *                    the constraint matrix and vector.
         */
        SQPOptimizer(Polytopes::LinearConstraints<mpq_rational>* constraints)
        {
            this->constraints = constraints;
            this->D = this->constraints->getD(); 
            this->N = this->constraints->getN(); 
            
            // Note that the inequality type for the internal quadratic program
            // should always be greater-than-or-equal-to 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
            this->deallocate_constraints = false;
        }

        /**
         * Destructor; deallocates the `LinearConstraints` and `Program`
         * instances.
         *
         * The former is not deallocated if its creation was not tied to the 
         * lifetime of `this`.
         */
        ~SQPOptimizer()
        {
            if (this->deallocate_constraints)
                delete this->constraints;
            delete this->program;
        }

        /**
         * Update the stored linear constraints.
         *
         * @param A New constraint matrix.
         * @param b New constraint vector.
         * @throws std::invalid_argument If the dimensions of `A` or `b` do not
         *                               match those implied by `this->D` and
         *                               `this->N`.
         */
        void setConstraints(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& A,
                            const Ref<const Matrix<mpq_rational, Dynamic, 1> >& b)
        {
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints->setAb(A, b);
        }

        /**
         * Compute the gradient of the given function at the given vector, with 
         * increment `delta` for finite difference approximation.
         *
         * @param func  Function whose gradient is to be approximated.
         * @param x     Input vector.
         * @param delta Increment for finite difference approximation.
         * @returns Gradient approximation. 
         */
        Matrix<T, Dynamic, 1> gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                       const Ref<const Matrix<T, Dynamic, 1> >& x,
                                       const T delta)
        {
            // Evaluate the function at 2 * D values, with each coordinate 
            // perturbed by +/- delta
            Matrix<T, Dynamic, 1> grad(this->D);
            Matrix<T, Dynamic, 1> x_(x); 
            for (int i = 0; i < this->D; ++i)
            {
                x_(i) += delta; 
                T f1 = func(x_); 
                x_(i) -= 2 * delta; 
                T f2 = func(x_); 
                x_(i) += delta; 
                grad(i) = (f1 - f2) / (2 * delta); 
            }

            return grad;  
        }

        /**
         * Compute the gradient of the Lagrangian of the given function at the 
         * given vector, with increment `delta` for finite difference approximation.
         *
         * @param func  Input function.
         * @param x     Input vector for `func`.
         * @param l     Input vector for additional Lagrangian variables (one
         *              for each constraint).
         * @param delta Increment for finite difference approximation.
         * @returns     Gradient approximation.
         */
        Matrix<T, Dynamic, 1> lagrangianGradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& x,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& l,
                                                 const T delta)
        {
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>();   // Convert from rationals to T
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();         // Convert from rationals to T
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            T sign = (type == Polytopes::InequalityType::GreaterThanOrEqualTo ? -1 : 1); 

            // Evaluate the function at 2 * D values, with each coordinate
            // perturbed by +/- delta, using gradient()
            Matrix<T, Dynamic, 1> grad_lagrangian = Matrix<T, Dynamic, 1>::Zero(this->D + this->N);
            Matrix<T, Dynamic, 1> grad_f = this->gradient(func, x, delta); 
            grad_lagrangian.head(this->D) += grad_f;

            // Incorporate the contributions of the *linear* constraints to 
            // each partial derivative of the Lagrangian 
            for (int i = 0; i < this->D; ++i)
                grad_lagrangian(i) += sign * A.col(i).dot(l);
            for (int i = 0; i < this->N; ++i)
                grad_lagrangian(this->D + i) = sign * (A.row(i).dot(x) - b(i));  
            
            return grad_lagrangian;
        }


        /**
         * Run one step of the SQP algorithm.
         *
         * 1) Given an input vector `(x,l)`, compute `f(x)` and `df(x)/dx`. 
         * 2) Compute the Lagrangian, `L(x,l) = f(x) - l.T * A * x`, where `A` is the
         *    constraint matrix, and its Hessian matrix of second derivatives w.r.t. `x`.
         *    - Use a quasi-Newton method to compute the Hessian if desired.
         *    - If the Hessian is not positive-definite, perturb by a small multiple
         *      of the identity until it is positive-definite. 
         * 3) Define the quadratic subproblem according to the above quantities and
         *    the constraints (see below). 
         * 4) Solve the quadratic subproblem, check that the new vector satisfies the
         *    constraints of the original problem, and output the new vector.
         *
         * @param func                    Objective function.
         * @param iter                    Iteration number.
         * @param quasi_newton            Quasi-Newton method.
         * @param regularize              Regularization method. 
         * @param regularize_weight       Regularization weight.
         * @param prev_data               Data regarding current iterate of the
         *                                optimization.
         * @param delta                   Increment for finite difference 
         *                                approximation.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semi-definiteness).
         * @param min_stepsize            Minimum stepsize.
         * @param x_tol                   Tolerance of change in input vector 
         *                                (L2 norm between successive iterates).
         * @param hessian_modify_max_iter Maximum number of Hessian modifications.
         * @param c1                      Constant multiplier in Armijo condition.
         * @param c2                      Constant multiplier in strong curvature
         *                                condition.
         * @param line_search_max_iter    Maximum number of iterations during 
         *                                line search.
         * @param zoom_max_iter           Maximum number of iterations in `zoom()`.
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param search_verbose          If true, output intermittent messages 
         *                                to `stdout` from `lineSearch()`.
         * @param zoom_verbose            If true, output intermittent messages 
         *                                to `stdout` from `zoom()`.
         */
        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                         const int iter, const QuasiNewtonMethod quasi_newton,
                         const RegularizationMethod regularize, 
                         const T regularize_weight, StepData<T> prev_data,
                         const T delta, const T beta, const T min_stepsize,
                         const T x_tol, const int hessian_modify_max_iter,
                         const T c1, const T c2, const int line_search_max_iter,
                         const int zoom_max_iter, const bool verbose = false,
                         const bool search_verbose = false,
                         const bool zoom_verbose = false)
        {
            using std::abs;
            using boost::multiprecision::abs;

            // Assume that prev_data incorporates contributions from regularization
            T f_curr = prev_data.f; 
            Matrix<T, Dynamic, 1> xl_curr = prev_data.xl;
            Matrix<T, Dynamic, 1> x_curr = xl_curr.head(this->D);
            Matrix<T, Dynamic, 1> grad_curr = prev_data.grad_f;
            Matrix<T, Dynamic, 1> grad_lagrangian = prev_data.grad_lagrangian;
            Matrix<T, Dynamic, Dynamic> hessian_lagrangian = modify<T>(
                prev_data.hessian_lagrangian, hessian_modify_max_iter, beta
            );

            // If any of the components have a non-finite coordinate, return as is
            if (!x_curr.array().isFinite().all() ||
                !grad_curr.array().isFinite().all() ||
                !grad_lagrangian.array().isFinite().all() ||
                !hessian_lagrangian.array().isFinite().all())
            {
                return prev_data;
            }

            // Evaluate the constraints and their gradients
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>();   // Convert from rationals to T
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();         // Convert from rationals to T
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            Matrix<T, Dynamic, 1> c;
            if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                c = -(A * x_curr - b);
            else 
                c = A * x_curr - b; 

            /** -------------------------------------------------------------- //
             * Set up the quadratic program (Nocedal and Wright, Eq. 18.11):
             * 
             * Minimize:
             *
             *   p.T * D * p + c.T * p + c0
             * = fk + Dfk.T * p + 0.5 * p.T * D2Lk * p
             *
             * subject to: 
             *
             *    A * p + (A * xk - b) >= 0   if the inequality type of the linear constraints is >=
             *   -A * p - (A * xk - b) >= 0   if the inequality type of the linear constraints is <=
             *
             * where:
             *
             * p    = variables to be optimized -- vector to be added to current iterate
             * xk   = current iterate 
             * fk   = f(xk)
             * Dfk  = gradient of f at xk
             * D2Lk = (approximation of) Hessian of Lagrangian at xk 
             * --------------------------------------------------------------- */
            // Note that, with the damped BFGS update, the Hessian matrix approximation
            // should be positive-definite
            for (int i = 0; i < this->D; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    // Sets 2D_ij and 2D_ji (the quadratic part of objective)
                    this->program->set_d(i, j, static_cast<double>(hessian_lagrangian(i, j)));
                }
                // Sets c_i (the linear part of objective)
                this->program->set_c(i, static_cast<double>(grad_curr(i)));
            }
            for (int i = 0; i < this->N; ++i)
            {
                for (int j = 0; j < this->D; ++j)
                {
                    // Sets A_ij (j-th coefficient of i-th constraint)
                    if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                        this->program->set_a(j, i, static_cast<double>(A(i, j)));
                    else 
                        this->program->set_a(j, i, -static_cast<double>(A(i, j))); 
                }
                // Sets b_i (i-th coordinate of -(A * xk - b) if inequality type is >=,
                // i-th coordinate of (A * xk - b) if inequality type is <=)
                this->program->set_b(i, static_cast<double>(c(i)));
            }
            // Note that the constant part of the objective (fk) is unnecessary

            // Solve the quadratic program ...
            Solution solution; 
            try
            {
                solution = CGAL::solve_quadratic_program(*this->program, ET());
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // ... if the program cannot be solved because the D matrix is not 
                // positive-semi-definite (this should never be the case), then replace
                // D with the identity matrix
                for (int i = 0; i < this->D; ++i)
                {
                    for (int j = 0; j <= i; ++j)
                    {
                        this->program->set_d(i, j, 2.0);    // Sets 2D_ij and 2D_ji
                    }
                }
                try
                {
                    solution = CGAL::solve_quadratic_program(*this->program, ET());
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // The program should never be infeasible, since we assume that 
            // the constraint matrix has full rank
            std::stringstream ss; 
            if (solution.is_infeasible())
            {
                ss << "Quadratic program is infeasible; check constraint matrix:\n" << A;
                throw std::runtime_error(ss.str());
            }
            // The program should also never yield an unbounded solution, 
            // since we assume that the constraint matrix specifies a 
            // bounded polytope 
            else if (solution.is_unbounded())
            {
                ss << "Quadratic program yielded unbounded solution; check constraint matrix:\n" << A;
                throw std::runtime_error(ss.str());
            }

            // Collect the values of the solution into a vector 
            Matrix<T, Dynamic, 1> p(this->D);
            int i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                p(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // If the vector has length > 1, then normalize by its length 
            // 
            // Note that if the vector has length < 1, then normalizing may
            // cause the constraints to be violated by a step of stepsize 1,
            // but since the constraints are linear (and thus convex), if the
            // vector has length > 1, then the normalized vector will still 
            // satisfy the constraints
            T p_norm = p.norm();
            if (p_norm > 1)
            {
                for (int i = 0; i < this->D; ++i)
                    p(i) /= p_norm;
            }

            // Collect the values of the new Lagrange multipliers (i.e., the
            // "optimality certificate")
            Matrix<T, Dynamic, 1> xl_new(this->D + this->N);
            i = 0;
            for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
            {
                xl_new(this->D + i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Identify a stepsize that (ideally) satisfies the strong Wolfe 
            // conditions for this iteration
            auto gradient = [this, &func, &delta](const Ref<const Matrix<T, Dynamic, 1> >& x) -> Matrix<T, Dynamic, 1>
            {
                return this->gradient(func, x, delta);
            };
            const T max_stepsize = 1;
            std::tuple<T, bool, bool> result = lineSearch<T>(
                func, gradient, x_curr, f_curr, std::numeric_limits<T>::quiet_NaN(),
                grad_curr, p, min_stepsize, max_stepsize, c1, c2, line_search_max_iter,
                zoom_max_iter, regularize, regularize_weight, search_verbose,
                zoom_verbose
            );
            T stepsize = std::get<0>(result);
            bool satisfies_armijo = std::get<1>(result);
            bool satisfies_curvature = std::get<2>(result);

            // Advance by the given stepsize in the given direction, and evaluate 
            // the objective and its gradient at the new input vector
            Matrix<T, Dynamic, 1> step = stepsize * p;
            Matrix<T, Dynamic, 1> x_new = x_curr + step;
            T f_new = func(x_new);
            Matrix<T, Dynamic, 1> grad_new = this->gradient(func, x_new, delta);
            T f_reg = 0;
            Matrix<T, Dynamic, 1> grad_reg = Matrix<T, Dynamic, 1>::Zero(this->D);
            switch (regularize)
            {
                case NOREG:
                    break; 

                case L1:
                    f_reg = regularize_weight * x_new.cwiseAbs().sum();
                    for (int i = 0; i < this->D; ++i)
                        grad_reg(i) = regularize_weight * ((x_new(i) > 0) - (x_new(i) < 0));
                    break;

                case L2:
                    f_reg = regularize_weight * x_new.array().pow(2).sum();
                    grad_reg = regularize_weight * 2 * x_new;
                    break;

                default:
                    break;
            }
            T f_combined = f_new + f_reg;
            Matrix<T, Dynamic, 1> grad_combined = grad_new + grad_reg;
            T change_x = step.norm(); 
            T change_f = f_combined - f_curr; 
            xl_new.head(this->D) = x_new;

            // Print the stepping direction, stepsize, new vector, and value of
            // the objective function and its gradient 
            if (verbose)
            {
                std::cout << "... stepping direction = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << p(i) << ", ";
                std::cout << p(this->D - 1) << ")" << std::endl;
                std::cout << "... stepsize = " << stepsize
                          << ": Armijo = " << satisfies_armijo
                          << ", strong curvature = " << satisfies_curvature
                          << std::endl; 
                std::cout << "Iteration " << iter << ": x = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << x_new(i) << ", "; 
                std::cout << x_new(this->D - 1)
                          << "); f(x) = " << f_combined 
                          << "; change in x = " << change_x 
                          << "; change in f = " << change_f
                          << "; gradient = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << grad_combined(i) << ", ";
                std::cout << grad_combined(this->D - 1)
                          << ")" << std::endl; 
            }
            
            // Evaluate the gradient and Hessian of the Lagrangian (with respect
            // to the input space)
            Matrix<T, Dynamic, 1> grad_lagrangian_mixed = this->lagrangianGradient(
                x_curr, xl_new.tail(this->N), grad_curr
            ); 
            Matrix<T, Dynamic, 1> grad_lagrangian_new = this->lagrangianGradient(
                x_new, xl_new.tail(this->N), grad_combined
            );
            Matrix<T, Dynamic, Dynamic> hessian_lagrangian_new;
            Matrix<T, Dynamic, 1> y = grad_lagrangian_new.head(this->D) - grad_lagrangian_mixed.head(this->D);
            auto _hessian_lagrangian = hessian_lagrangian.template selfadjointView<Lower>(); 
            switch (quasi_newton)
            {
                case BFGS:
                    hessian_lagrangian_new = updateBFGSDamped<T>(_hessian_lagrangian, step, y);
                    break;

                case SR1:
                    hessian_lagrangian_new = updateSR1<T>(_hessian_lagrangian, step, y); 
                    break;

                default:
                    break;
            }

            // Return the new data
            StepData<T> new_data;
            new_data.f = f_combined; 
            new_data.xl = xl_new;
            new_data.grad_f = grad_combined;
            new_data.grad_lagrangian = grad_lagrangian_new;
            new_data.hessian_lagrangian = hessian_lagrangian_new;
            return new_data;
        }

        /**
         * Run the optimization with the given objective function, initial
         * vector for the objective function, initial vector of Lagrange 
         * multipliers, and additional settings.
         *
         * @param func
         * @param x_init
         * @param l_init
         * @param delta                   Increment for finite difference 
         *                                approximation.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semi-definiteness).
         * @param min_stepsize            Minimum stepsize.
         * @param max_iter                Maximum number of steps.
         * @param tol                     Tolerance for change in objective value.
         * @param x_tol                   Tolerance for change in input vector 
         *                                (L2 norm between successive iterates).
         * @param quasi_newton            Quasi-Newton method.
         * @param regularize              Regularization method.
         * @param regularize_weight       Regularization weight.
         * @param hessian_modify_max_iter Maximum number of Hessian modifications.
         * @param c1                      Constant multiplier in Armijo condition.
         * @param c2                      Constant multiplier in strong curvature
         *                                condition.
         * @param line_search_max_iter    Maximum number of iterations during 
         *                                line search.
         * @param zoom_max_iter           Maximum number of iterations in `zoom()`.
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param search_verbose          If true, output intermittent messages 
         *                                to `stdout` from `lineSearch()`.
         * @param zoom_verbose            If true, output intermittent messages 
         *                                to `stdout` from `zoom()`.
         * @returns Minimizing input vector.
         */
        Matrix<T, Dynamic, 1> run(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                  const Ref<const Matrix<T, Dynamic, 1> >& x_init, 
                                  const Ref<const Matrix<T, Dynamic, 1> >& l_init,
                                  const T delta, const T beta, const T min_stepsize,
                                  const int max_iter, const T tol, const T x_tol,
                                  const QuasiNewtonMethod quasi_newton,
                                  const RegularizationMethod regularize,
                                  const T regularize_weight, 
                                  const int hessian_modify_max_iter, const T c1,
                                  const T c2, const int line_search_max_iter,
                                  const int zoom_max_iter,
                                  const bool verbose = false,
                                  const bool search_verbose = false,
                                  const bool zoom_verbose = false)
        {
            using std::abs;
            using boost::multiprecision::abs;

            // Evaluate the objective and its gradient
            T f_init = func(x_init);
            Matrix<T, Dynamic, 1> grad_init = this->gradient(func, x_init, delta);
            T f_reg = 0; 
            Matrix<T, Dynamic, 1> grad_reg = Matrix<T, Dynamic, 1>::Zero(this->D); 
            switch (regularize)
            {
                case NOREG:
                    break; 

                case L1:
                    f_reg = regularize_weight * x_init.cwiseAbs().sum();
                    for (int i = 0; i < this->D; ++i)
                        grad_reg(i) = regularize_weight * ((x_init(i) > 0) - (x_init(i) < 0));
                    break;

                case L2:
                    f_reg = regularize_weight * x_init.array().pow(2).sum();
                    grad_reg = regularize_weight * 2 * x_init;
                    break;

                default:
                    break;
            }
            T f_combined = f_init + f_reg;
            Matrix<T, Dynamic, 1> grad_combined = grad_init + grad_reg;

            // Print the input vector and value of the objective function
            if (verbose)
            {
                std::cout << "Initial vector: x = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << x_init(i) << ", ";
                std::cout << x_init(this->D - 1) << "); f(x) = "
                          << f_combined << "; gradient = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << grad_combined(i) << ", ";
                std::cout << grad_combined(this->D - 1) << ")" << std::endl; 
            }
            
            // Evaluate the Lagrangian and its gradient
            StepData<T> curr_data;
            curr_data.f = f_combined;
            curr_data.xl.conservativeResize(this->D + this->N);
            curr_data.xl.head(this->D) = x_init;
            curr_data.xl.tail(this->N) = l_init; 
            curr_data.grad_f = grad_combined;
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>(); 
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();
            Polytopes::InequalityType type = this->constraints->getInequalityType(); 
            T sign = (type == Polytopes::InequalityType::GreaterThanOrEqualTo ? -1 : 1);  
            //T lagrangian = f_combined + sign * l_init.dot(A * x_init - b);    // No need to compute
            Matrix<T, Dynamic, 1> grad_lagrangian = this->lagrangianGradient(x_init, l_init, grad_combined);
            curr_data.grad_lagrangian = grad_lagrangian;
            curr_data.hessian_lagrangian = Matrix<T, Dynamic, Dynamic>::Identity(this->D, this->D);

            int i = 0;
            T change_x = 2 * x_tol;
            T change_f = 2 * tol; 
            while (i < max_iter && (change_x > x_tol || change_f > tol))
            {
                StepData<T> next_data = this->step(
                    func, i, quasi_newton, regularize, regularize_weight, curr_data,
                    delta, beta, min_stepsize, x_tol, hessian_modify_max_iter,
                    c1, c2, line_search_max_iter, zoom_max_iter, verbose,
                    search_verbose, zoom_verbose
                ); 
                change_x = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).norm(); 
                change_f = abs(curr_data.f - next_data.f); 
                i++;
                curr_data.f = next_data.f; 
                curr_data.xl = next_data.xl;
                curr_data.grad_f = next_data.grad_f;
                curr_data.grad_lagrangian = next_data.grad_lagrangian;
                curr_data.hessian_lagrangian = next_data.hessian_lagrangian;
            }
            
            // Print the final vector and value of the objective function
            if (verbose)
            {
                std::cout << "Final vector: x = (";
                for (int j = 0; j < this->D - 1; ++j)
                    std::cout << curr_data.xl(j) << ", ";
                std::cout << curr_data.xl(this->D - 1) << "); f(x) = " 
                          << curr_data.f << std::endl; 
            }

            return curr_data.xl.head(this->D).template cast<T>();
        }
};

/**
 * An implementation of line-search sequential quadratic programming for
 * nonlinear optimization on convex polytopes (i.e., linear inequality
 * constraints) with forward-mode automatic differentiation for gradient
 * computation. 
 */
template <typename T>
class ForwardAutoDiffSQPOptimizer : public SQPOptimizer<T>
{
    public:
        /**
         * All inherited constructors from `SQPOptimizer`. 
         */
        ForwardAutoDiffSQPOptimizer(const unsigned D) : SQPOptimizer<T>(D)
        {
        }

        ForwardAutoDiffSQPOptimizer(const unsigned D, const unsigned N,
                                    const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                    const Ref<const Matrix<T, Dynamic, Dynamic> >& b)
            : SQPOptimizer<T>(D, N, A, b)
        {
        }

        ForwardAutoDiffSQPOptimizer(const unsigned D, const unsigned N,
                                    const Polytopes::InequalityType type, 
                                    const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                    const Ref<const Matrix<T, Dynamic, 1> >& b)
            : SQPOptimizer<T>(D, N, type, A, b)
        {
        }

        ForwardAutoDiffSQPOptimizer(Polytopes::LinearConstraints<mpq_rational>* constraints)
            : SQPOptimizer<T>(constraints)
        {
        }

        /**
         * Compute the gradient of the given function at the given vector. 
         *
         * Note that the input function should take in and return *dual* numbers. 
         *
         * @param func  Function whose gradient is to be approximated.
         * @param x     Input vector.
         * @returns Gradient approximation. 
         */
        Matrix<T, Dynamic, 1> gradient(std::function<Dual<T>(const Ref<const Matrix<Dual<T>, Dynamic, 1> >&)> func,
                                       const Ref<const Matrix<T, Dynamic, 1> >& x)
        {
            // Evaluate the function at D dual vectors, with derivative vectors 
            // initialized to standard unit vectors, one for each coordinate
            Matrix<T, Dynamic, 1> grad(this->D); 
            Matrix<Dual<T>, Dynamic, 1> x_(this->D);
            for (unsigned i = 0; i < this->D; ++i)
            {
                x_(i).a = x(i); 
                x_(i).b = 0; 
            } 
            for (unsigned i = 0; i < this->D; ++i)
            {
                x_(i).b = 1;
                grad(i) = func(x_).b;
                x_(i).b = 0;  
            }

            return grad;  
        }

        /**
         * Compute the gradient of the Lagrangian of the given function at 
         * the given vector.
         *
         * Note that the input function should take in and return *dual* numbers. 
         *
         * @param func  Input function.
         * @param x     Input vector for `func`.
         * @param l     Input vector for additional Lagrangian variables (one
         *              for each constraint).
         * @returns     Gradient approximation.
         */
        Matrix<T, Dynamic, 1> lagrangianGradient(std::function<Dual<T>(const Ref<const Matrix<Dual<T>, Dynamic, 1> >&)> func,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& x,
                                                 const Ref<const Matrix<T, Dynamic, 1> >& l)
        {
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>();   // Convert from rationals to T
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();         // Convert from rationals to T
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            T sign = (type == Polytopes::InequalityType::GreaterThanOrEqualTo ? -1 : 1);

            // Evaluate the function at 2 * D values, with each coordinate
            // perturbed by +/- delta, using gradient()
            Matrix<T, Dynamic, 1> grad_lagrangian = Matrix<T, Dynamic, 1>::Zero(this->D + this->N);
            Matrix<T, Dynamic, 1> grad_f = this->gradient(func, x); 
            grad_lagrangian.head(this->D) += grad_f;

            // Incorporate the contributions of the *linear* constraints to 
            // each partial derivative of the Lagrangian 
            for (int i = 0; i < this->D; ++i)
                grad_lagrangian(i) += sign * A.col(i).dot(l);
            for (int i = 0; i < this->N; ++i)
                grad_lagrangian(this->D + i) = sign * (A.row(i).dot(x) - b(i));  
            
            return grad_lagrangian;
        }

        /**
         * Run one step of the SQP algorithm.
         *
         * 1) Given an input vector `(x,l)`, compute `f(x)` and `df(x)/dx`. 
         * 2) Compute the Lagrangian, `L(x,l) = f(x) - l.T * A * x`, where `A` is the
         *    constraint matrix, and its Hessian matrix of second derivatives w.r.t. `x`.
         *    - Use a quasi-Newton method to compute the Hessian if desired.
         *    - If the Hessian is not positive-definite, perturb by a small multiple
         *      of the identity until it is positive-definite. 
         * 3) Define the quadratic subproblem according to the above quantities and
         *    the constraints (see below). 
         * 4) Solve the quadratic subproblem, check that the new vector satisfies the
         *    constraints of the original problem, and output the new vector.
         *
         * Note that the input function should take in and return *dual* numbers.
         */
        StepData<T> step(std::function<Dual<T>(const Ref<const Matrix<Dual<T>, Dynamic, 1> >&)> func,
                         const unsigned iter, const QuasiNewtonMethod quasi_newton,
                         StepData<T> prev_data, const T tau, const T beta,
                         const T tol, const bool use_only_armijo,
                         const bool use_strong_wolfe,
                         const unsigned hessian_modify_max_iter,
                         const T c1 = 1e-4, const T c2 = 0.9,
                         const bool verbose = false)
        {
            using std::abs;
            using boost::multiprecision::abs;

            T f = prev_data.f; 
            Matrix<T, Dynamic, 1> xl = prev_data.xl;
            Matrix<T, Dynamic, 1> x = xl.head(this->D);
            Matrix<T, Dynamic, 1> df = prev_data.df;
            Matrix<T, Dynamic, 1> dL = prev_data.dL;
            Matrix<T, Dynamic, Dynamic> d2L = modify<T>(prev_data.d2L, hessian_modify_max_iter, beta);

            // If any of the components have a non-finite coordinate, return as is
            if (!x.array().isFinite().all() || !df.array().isFinite().all() || !dL.array().isFinite().all() || !d2L.array().isFinite().all())
                return prev_data;

            // Evaluate the constraints and their gradients
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>();   // Convert from rationals to T
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();         // Convert from rationals to T
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            Matrix<T, Dynamic, 1> c;
            if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                c = -(A * x - b);
            else 
                c = A * x - b; 

            /** -------------------------------------------------------------- //
             * Set up the quadratic program (Nocedal and Wright, Eq. 18.11):
             * 
             * Minimize:
             *
             *   p.T * D * p + c.T * p + c0
             * = fk + Dfk.T * p + 0.5 * p.T * D2Lk * p
             *
             * subject to: 
             *
             *    A * p + (A * xk - b) >= 0   if the inequality type of the linear constraints is >=
             *   -A * p - (A * xk - b) >= 0   if the inequality type of the linear constraints is <=
             *
             * where:
             *
             * p    = variables to be optimized -- vector to be added to current iterate
             * xk   = current iterate 
             * fk   = f(xk)
             * Dfk  = gradient of f at xk
             * D2Lk = (approximation of) Hessian of Lagrangian at xk 
             * --------------------------------------------------------------- */
            // Note that, with the damped BFGS update, the Hessian matrix approximation
            // should be positive-definite
            for (unsigned i = 0; i < this->D; ++i)
            {
                for (unsigned j = 0; j <= i; ++j)
                {
                    // Sets 2D_ij and 2D_ji (the quadratic part of objective)
                    this->program->set_d(i, j, static_cast<double>(d2L(i,j)));
                }
                // Sets c_i (the linear part of objective)
                this->program->set_c(i, static_cast<double>(df(i)));
            }
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                {
                    // Sets A_ij (j-th coefficient of i-th constraint)
                    if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                        this->program->set_a(j, i, static_cast<double>(A(i,j)));
                    else 
                        this->program->set_a(j, i, -static_cast<double>(A(i,j))); 
                }
                // Sets b_i (i-th coordinate of -(A * xk - b) if inequality type is >=,
                // i-th coordinate of (A * xk - b) if inequality type is <=)
                this->program->set_b(i, static_cast<double>(c(i)));
            }
            // Note that the constant part of the objective (fk) is unnecessary

            // Solve the quadratic program ...
            Solution solution; 
            try
            {
                solution = CGAL::solve_quadratic_program(*this->program, ET());
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // ... if the program cannot be solved because the D matrix is not 
                // positive-semi-definite (this should never be the case), then replace
                // D with the identity matrix
                std::cout << "Setting matrix in quadratic part of objective to identity" << std::endl;
                for (unsigned i = 0; i < this->D; ++i)
                {
                    for (unsigned j = 0; j <= i; ++j)
                    {
                        this->program->set_d(i, j, 2.0);    // Sets 2D_ij and 2D_ji
                    }
                }
                try
                {
                    solution = CGAL::solve_quadratic_program(*this->program, ET());
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // The program should never be infeasible, since we assume that 
            // the constraint matrix has full rank
            std::stringstream ss; 
            if (solution.is_infeasible())
            {
                ss << "Quadratic program is infeasible; check constraint matrix:\n" << A;
                throw std::runtime_error(ss.str());
            }
            // The program should also never yield an unbounded solution, 
            // since we assume that the constraint matrix specifies a 
            // bounded polytope 
            else if (solution.is_unbounded())
            {
                ss << "Quadratic program yielded unbounded solution; check constraint matrix:\n" << A;
                throw std::runtime_error(ss.str());
            }

            // Collect the values of the solution into a vector 
            Matrix<T, Dynamic, 1> p(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                p(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Collect the values of the new Lagrange multipliers (i.e., the
            // "optimality certificate")
            Matrix<T, Dynamic, 1> xl_new(this->D + this->N);
            i = 0;
            for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
            {
                xl_new(this->D + i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Try updates of decreasing stepsizes from the input vector, 
            // choosing the largest stepsize that satisfies the Wolfe conditions
            Matrix<T, Dynamic, 1> x_new(this->D);
            T stepsize = 1;
            T factor = tau;
            Matrix<T, Dynamic, 1> step = stepsize * p;
            x_new = x + step;
            Matrix<Dual<T>, Dynamic, 1> x_new_(this->D);
            for (int i = 0; i < this->D; ++i)
            {
                x_new_(i).a = x_new(i); 
                x_new_(i).b = 0;
            } 
            T f_new = func(x_new_).a;
            Matrix<T, Dynamic, 1> df_new = this->gradient(func, x_new);  
            bool satisfies_armijo = wolfeArmijo(p, stepsize, f, f_new, df, c1);
            bool satisfies_curvature = false;  
            if (use_only_armijo)
                satisfies_curvature = true;
            else 
                satisfies_curvature = (
                    use_strong_wolfe
                    ? wolfeStrongCurvature(p, df, df_new, c2)
                    : wolfeCurvature(p, df, df_new, c2)
                );
            T change_x = step.norm(); 
            T change_f = f_new - f; 
            if (verbose)
            {
                std::cout << "... stepping direction = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << p(i) << ", ";
                std::cout << p(this->D - 1) << ")" << std::endl
                          << "... trying stepsize = " << stepsize
                          << ": Armijo = " << satisfies_armijo;
                if (!use_only_armijo)
                    std::cout << ", curvature = " << satisfies_curvature;
                std::cout << std::endl; 
            }
            while ((change_x > tol || abs(change_f) > tol) && !(satisfies_armijo && satisfies_curvature))
            {
                stepsize *= factor;
                factor /= 2;
                step = stepsize * p; 
                x_new = x + step;
                for (int i = 0; i < this->D; ++i)
                {
                    x_new_(i).a = x_new(i); 
                    x_new_(i).b = 0;
                } 
                f_new = func(x_new_).a; 
                df_new = this->gradient(func, x_new); 
                satisfies_armijo = wolfeArmijo(p, stepsize, f, f_new, df, c1);
                if (use_only_armijo)
                    satisfies_curvature = true;
                else  
                    satisfies_curvature = (
                        use_strong_wolfe
                        ? wolfeStrongCurvature(p, df, df_new, c2)
                        : wolfeCurvature(p, df, df_new, c2)
                    );
                change_x = step.norm();
                change_f = f_new - f; 
                if (verbose)
                {
                    std::cout << "... trying stepsize = " << stepsize
                              << ": Armijo = " << satisfies_armijo;
                    if (!use_only_armijo)
                        std::cout << ", curvature = " << satisfies_curvature;
                    std::cout << std::endl; 
                }
            }  
            xl_new.head(this->D) = x_new;

            // Print the new vector and value of the objective function
            // and its gradient
            if (verbose)
            {
                std::cout << "Iteration " << iter << ": x = (";
                for (int i = 0; i < x_new.size() - 1; ++i)
                    std::cout << x_new(i) << ", "; 
                std::cout << x_new(x_new.size() - 1)
                          << "); f(x) = " << f_new
                          << "; change in x = " << change_x 
                          << "; change in f = " << change_f
                          << "; gradient = (";
                for (int i = 0; i < df_new.size() - 1; ++i)
                    std::cout << df_new(i) << ", ";
                std::cout << df_new(df_new.size() - 1)
                          << ")" << std::endl; 
            }
            
            // Evaluate the Hessian of the Lagrangian (with respect to the input space)
            Matrix<T, Dynamic, 1> dL_mixed = this->lagrangianGradient(func, x, xl_new.tail(this->N)); 
            Matrix<T, Dynamic, 1> dL_new = this->lagrangianGradient(func, x_new, xl_new.tail(this->N));
            Matrix<T, Dynamic, Dynamic> d2L_new;
            Matrix<T, Dynamic, 1> y = dL_new.head(this->D) - dL_mixed.head(this->D);
            auto d2L_ = d2L.template selfadjointView<Lower>(); 
            switch (quasi_newton)
            {
                case BFGS:
                    d2L_new = updateBFGSDamped<T>(d2L_, step, y); 
                    break;

                case SR1:
                    d2L_new = updateSR1<T>(d2L_, step, y); 
                    break;

                default:
                    break;
            }

            // Return the new data
            StepData<T> new_data;
            new_data.f = f_new; 
            new_data.xl = xl_new;
            new_data.df = df_new;
            new_data.dL = dL_new;
            new_data.d2L = d2L_new;
            return new_data;
        }

        /**
         * Run the optimization with the given objective function, initial
         * vector for the objective function, initial vector of Lagrange 
         * multipliers, and additional settings.
         *
         * Note that the input function should take in and return *dual* numbers.
         */
        Matrix<T, Dynamic, 1> run(std::function<Dual<T>(const Ref<const Matrix<Dual<T>, Dynamic, 1> >&)> func,
                                  const Ref<const Matrix<T, Dynamic, 1> >& x_init, 
                                  const Ref<const Matrix<T, Dynamic, 1> >& l_init,
                                  const T tau, const T beta,
                                  const unsigned max_iter, const T tol,
                                  const QuasiNewtonMethod quasi_newton,
                                  const RegularizationMethod regularize,
                                  const T regularize_weight, 
                                  const bool use_only_armijo, 
                                  const bool use_strong_wolfe, 
                                  const unsigned hessian_modify_max_iter,
                                  const T c1 = 1e-4, const T c2 = 0.9,
                                  const bool verbose = false)
        {
            using std::abs;
            using boost::multiprecision::abs;

            // Define an objective function to be optimized from the given
            // function and the desired regularization method
            std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> obj; 
            switch (regularize)
            {
                case NOREG:
                    obj = func;
                    break; 

                case L1:
                    obj = [&func, &regularize_weight](const Ref<const Matrix<T, Dynamic, 1> >& x) -> T
                    {
                        return func(x) + regularize_weight * x.cwiseAbs().sum(); 
                    };
                    break; 

                case L2:
                    obj = [&func, &regularize_weight](const Ref<const Matrix<T, Dynamic, 1> >& x) -> T
                    {
                        return func(x) + regularize_weight * x.cwiseAbs2().sum(); 
                    };
                    break;

                default:
                    break;  
            }

            // Evaluate the objective and its gradient
            Matrix<Dual<T>, Dynamic, 1> x_init_(this->D); 
            for (unsigned i = 0; i < this->D; ++i)
            {
                x_init_(i).a = x_init(i);
                x_init_(i).b = 0;  
            }
            T f = obj(x_init_).a;
            Matrix<T, Dynamic, 1> df = this->gradient(obj, x_init);

            // Print the input vector and value of the objective function
            if (verbose)
            {
                std::cout << "Initial vector: x = " << x_init.transpose()
                          << "; " << "f(x) = " << f << std::endl; 
            }
            
            // Evaluate the Lagrangian and its gradient
            StepData<T> curr_data;
            curr_data.f = f; 
            curr_data.xl.conservativeResize(this->D + this->N);
            curr_data.xl.head(this->D) = x_init;
            curr_data.xl.tail(this->N) = l_init; 
            curr_data.df = df;
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>(); 
            Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();
            Polytopes::InequalityType type = this->constraints->getInequalityType();
            T sign = (type == Polytopes::InequalityType::GreaterThanOrEqualTo ? -1 : 1);  
            T L = obj(x_init_).a + sign * l_init.dot(A * x_init - b);
            Matrix<T, Dynamic, 1> dL = this->lagrangianGradient(obj, x_init, l_init); 
            curr_data.dL = dL;
            curr_data.d2L = Matrix<T, Dynamic, Dynamic>::Identity(this->D, this->D);

            unsigned i = 0;
            T change_x = 2 * tol;
            T change_f = 2 * tol;  
            while (i < max_iter && (change_x > tol || change_f > tol))
            {
                StepData<T> next_data = this->step(
                    obj, i, quasi_newton, curr_data, tau, beta, tol,
                    use_only_armijo, use_strong_wolfe, hessian_modify_max_iter,
                    c1, c2, verbose
                ); 
                change_x = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).norm(); 
                change_f = abs(curr_data.f - next_data.f); 
                i++;
                curr_data.f = next_data.f; 
                curr_data.xl = next_data.xl;
                curr_data.df = next_data.df;
                curr_data.dL = next_data.dL;
                curr_data.d2L = next_data.d2L;
            }
            return curr_data.xl.head(this->D).template cast<T>();
        }
};

#endif 
