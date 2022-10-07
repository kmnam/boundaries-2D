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
 *   form x_k + a_k * p_k, with step-size a_k < 1 and direction p_k, satisfies
 *   all constraints if x_k + p_k does; and 
 * - inconsistent linearizations in the style of Nocedal and Wright, Eq. 18.12
 *   are not encountered.
 *
 * Step-sizes are determined to satisfy the (strong) Wolfe conditions, given by
 * Eqs. 3.6 and 3.7 in Nocedal and Wright.
 *
 * L1, L2, or elastic-net regularization may be incorporated into the
 * objective function if desired.  
 *
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     10/7/2022
 */

#ifndef SQP_OPTIMIZER_HPP
#define SQP_OPTIMIZER_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <boost/multiprecision/gmp.hpp>
#include <linearConstraints.hpp>
#include "quasiNewton.hpp"
#include "duals.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
using std::sqrt;
using boost::multiprecision::sqrt;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

enum QuasiNewtonMethod
{
    NONE,
    BFGS,
    SR1,
};

enum RegularizationMethod
{
    NOREG,
    L1,
    L2,
};

template <typename T>
struct StepData
{
    public:
        Matrix<T, Dynamic, 1> xl;
        T f; 
        Matrix<T, Dynamic, 1> df;
        Matrix<T, Dynamic, 1> dL;
        Matrix<T, Dynamic, Dynamic> d2L;
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
 * Return the sign of `x`.
 *
 * @param x Input value.
 * @returns 1 if `x > 0`, 0 if `x == 0`, -1 if `x < 0`.
 */
template <typename T>
T sign(const T x)
{
    if (x > 0)      return 1; 
    else if (x < 0) return -1;
    else            return 0;
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
         * Test whether the Armijo condition (Nocedal and Wright, Eq. 3.6a)
         * is satisfied by the given pair of old and new solutions.
         *
         * @param dir      Candidate direction for updating the old solution.
         * @param stepsize Candidate step-size for updating the old solution.
         * @param f_old    Pre-computed value of objective at the old solution.
         * @param f_new    Pre-computed value of objective at the new solution.
         * @param grad_old Pre-computed value of gradient at the old solution.
         * @param c        Constant multiplier. 
         */
        bool wolfeArmijo(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                         const T stepsize, const T f_old, const T f_new, 
                         const Ref<const Matrix<T, Dynamic, 1> >& grad_old,
                         const T c)
        {
            return (f_new <= f_old + c * stepsize * grad_old.dot(dir)); 
        }

        /**
         * Test whether the curvature condition (Nocedal and Wright, Eq. 3.6b)
         * is satisfied by the given pair of old and new solutions.
         *
         * @param dir      Candidate direction for updating the old solution.
         * @param grad_old Pre-computed value of gradient at the old solution.
         * @param grad_new Pre-computed value of gradient at the new solution.
         * @param c        Constant multiplier. 
         */
        bool wolfeCurvature(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                            const Ref<const Matrix<T, Dynamic, 1> >& grad_old,
                            const Ref<const Matrix<T, Dynamic, 1> >& grad_new,
                            const T c)
        {
            return (grad_new.dot(dir) >= c * grad_old.dot(dir));
        }

        /**
         * Test whether the *strong* curvature condition (Nocedal and Wright,
         * Eq. 3.7b) is satisfied by the given pair of old and new solutions.
         *
         * @param dir      Candidate direction for updating the old solution.
         * @param grad_old Pre-computed value of gradient at the old solution.
         * @param grad_new Pre-computed value of gradient at the new solution.
         * @param c        Constant multiplier. 
         */
        bool wolfeStrongCurvature(const Ref<const Matrix<T, Dynamic, 1> >& dir,
                                  const Ref<const Matrix<T, Dynamic, 1> >& grad_old,
                                  const Ref<const Matrix<T, Dynamic, 1> >& grad_new,
                                  const T c)
        {
            using std::abs;
            using boost::multiprecision::abs; 

            return (abs(grad_new.dot(dir)) <= c * abs(grad_old.dot(dir))); 
        }

        /**
         * An implementation of the zoom function (Algorithm 3.6 in Nocedal
         * and Wright), which chooses a stepsize that satisfies the strong
         * Wolfe conditions.
         *
         * @param func        Objective function.
         * @param x           Input vector.
         * @param f           Pre-computed value of objective at `x`.
         * @param df          Pre-computed value of gradient at `x`.
         * @param dir         Step vector.
         * @param stepsize_lo Min/max allowed stepsize, together with `stepsize_hi`.
         * @param stepsize_hi Min/max allowed stepsize, together with `stepsize_lo`.
         * @param c1          Constant multiplier in Armijo condition.
         * @param c2          Constant multiplier in strong curvature condition.
         * @param delta       Increment for finite difference approximation.
         * @param verbose     If true, output intermittent messages to `stdout`.
         * @returns New stepsize, together with indicators as to whether the 
         *          stepsize satisfies the Armijo and strong curvature conditions.
         */
        std::tuple<T, bool, bool> zoom(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                       const Ref<const Matrix<T, Dynamic, 1> >& x, const T f,
                                       const Ref<const Matrix<T, Dynamic, 1> >& grad,
                                       const Ref<const Matrix<T, Dynamic, 1> >& dir,
                                       T stepsize_lo, T stepsize_hi, const T stepsize_tol, 
                                       const T c1, const T c2, const T delta,
                                       const bool verbose)
        {
            bool satisfies_armijo = false; 
            bool satisfies_curvature = false;
            while (abs(stepsize_lo - stepsize_hi) > stepsize_tol)
            {
                T f_lo = func(x + stepsize_lo * dir);
                if (verbose)
                {
                    std::cout << "... zooming into stepsize interval between "
                              << stepsize_lo << " and " << stepsize_hi << std::endl;
                }

                // Determine new stepsize as the average of the two limiting stepsizes 
                T new_stepsize = (stepsize_lo + stepsize_hi) / 2;
                T f_new = func(x + new_stepsize * dir);
                T phi_deriv_zero = dir.dot(grad);
                if (verbose)
                {
                    std::cout << "...... candidate stepsize = " << new_stepsize << std::endl;
                }
                
                // The quantities defined by Nocedal and Wright below are 
                // given here by:
                //
                // \alpha_j = new_stepsize
                // \alpha_{lo} = stepsize_lo
                // \alpha_{hi} = stepsize_hi
                // \phi(0) = f
                // \phi'(0) = phi_deriv_zero = dir.dot(grad)
                // \phi(\alpha_j) = f_new
                // \phi(\alpha_{lo}) = func(x + stepsize_lo * dir) = f_lo
                satisfies_armijo = this->wolfeArmijo(dir, new_stepsize, f, f_new, grad, c1);
                if (!satisfies_armijo)
                {
                    if (verbose)
                    {
                        std::cout << "...... violates Armijo condition" << std::endl;
                    }
                    stepsize_hi = new_stepsize;
                }
                else if (f_new >= f_lo)
                {
                    if (verbose)
                    {
                        std::cout << "...... causes increase in objective relative to "
                                  << "stepsize_lo = " << stepsize_lo << std::endl;
                    }
                    stepsize_hi = new_stepsize;
                }
                else 
                {
                    Matrix<T, Dynamic, 1> grad_new = this->gradient(func, x + new_stepsize * dir, delta); 
                    satisfies_curvature = this->wolfeStrongCurvature(dir, grad, grad_new, c2); 
                    if (satisfies_curvature)
                    {
                        if (verbose)
                        {
                            std::cout << "...... found stepsize satisfying both Armijo and strong "
                                      << "curvature conditions" << std::endl;
                        }
                        return std::make_tuple(new_stepsize, true, true);   // Loop exits here
                    }
                    if (verbose)
                    {
                        std::cout << "...... violates strong curvature condition" << std::endl;
                    }
                    if (dir.dot(grad_new) * (stepsize_hi - stepsize_lo) >= 0)
                    {
                        stepsize_hi = stepsize_lo;
                    }
                    stepsize_lo = new_stepsize;
                }
            }
            T final_stepsize = (stepsize_lo + stepsize_hi) / 2;
            T phi_final = func(x + final_stepsize * dir);
            Matrix<T, Dynamic, 1> grad_final = this->gradient(func, x + final_stepsize * dir, delta); 
            satisfies_armijo = this->wolfeArmijo(dir, final_stepsize, f, phi_final, grad, c1);
            satisfies_curvature = this->wolfeStrongCurvature(dir, grad, grad_final, c2);
            if (verbose)
            {
                std::cout << "... terminating zoom (interval between " << stepsize_lo
                          << " and " << stepsize_hi << ")" << std::endl
                          << "... final stepsize = " << final_stepsize
                          << "; Armijo = " << satisfies_armijo 
                          << "; strong curvature = " << satisfies_curvature << std::endl; 
            }
            return std::make_tuple(final_stepsize, satisfies_armijo, satisfies_curvature);
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
            Matrix<T, Dynamic, 1> dL = Matrix<T, Dynamic, 1>::Zero(this->D + this->N);
            Matrix<T, Dynamic, 1> df = this->gradient(func, x, delta); 
            dL.head(this->D) += df;

            // Incorporate the contributions of the *linear* constraints to 
            // each partial derivative of the Lagrangian 
            for (int i = 0; i < this->D; ++i)
                dL(i) += sign * A.col(i).dot(l);
            for (int i = 0; i < this->N; ++i)
                dL(this->D + i) = sign * (A.row(i).dot(x) - b(i));  
            
            return dL;
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
         * @param prev_data               Data regarding current iterate of the
         *                                optimization.
         * @param delta                   Increment for finite difference 
         *                                approximation.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semi-definiteness).
         * @param stepsize_multiple       Multiple with which to increment 
         *                                stepsizes in search (see below).
         * @param stepsize_min            Minimum stepsize.
         * @param x_tol                   Tolerance of change in input vector 
         *                                (L2 norm between successive iterates).
         * @param hessian_modify_max_iter Maximum number of Hessian modifications.
         * @param c1                      Constant multiplier in Armijo condition.
         * @param c2                      Constant multiplier in strong curvature
         *                                condition.
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param zoom_verbose            If true, output intermittent messages 
         *                                to `stdout` from `zoom()`.
         */
        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                         const int iter, const QuasiNewtonMethod quasi_newton,
                         StepData<T> prev_data, const T delta, const T beta, 
                         const T stepsize_multiple, const T stepsize_min, 
                         const T x_tol, const int hessian_modify_max_iter,
                         const T c1 = 1e-4, const T c2 = 0.9, const bool verbose = false,
                         const bool zoom_verbose = false)
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
            for (int i = 0; i < this->D; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    // Sets 2D_ij and 2D_ji (the quadratic part of objective)
                    this->program->set_d(i, j, static_cast<double>(d2L(i,j)));
                }
                // Sets c_i (the linear part of objective)
                this->program->set_c(i, static_cast<double>(df(i)));
            }
            for (int i = 0; i < this->N; ++i)
            {
                for (int j = 0; j < this->D; ++j)
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

            // Collect the values of the new Lagrange multipliers (i.e., the
            // "optimality certificate")
            Matrix<T, Dynamic, 1> xl_new(this->D + this->N);
            i = 0;
            for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
            {
                xl_new(this->D + i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            /** -------------------------------------------------------------- //
             * Identify stepsize with an interpolation-based approach 
             * (Nocedal and Wright, Algorithms 3.5 and 3.6)
             *
             * This algorithm identifies a stepsize between stepsize_min and 1.0
             * that satisfies the *strong* Wolfe conditions by taking two
             * candidate stepsizes, stepsize0 and stepsize1, and choosing as
             * the stepsize:
             *
             * - stepsize1 itself if stepsize1 satisfies the strong Wolfe
             *   conditions;
             * - a value in [stepsize0, stepsize1] if *any* of the three
             *   conditions are satisfied:
             *   1) stepsize1 violates the Armijo condition, 
             *   2) func(x + stepsize1 * p) >= func(x + stepsize0 * p), meaning
             *      that stepsize1 leads to an increased objective value 
             *      relative to stepsize0, or
             *   3) the descent direction at stepsize1 is nonnegative; *or*
             * - a value in (a subinterval of) [stepsize1, 1.0] by repeating
             *   the above checks with stepsize1 <- stepsize1 + stepsize_multiple
             *   and stepsize0 <- stepsize1
             * --------------------------------------------------------------- */
            const T stepsize_max = 1;
            T stepsize0 = stepsize_min;
            T stepsize1 = stepsize_multiple;
            i = 1;

            // Terminate the search when the search exceeds stepsize_max
            T stepsize = stepsize_min;
            bool satisfies_armijo = false;
            bool satisfies_curvature = false;
            while (stepsize1 < stepsize_max)
            {
                // Evaluate func(x + stepsize0 * p) and func(x + stepsize1 * p)
                T f0 = func(x + stepsize0 * p);
                T f1 = func(x + stepsize1 * p);

                // Does stepsize1 satisfy the Armijo condition? 
                if (!this->wolfeArmijo(p, stepsize1, f, f1, df, c1))
                {
                    // If it does *not*, then the interval between stepsize0 and
                    // stepsize1 must contain stepsizes that satisfy both strong
                    // Wolfe conditions
                    //
                    // In this case, stepsize0 should still satisfy the Armijo
                    // condition (and hence can be passed first as "\alpha_lo")
                    auto result = zoom(
                        func, x, f, df, p, stepsize0, stepsize1, stepsize_min,
                        c1, c2, delta, zoom_verbose
                    );
                    stepsize = std::get<0>(result);
                    satisfies_armijo = std::get<1>(result); 
                    satisfies_curvature = std::get<2>(result);  
                    break;
                }
                // Or does stepsize1 lead to an increased objective value relative
                // to stepsize0?
                else if (f1 >= f0 || i > 1)
                {
                    // If it does, then again the interval between stepsize0 and 
                    // stepsize1 must contain stepsizes that satisfy both strong
                    // Wolfe conditions
                    //
                    // In this case, both stepsize0 and stepsize1 should satisfy
                    // the Armijo condition (and hence either can be passed first
                    // as "\alpha_lo")
                    auto result = zoom(
                        func, x, f, df, p, stepsize0, stepsize1, stepsize_min,
                        c1, c2, delta, zoom_verbose
                    );
                    stepsize = std::get<0>(result);
                    satisfies_armijo = std::get<1>(result); 
                    satisfies_curvature = std::get<2>(result);  
                    break;
                }

                // Otherwise, does stepsize1 satisfy the strong curvature condition?
                Matrix<T, Dynamic, 1> df1 = this->gradient(func, x + stepsize1 * p, delta); 
                if (this->wolfeStrongCurvature(p, df, df1, c2))
                {
                    // If it does, then seeing as stepsize1 must also satisfy
                    // the Armijo condition (see above), stepsize1 satisfies 
                    // both strong Wolfe conditions 
                    stepsize = stepsize1;
                    satisfies_armijo = true;
                    satisfies_curvature = true;
                    break;
                }

                // Otherwise, is the descent direction at x + stepsize1 * p nonnegative?
                if (p.dot(df1) >= 0)
                {
                    // If it is, then the interval between stepsize0 and stepsize1
                    // must contain stepsizes that satisfy both strong Wolfe
                    // conditions
                    //
                    // In this case, stepsize1 should satisfy the Armijo condition
                    // (and hence can be passed first as "\alpha_lo")
                    auto result = zoom(
                        func, x, f, df, p, stepsize1, stepsize0, stepsize_min,
                        c1, c2, delta, zoom_verbose
                    );
                    stepsize = std::get<0>(result);
                    satisfies_armijo = std::get<1>(result); 
                    satisfies_curvature = std::get<2>(result);  
                    break;
                }

                // Update stepsize0, stepsize1, and i
                stepsize0 = stepsize1; 
                stepsize1 += stepsize_multiple;
                i++;
            }
            if (stepsize > stepsize_max) 
                std::cout << "\n\n\nPROBLEM!! WHY???\n\n\n";
            Matrix<T, Dynamic, 1> step = stepsize * p;
            Matrix<T, Dynamic, 1> x_new = x + step;
            T f_new = func(x_new);
            Matrix<T, Dynamic, 1> df_new = this->gradient(func, x_new, delta);
            T change_x = step.norm(); 
            T change_f = f_new - f; 
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
            Matrix<T, Dynamic, 1> dL_mixed = this->lagrangianGradient(func, x, xl_new.tail(this->N), delta); 
            Matrix<T, Dynamic, 1> dL_new = this->lagrangianGradient(func, x_new, xl_new.tail(this->N), delta);
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
         */
        Matrix<T, Dynamic, 1> run(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                  const Ref<const Matrix<T, Dynamic, 1> >& x_init, 
                                  const Ref<const Matrix<T, Dynamic, 1> >& l_init,
                                  const T delta, const T beta, const T stepsize_multiple, 
                                  const T stepsize_min, const int max_iter,
                                  const T tol, const T x_tol,
                                  const QuasiNewtonMethod quasi_newton,
                                  const RegularizationMethod regularize,
                                  const T regularize_weight, 
                                  const int hessian_modify_max_iter,
                                  const T c1 = 1e-4, const T c2 = 0.9,
                                  const bool verbose = false,
                                  const bool zoom_verbose = false)
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
            T f = obj(x_init);
            Matrix<T, Dynamic, 1> df = this->gradient(obj, x_init, delta);

            // Print the input vector and value of the objective function
            if (verbose)
            {
                std::cout << "Initial vector: x = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << x_init(i) << ", ";
                std::cout << x_init(this->D - 1) << "); f(x) = "
                          << f << std::endl; 
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
            T L = obj(x_init) + sign * l_init.dot(A * x_init - b); 
            Matrix<T, Dynamic, 1> dL = this->lagrangianGradient(obj, x_init, l_init, delta); 
            curr_data.dL = dL;
            curr_data.d2L = Matrix<T, Dynamic, Dynamic>::Identity(this->D, this->D);

            int i = 0;
            T change_x = 2 * x_tol;
            T change_f = 2 * tol; 
            while (i < max_iter && (change_x > x_tol || change_f > tol))
            {
                StepData<T> next_data = this->step(
                    obj, i, quasi_newton, curr_data, delta, beta, stepsize_multiple,
                    stepsize_min, x_tol, hessian_modify_max_iter, c1, c2, verbose,
                    zoom_verbose
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
            Matrix<T, Dynamic, 1> dL = Matrix<T, Dynamic, 1>::Zero(this->D + this->N);
            Matrix<T, Dynamic, 1> df = this->gradient(func, x); 
            dL.head(this->D) += df;

            // Incorporate the contributions of the *linear* constraints to 
            // each partial derivative of the Lagrangian 
            for (int i = 0; i < this->D; ++i)
                dL(i) += sign * A.col(i).dot(l);
            for (int i = 0; i < this->N; ++i)
                dL(this->D + i) = sign * (A.row(i).dot(x) - b(i));  
            
            return dL;
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
            bool satisfies_armijo = this->wolfeArmijo(p, stepsize, f, f_new, df, c1);
            bool satisfies_curvature = false;  
            if (use_only_armijo)
                satisfies_curvature = true;
            else 
                satisfies_curvature = (
                    use_strong_wolfe
                    ? this->wolfeStrongCurvature(p, df, df_new, c2)
                    : this->wolfeCurvature(p, df, df_new, c2)
                );
            T change_x = step.norm(); 
            T change_f = f_new - f; 
            if (verbose)
            {
                std::cout << "... stepping direction = (";
                for (int i = 0; i < this->D - 1; ++i)
                    std::cout << p(i) << ", ";
                std::cout << p(this->D - 1) << ")" << std::endl
                          << "... trying step-size = " << stepsize
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
                satisfies_armijo = this->wolfeArmijo(p, stepsize, f, f_new, df, c1);
                if (use_only_armijo)
                    satisfies_curvature = true;
                else  
                    satisfies_curvature = (
                        use_strong_wolfe
                        ? this->wolfeStrongCurvature(p, df, df_new, c2)
                        : this->wolfeCurvature(p, df, df_new, c2)
                    );
                change_x = step.norm();
                change_f = f_new - f; 
                if (verbose)
                {
                    std::cout << "... trying step-size = " << stepsize
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
