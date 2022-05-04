/**
 * An implementation of a nonlinear optimizer with respect to linear 
 * constraints with sequential quadratic programming (SQP) with automatic
 * differentiation and/or quasi-Newton Hessian approximations.
 *
 * Note that this implementation only deals with problems with *linear*
 * constraints. This means that inconsistent linearizations in the style 
 * of Nocedal and Wright, Eq. 18.12 are not encountered.  
 *
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     5/3/2022
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
        Matrix<T, Dynamic, 1> df;
        Matrix<T, Dynamic, 1> dL;
        Matrix<T, Dynamic, Dynamic> d2L;

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

template <typename T>
Matrix<T, Dynamic, Dynamic> modify(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                   unsigned max_iter, T beta)
{
    /*
     * Following the prescription of Nocedal and Wright (Algorithm 3.3, p.51), 
     * add successive multiples of the identity until the given matrix is 
     * positive-definite.
     *
     * The input matrix is assumed to be symmetric.
     *
     * Note that this function is not necessary with the damped BFGS update 
     * proposed in Procedure 18.2. 
     */
    // Check that A is positive-definite with the Cholesky decomposition
    LLT<Matrix<T, Dynamic, Dynamic> > dec(A);
    bool posdef = (dec.info() == Success);
    if (posdef) return A; 

    // If A is not positive-definite ...
    Matrix<T, Dynamic, Dynamic> B(A);
    T tau = 0.0;
    for (unsigned i = 0; i < max_iter; ++i)
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

        if (posdef) break;    // If so, then we are done 
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
        unsigned D;                                                 // Dimension of input space
        unsigned N;                                                 // Number of constraints
        Polytopes::LinearConstraints<mpq_rational>* constraints;    // Linear inequality constraints
        Program* program;                                           // Internal quadratic program to
                                                                    // be solved at each step

    public:
        /**
         * Straightforward constructor with `D` variables.
         *
         * Each variable is constrained to be greater than or equal to zero.
         *
         * @param D Number of variables.
         */
        SQPOptimizer(const unsigned D)
        {
            this->D = D;
            this->N = D;    // One constraint for each variable
            Matrix<mpq_rational, Dynamic, Dynamic> A = Matrix<mpq_rational, Dynamic, Dynamic>::Identity(D, D); 
            Matrix<mpq_rational, Dynamic, 1> b = Matrix<mpq_rational, Dynamic, 1>::Zero(D); 
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
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
        SQPOptimizer(const unsigned D, const unsigned N,
                     const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                     const Ref<const Matrix<T, Dynamic, Dynamic> >& b)
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
        SQPOptimizer(const unsigned D, const unsigned N,
                     const Polytopes::InequalityType type, 
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
        }

        /**
         * Destructor; deallocates the `LinearConstraints` and `Program` instances.
         */
        ~SQPOptimizer()
        {
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
            for (unsigned i = 0; i < this->D; ++i)
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
            auto lagrangian = [this, &func](const Ref<const Matrix<T, Dynamic, 1> >& x,
                                            const Ref<const Matrix<T, Dynamic, 1> >& l)
                {
                    Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>(); 
                    Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();
                    Polytopes::InequalityType type = this->constraints->getInequalityType(); 
                    if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                        return func(x) - l.dot(A * x - b);
                    else 
                        return func(x) + l.dot(A * x - b);  
                }; 

            // Evaluate the Lagrangian at 2 * D values, with each coordinate 
            // perturbed by +/- delta
            Matrix<T, Dynamic, 1> dL(this->D + this->N);
            Matrix<T, Dynamic, 1> x_(x);
            Matrix<T, Dynamic, 1> l_(l);
            for (unsigned i = 0; i < this->D; ++i)
            {
                x_(i) += delta;
                T f1 = lagrangian(x_, l_);
                x_(i) -= 2 * delta;
                T f2 = lagrangian(x_, l_);
                x_(i) += delta; 
                dL(i) = (f1 - f2) / (2 * delta);
            }
            for (unsigned i = 0; i < this->N; ++i)
            {
                l_(i) += delta;
                T f1 = lagrangian(x_, l_);
                l_(i) -= 2 * delta;
                T f2 = lagrangian(x_, l_);
                l_(i) += delta; 
                dL(this->D + i) = (f1 - f2) / (2 * delta);
            }

            return dL;
        }

        /**
         * Compute the L1 merit function (Nocedal and Wright, Eq. 15.24) with
         * respect to the given function and stored constraints.
         *
         * @param func Input function.
         * @param x    Input vector.
         * @param mu   Mu parameter for merit function (Nocedal and Wright, 
         *             Eq. 15.24)
         * @returns    Value of corresponding merit function.  
         */
        T meritL1(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func, 
                  const Ref<const Matrix<T, Dynamic, 1> >& x, const T mu)
        {
            T total = 0;
            Matrix<T, Dynamic, 1> y;  
            if (this->constraints->getInequalityType() == Polytopes::InequalityType::GreaterThanOrEqualTo)
                y = this->constraints->getA().template cast<T>() * x - this->constraints->getb().template cast<T>();
            else
                y = -this->constraints->getA().template cast<T>() * x + this->constraints->getb().template cast<T>();
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (y(i) < 0)
                    total -= y(i); 
            }
            return func(x) + mu * total; 
        }

        /**
         * Compute the L1 merit function (Nocedal and Wright, Eq. 15.24) with
         * respect to the given pre-computed function value and stored constraints.
         *
         * @param eval Pre-computed function value.
         * @param x    Input vector.
         * @param mu   Mu parameter for merit function (Nocedal and Wright, 
         *             Eq. 15.24)
         * @returns    Value of corresponding merit function.  
         */
        T meritL1(T eval, const Ref<const Matrix<T, Dynamic, 1> >& x, const T mu)
        {
            T total = 0;
            Matrix<T, Dynamic, 1> y;  
            if (this->constraints->getInequalityType() == Polytopes::InequalityType::GreaterThanOrEqualTo)
                y = this->constraints->getA().template cast<T>() * x - this->constraints->getb().template cast<T>();
            else
                y = -this->constraints->getA().template cast<T>() * x + this->constraints->getb().template cast<T>();
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (y(i) < 0)
                    total -= y(i); 
            }
            return eval + mu * total; 
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
         */
        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                         const unsigned iter, const QuasiNewtonMethod quasi_newton,
                         StepData<T> prev_data, const T delta, const T beta, 
                         const bool verbose, const unsigned hessian_modify_max_iter)
        {
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
            // Sets constant part of objective (fk)
            this->program->set_c0(static_cast<double>(f));

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
                try
                {
                    for (unsigned i = 0; i < this->D; ++i)
                    {
                        for (unsigned j = 0; j <= i; ++j)
                        {
                            this->program->set_d(i, j, 2.0);    // Sets 2D_ij and 2D_ji
                        }
                    }
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
            Matrix<T, Dynamic, 1> sol(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                sol(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Collect the values of the new Lagrange multipliers (i.e., the
            // "optimality certificate")
            Matrix<T, Dynamic, 1> mult(this->N);
            i = 0;
            for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
            {
                mult(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Increment the input vector and update the Lagrange multipliers
            Matrix<T, Dynamic, 1> xl_new(this->D + this->N);
            xl_new.head(this->D) = xl.head(this->D) + sol;
            xl_new.tail(this->N) = mult;
            Matrix<T, Dynamic, 1> x_new = xl_new.head(this->D);

            // Print the new vector and value of the objective function
            T f_new = func(x_new); 
            if (verbose)
            {
                std::cout << "Iteration " << iter << ": x = " << x_new.transpose() 
                          << "; f(x) = " << f_new 
                          << "; change = " << f_new - f << std::endl; 
            }
            
            // Evaluate the Hessian of the Lagrangian (with respect to the input space)
            Matrix<T, Dynamic, 1> df_new = this->gradient(func, x_new, delta);
            Matrix<T, Dynamic, 1> xl_mixed(xl);
            xl_mixed.tail(this->N) = mult; 
            Matrix<T, Dynamic, 1> dL_mixed = this->lagrangianGradient(func, x, mult, delta); 
            Matrix<T, Dynamic, 1> dL_new = this->lagrangianGradient(func, x_new, mult, delta);
            Matrix<T, Dynamic, Dynamic> d2L_new;
            Matrix<T, Dynamic, 1> y = dL_new.head(this->D) - dL_mixed.head(this->D);
            auto d2L_ = d2L.template selfadjointView<Lower>(); 
            switch (quasi_newton)
            {
                case BFGS:
                    d2L_new = updateBFGSDamped<T>(d2L_, sol, y); 
                    break;

                case SR1:
                    d2L_new = updateSR1<T>(d2L_, sol, y); 
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
                                  const T delta, const T beta, const unsigned max_iter,
                                  const T tol, const QuasiNewtonMethod quasi_newton,
                                  const bool verbose,
                                  const unsigned hessian_modify_max_iter)
        {
            // Evaluate the objective and its gradient
            T f = func(x_init);
            Matrix<T, Dynamic, 1> df = this->gradient(func, x_init, delta);

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
            auto lagrangian = [this, &func](const Ref<const Matrix<T, Dynamic, 1> >& x,
                                            const Ref<const Matrix<T, Dynamic, 1> >& l)
                {
                    Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>(); 
                    Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();
                    Polytopes::InequalityType type = this->constraints->getInequalityType(); 
                    if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                        return func(x) - l.dot(A * x - b);
                    else 
                        return func(x) + l.dot(A * x - b);  
                }; 
            T L = lagrangian(x_init, l_init); 
            Matrix<T, Dynamic, 1> dL = this->lagrangianGradient(func, x_init, l_init, delta); 
            curr_data.dL = dL;
            curr_data.d2L = Matrix<T, Dynamic, Dynamic>::Identity(this->D, this->D);

            unsigned i = 0;
            T change = 2 * tol; 
            while (i < max_iter && change > tol)
            {
                StepData<T> next_data = this->step(
                    func, i, quasi_newton, curr_data, delta, beta, verbose,
                    hessian_modify_max_iter
                ); 
                change = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).template cast<T>().norm();
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

/**
 * An implementation of *line-search* sequential quadratic programming for
 * nonlinear optimization on convex polytopes (i.e., linear inequality
 * constraints). 
 */
template <typename T>
class LineSearchSQPOptimizer : public SQPOptimizer<T>
{
    public:
        /**
         * All inherited constructors from `SQPOptimizer`. 
         */
        LineSearchSQPOptimizer(const unsigned N) : SQPOptimizer<T>(N)
        {
        }
        
        LineSearchSQPOptimizer(const unsigned D, const unsigned N,
                               const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                               const Ref<const Matrix<T, Dynamic, 1> >& b)
            : SQPOptimizer<T>(D, N, A, b)
        {
        }

        LineSearchSQPOptimizer(const unsigned D, const unsigned N,
                               const Polytopes::InequalityType type, 
                               const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                               const Ref<const Matrix<T, Dynamic, 1> >& b)
            : SQPOptimizer<T>(D, N, type, A, b) 
        {
        }

        LineSearchSQPOptimizer(Polytopes::LinearConstraints<mpq_rational>* constraints)
            : SQPOptimizer<T>(constraints)
        {
        }

        /**
         * Run one step of the line-search SQP algorithm.
         *
         * 1) Given an input vector `(x,l)`, compute `f(x)` and `df(x)/dx`. 
         * 2) Compute the Lagrangian, `L(x,l) = f(x) - l.T * A * x`, where `A` is the
         *    constraint matrix, and its Hessian matrix of second derivatives w.r.t. `x`.
         *    - Use a quasi-Newton method to compute the Hessian if desired.
         *    - If the Hessian is not positive-definite, perturb by a small multiple
         *      of the identity until it is positive-definite. 
         * 3) Define the quadratic subproblem according to the above quantities and
         *    the constraints (see below). 
         * 4) Solve the quadratic subproblem.
         * 5) Check that the newly incremented vector satisfies the merit function 
         *    constraint, iteratively decreasing the increment until it does.  
         * 6) Check that the new vector satisfies the constraints of the original
         *    problem, and output the new vector.
         */
        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                         const unsigned iter, const QuasiNewtonMethod quasi_newton,
                         StepData<T> prev_data, const T eta, const T tau, 
                         const T delta, const T beta, const bool verbose,
                         const unsigned hessian_modify_max_iter)
        {
            T f = prev_data.f; 
            Matrix<T, Dynamic, 1> xl = prev_data.xl;
            Matrix<T, Dynamic, 1> x = xl.head(this->D);
            Matrix<T, Dynamic, 1> l = xl.tail(this->N); 
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
                try
                {
                    for (unsigned i = 0; i < this->D; ++i)
                    {
                        for (unsigned j = 0; j <= i; ++j)
                        {
                            this->program->set_d(i, j, 2.0);    // Sets 2D_ij and 2D_ji
                        }
                    }
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
            Matrix<T, Dynamic, 1> sol(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                sol(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Collect the values of the new Lagrange multipliers (i.e., the
            // "optimality certificate")
            Matrix<T, Dynamic, 1> mult(this->N);
            i = 0;
            for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
            {
                mult(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }

            // Compute the difference in Lagrange multiplier values 
            Matrix<T, Dynamic, 1> L_delta = mult - l; 

            // Compute a value of mu that satisfies the inequality in Nocedal and
            // Wright, Eq. 18.36
            T sigma = 1;
            T rho = 0.5; 
            T epsilon = 1e-3;
            T c_norm = c.array().abs().sum();
            T df_dot_sol = df.dot(sol);  
            T mu = (
                (1 + epsilon) * (df_dot_sol + (sigma / 2) * sol.dot(d2L * sol))
                / ((1 - rho) * c_norm)
            );

            // Set initial stepsize
            T stepsize = 1;

            // Evaluate the merit function at the incremented input vector with the 
            // current stepsize
            Matrix<T, Dynamic, 1> x_new = x + stepsize * sol;
            T merit_old = this->meritL1(func, x, mu); 
            T merit_new = this->meritL1(func, x_new, mu);
            T factor = tau / 2;
            T dir_deriv = df_dot_sol - mu * c_norm;  
            while (merit_new > merit_old + eta * stepsize * dir_deriv)
            {
                stepsize *= factor;
                x_new = x + stepsize * sol;
                merit_new = this->meritL1(func, x_new, mu);
                factor /= 2;  
            }  

            // Increment the input vector and update the Lagrange multipliers
            Matrix<T, Dynamic, 1> xl_new(this->D + this->N);
            xl_new.head(this->D) = x_new; 
            xl_new.tail(this->N) = l + stepsize * mult;
            
            // Print the new vector and value of the objective function
            T f_new = func(x_new); 
            if (verbose)
            {
                std::cout << "Iteration " << iter << ": x = " << x_new.transpose() 
                          << "; f(x) = " << f_new 
                          << "; change = " << f_new - f << std::endl; 
            }

            // Evaluate the Hessian of the Lagrangian (with respect to the input space)
            Matrix<T, Dynamic, 1> df_new = this->gradient(func, x_new, delta); 
            Matrix<T, Dynamic, 1> xl_mixed(xl);
            xl_mixed.tail(this->N) = mult; 
            Matrix<T, Dynamic, 1> dL_mixed = this->lagrangianGradient(func, x, mult, delta); 
            Matrix<T, Dynamic, 1> dL_new = this->lagrangianGradient(func, x_new, mult, delta); 
            Matrix<T, Dynamic, Dynamic> d2L_new;
            Matrix<T, Dynamic, 1> s = x_new - x; 
            Matrix<T, Dynamic, 1> y = dL_new.head(this->D) - dL_mixed.head(this->D);
            auto d2L_ = d2L.template selfadjointView<Lower>(); 
            switch (quasi_newton)
            {
                case BFGS:
                    d2L_new = updateBFGSDamped<T>(d2L_, s, y); 
                    break;

                case SR1:
                    d2L_new = updateSR1<T>(d2L_, s, y); 
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
                                  const T eta, const T tau, const T delta, const T beta,
                                  const unsigned max_iter, const T tol,
                                  const QuasiNewtonMethod quasi_newton,
                                  const bool verbose,
                                  const unsigned hessian_modify_max_iter)
        {
            // Evaluate the objective and its gradient
            T f = func(x_init);
            Matrix<T, Dynamic, 1> df = this->gradient(func, x_init, delta);

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
            auto lagrangian = [this, &func](const Ref<const Matrix<T, Dynamic, 1> >& x,
                                            const Ref<const Matrix<T, Dynamic, 1> >& l)
                {
                    Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().template cast<T>(); 
                    Matrix<T, Dynamic, 1> b = this->constraints->getb().template cast<T>();
                    Polytopes::InequalityType type = this->constraints->getInequalityType(); 
                    if (type == Polytopes::InequalityType::GreaterThanOrEqualTo)
                        return func(x) - l.dot(A * x - b);
                    else 
                        return func(x) + l.dot(A * x - b);  
                }; 
            T L = lagrangian(x_init, l_init);
            Matrix<T, Dynamic, 1> dL = this->lagrangianGradient(func, x_init, l_init, delta);
            curr_data.dL = dL;
            curr_data.d2L = Matrix<T, Dynamic, Dynamic>::Identity(this->D, this->D);

            unsigned i = 0;
            T change = 2 * tol;
            while (i < max_iter && change > tol)
            {
                StepData<T> next_data = this->step(
                    func, i, quasi_newton, curr_data, eta, tau, delta, beta,
                    verbose, hessian_modify_max_iter
                );
                change = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).template cast<T>().norm();
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
