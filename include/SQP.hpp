/**
 * An implementation of a nonlinear optimizer with respect to linear 
 * constraints with sequential quadratic programming (SQP) with automatic
 * differentiation and/or quasi-Newton Hessian approximations. 
 *
 * **Authors:** 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     4/11/2022
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
#include "linearConstraints.hpp"
#include "quasiNewton.hpp"

using namespace Eigen;
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
        VectorXd df;
        VectorXd dL;
        MatrixXd d2L;

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

MatrixXd modify(const Ref<const MatrixXd>& A, unsigned max_iter, double beta = 1e-3)
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
    // Check that A is positive definite with the Cholesky decomposition
    LLT<MatrixXd> dec(A);
    bool posdef = (dec.info() == Success);
    if (posdef) return A; 

    // TODO: Make this customizable
    MatrixXd B(A);
    double tau = 0.0;
    for (unsigned i = 0; i < max_iter; ++i)
    {
        // Add beta to the diagonal ...
        if (tau == 0.0)
            tau = B.cwiseAbs().diagonal().minCoeff() + beta;
        else
            tau *= 2.0;
        B += tau * MatrixXd::Identity(B.rows(), B.cols());
        dec.compute(B);

        // ... until the matrix is positive definite
        posdef = (dec.info() == Success);

        if (posdef) break;    // If so, then we are done 
    }

    return B;
}

template <typename T>
class SQPOptimizer
{
    /*
     * A lightweight implementation of sequential quadratic programming 
     * for nonlinear optimization on convex polytopes (i.e., linear 
     * inequality constraints). 
     */
    private:
        unsigned D;                                                 // Dimension of input space
        unsigned N;                                                 // Number of constraints 
        Polytopes::LinearConstraints<mpq_rational>* constraints;    // Linear inequality constraints
        Program* program;                                           // Internal quadratic program to
                                                                    // be solved at each step

    public:
        /**
         * Straightforward constructor with `N` variables.
         *
         * Each variable is constrained to be greater than or equal to zero. 
         */
        SQPOptimizer(const unsigned N)
        {
            this->D = D;
            this->N = D;    // One constraint for each variable
            MatrixXd A = MatrixXd::Identity(D, D); 
            VectorXd b = VectorXd::Zero(D); 
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
        }

        /**
         * Constructor with constraint matrix and vector specified, and
         * inequality set to greater-than-or-equal-to. 
         */
        SQPOptimizer(const unsigned D, const unsigned N,
                     const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            this->D = D;
            this->N = N;
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(
                Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
            ); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
        }

        /**
         * Constructor with constraint matrix, vector, and inequality type
         * specified. 
         */
        SQPOptimizer(const unsigned D, const unsigned N,
                     const Polytopes::InequalityType type, 
                     const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            this->D = D;
            this->N = N;
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(type, A, b); 
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
        }

        /**
         * Constructor with linear constraints specified via a `LinearConstraints`
         * instance. 
         */
        SQPOptimizer(const LinearConstraints<mpq_rational>* constraints)
        {
            this->constraints = constraints;
            this->D = constraints->getD(); 
            this->N = constraints->getN(); 
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
         */
        void setConstraints(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& A,
                            const Ref<const Matrix<mpq_rational, Dynamic, 1> >& b)
        {
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints->setAb(A, b);
        }

        std::pair<T, VectorXd> func_with_gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                                  const Ref<const Matrix<T, Dynamic, 1> >& x); 

        std::pair<T, VectorXd> lagrangian_with_gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                                        const Ref<const Matrix<T, Dynamic, 1> >& xl);

        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                         const unsigned iter, const QuasiNewtonMethod quasi_newton,
                                         const StepData<T> prev_data, const bool verbose,
                                         const unsigned hessian_modify_max_iter);

        VectorXd run(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                     const Ref<const Matrix<T, Dynamic, 1> >& xl_init,
                     const unsigned max_iter, const double tol,
                     const QuasiNewtonMethod quasi_newton, const bool verbose)
        {
            /*
             * Run the optimization with the given objective, initial vector,
             * and settings.   
             */
            // Print the input vector and value of the objective function
            if (verbose)
            {
                std::cout << "Initial vector: x = " << xl_init.head(this->D).transpose()
                          << "; " << "f(x) = " << func(xl_init.head(this->D)) << std::endl; 
            }

            // Evaluate the objective and its gradient
            Matrix<T, Dynamic, 1> x = xl_init.head(this->D);
            std::pair<T, VectorXd> grad = this->func_with_gradient(func, x);
            T f = grad.first;
            VectorXd df = grad.second;
            
            // Evaluate the Lagrangian and its gradient
            std::pair<T, VectorXd> lagr = this->lagrangian_with_gradient(func, xl_init);
            T L = lagr.first;
            VectorXd dL = lagr.second.head(this->D);

            // Collect current objective, gradient, and Hessian information
            StepData<T> curr_data;
            curr_data.f = f; 
            curr_data.xl = xl_init;
            curr_data.df = df;
            curr_data.dL = dL;
            curr_data.d2L = MatrixXd::Identity(this->D, this->D);

            unsigned i = 0;
            double delta = 2 * tol;
            unsigned hessian_modify_max_iter = max_iter; 
            while (i < max_iter && delta > tol)
            {
                StepData<T> next_data = this->step(func, i, quasi_newton, curr_data, verbose, hessian_modify_max_iter); 
                delta = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).template cast<double>().norm();
                i++;
                curr_data.f = next_data.f; 
                curr_data.xl = next_data.xl;
                curr_data.df = next_data.df;
                curr_data.dL = next_data.dL;
                curr_data.d2L = next_data.d2L;
            }
            return curr_data.xl.head(this->D).template cast<double>();
        }
};

// -------------------------------------------------------------- //
//         CLASS TEMPLATE SPECIALIZATION FOR PLAIN DOUBLES        //
//           WITH FINITE-DIFFERENCES GRADIENT ESTIMATION          //
// -------------------------------------------------------------- //
template <>
std::pair<double, VectorXd>
    SQPOptimizer<double>::func_with_gradient(std::function<double(const Ref<const VectorXd>&)> func,
                                             const Ref<const VectorXd>& x)
{
    /*
     * Compute the given function and its gradient at the given vector,
     * with delta = 1e-7 for finite difference approximation. 
     */
    const double delta = 1e-7;
    
    // Evaluate the function at 2 * D values, with each coordinate 
    // perturbed by +/- delta
    VectorXd grad(this->D);
    for (unsigned i = 0; i < this->D; ++i)
    {
        VectorXd y(x);
        y(i) += delta;
        double f1 = func(y);
        y(i) -= 2 * delta;
        double f2 = func(y);
        grad(i) = (f1 - f2) / (2 * delta);
    }
    return std::make_pair(func(x), grad);
}

template <>
std::pair<double, VectorXd>
    SQPOptimizer<double>::lagrangian_with_gradient(std::function<double(const Ref<const VectorXd>&)> func,
                                                   const Ref<const VectorXd>& xl)
{
    /*
     * Compute the Lagrangian of the given function and its gradient at
     * the given vector, with delta = 1e-7 for finite difference
     * approximation.
     */
    const double delta = 1e-7;

    VectorXd x = xl.head(this->D);
    VectorXd l = xl.tail(this->N);
    MatrixXd A = this->constraints->getA().template cast<double>();
    VectorXd b = this->constraints->getb().template cast<double>();
    double L = func(x) - l.dot(A * x - b);

    // Evaluate the Lagrangian at 2 * D values, with each coordinate 
    // perturbed by +/- delta
    VectorXd dL(this->D + this->N);
    for (unsigned i = 0; i < this->D + this->N; ++i)
    {
        VectorXd y(xl);
        y(i) += delta;
        double f1 = func(y.head(this->D)) - y.tail(this->N).dot(A * y.head(this->D) - b);
        y(i) -= 2 * delta;
        double f2 = func(y.head(this->D)) - y.tail(this->N).dot(A * y.head(this->D) - b);
        dL(i) = (f1 - f2) / (2 * delta);
    }
    return std::make_pair(L, dL);
}

template <>
StepData<double> SQPOptimizer<double>::step(std::function<double(const Ref<const VectorXd>&)> func,
                                            const unsigned iter,
                                            const QuasiNewtonMethod quasi_newton,
                                            StepData<double> prev_data,
                                            const bool verbose,
                                            const unsigned hessian_modify_max_iter)
{
    /*
     * Run one step of the SQP algorithm with double scalars.
     *
     * 1) Given an input vector xl = (x,l) with this->D + this->N
     *    coordinates, compute f(x) and df(x)/dx. 
     * 2) Compute the Lagrangian, L(x,l) = f(x) - l.T * A * x, where
     *    A is the constraint matrix, and its Hessian matrix of 
     *    second derivatives w.r.t. x.
     *    - Use a quasi-Newton method to compute the Hessian if desired.
     *    - If the Hessian is not positive definite, perturb by 
     *      a small multiple of the identity until it is positive
     *      definite. 
     * 3) Define the quadratic subproblem according to the above
     *    quantities and the constraints (see below). 
     * 4) Solve the quadratic subproblem, check that the new vector
     *    satisfies the constraints of the original problem, and 
     *    output the new vector.
     */
    double f = prev_data.f; 
    VectorXd xl = prev_data.xl;
    VectorXd x = xl.head(this->D);
    VectorXd df = prev_data.df;
    VectorXd dL = prev_data.dL;
    MatrixXd d2L = modify(prev_data.d2L, hessian_modify_max_iter);

    // If any of the components have a non-finite coordinate, return as is
    if (!x.array().isFinite().all() || !df.array().isFinite().all() || !dL.array().isFinite().all() || !d2L.array().isFinite().all())
        return prev_data;

    // Evaluate the constraints and their gradients
    MatrixXd A = this->constraints->getA().template cast<double>();
    VectorXd c = -(A * x.cast<double>() - this->constraints->getb().template cast<double>());

    // Set up the quadratic program:
    //
    // Minimize x.T D x + c.T x + c0 = fk + Dfk.T p + 0.5 * p.T D2Lk p
    // subject to A p + (A xk - b) >= 0
    //
    // where:
    //
    // p = variable to be optimized -- vector to be added to current iterate
    // xk = current iterate 
    // fk = f(xk)
    // Dfk = gradient of f at xk
    // D2Lk = (approximation of) Hessian of Lagrangian at xk 
    //
    // Note that, with the damped BFGS update, the Hessian matrix approximation
    // should be positive definite
    for (unsigned i = 0; i < this->D; ++i)
    {
        for (unsigned j = 0; j <= i; ++j)
        {
            this->program->set_d(i, j, d2L(i,j));    // Sets 2D_ij and 2D_ji (the quadratic part of objective)
        }
        this->program->set_c(i, df(i));              // Sets c_i (the linear part of objective)
    }
    for (unsigned i = 0; i < this->N; ++i)
    {
        for (unsigned j = 0; j < this->D; ++j)
        {
            this->program->set_a(j, i, A(i,j));      // Sets A_ij (j-th coefficient of i-th constraint)
        }
        this->program->set_b(i, c(i));               // Sets b_i (i-th coordinate of -(A xk - b))
    }
    this->program->set_c0(f);                        // Sets constant part of objective (fk)

    // Solve the quadratic program ...
    Solution solution; 
    try
    {
        solution = CGAL::solve_quadratic_program(*this->program, ET());
    }
    catch (CGAL::Assertion_exception& e) 
    {
        // ... if the program cannot be solved because the D matrix is not 
        // positive semi-definite (this should never be the case), then replace
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

    // Collect the values of the solution into a VectorXd
    VectorXd sol(this->D);
    unsigned i = 0;
    for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
    {
        sol(i) = CGAL::to_double(*it);
        i++;
    }

    // Collect the values of the new Lagrange multipliers (i.e., the
    // "optimality certificate")
    VectorXd mult(this->N);
    i = 0;
    for (auto it = solution.optimality_certificate_begin(); it != solution.optimality_certificate_end(); ++it)
    {
        mult(i) = CGAL::to_double(*it);
        i++;
    }

    // Increment the input vector and update the Lagrange multipliers
    VectorXd xl_new(this->D + this->N);
    xl_new.head(this->D) = xl.head(this->D) + sol;
    xl_new.tail(this->N) = mult;

    // Print the new vector and value of the objective function
    if (verbose)
    {
        std::cout << "Iteration " << iter << ": x = " << xl_new.head(this->D).transpose()
                  << "; " << "f(x) = " << func(xl_new.head(this->D)) << std::endl; 
    }

    // Evaluate the Hessian of the Lagrangian (with respect to 
    // the input space)
    VectorXd x_new = xl_new.head(this->D);
    VectorXd df_new = this->func_with_gradient(func, x_new).second;
    VectorXd xl_mixed(xl);
    xl_mixed.tail(this->N) = xl_new.tail(this->N);
    std::pair<double, VectorXd> lagr_mixed = this->lagrangian_with_gradient(func, xl_mixed);
    std::pair<double, VectorXd> lagr_new = this->lagrangian_with_gradient(func, xl_new);
    double L_new = lagr_new.first;
    VectorXd dL_new = lagr_new.second.head(this->D);
    MatrixXd d2L_new;
    VectorXd s, y; 
    switch (quasi_newton)
    {
        case BFGS:
            s = x_new - x;
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = updateBFGSDamped<double>(d2L, s, y);
            break;

        case SR1:
            s = x_new - x;
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = updateSR1<double>(d2L, s, y);
            break;

        default:
            break;
    } 

    // Return the new data
    StepData<double> new_data;
    new_data.xl = xl_new;
    new_data.df = df_new;
    new_data.dL = dL_new;
    new_data.d2L = d2L_new;
    return new_data;
}

#endif 
