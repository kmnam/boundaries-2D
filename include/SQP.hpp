#ifndef SQP_OPTIMIZER_HPP
#define SQP_OPTIMIZER_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include "linearConstraints.hpp"
#include "quasiNewton.hpp"

/*
 * An implementation of a nonlinear optimizer with respect to linear 
 * constraints with sequential quadratic programming (SQP) with automatic
 * differentiation and/or quasi-Newton Hessian approximations. 
 *
 * Authors: 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */
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

MatrixXd modify(const Ref<const MatrixXd>& A)
{
    /*
     * Following the prescription of Nocedal and Wright (Algorithm 3.3, p.51), 
     * add successive multiples of the identity until the given matrix is 
     * positive-definite.
     *
     * The input matrix is assumed to be symmetric. 
     */
    // Check that A is positive definite with the Cholesky decomposition
    LLT<MatrixXd> dec(A);
    bool posdef = (dec.info() == Success);

    // TODO Make this customizable
    MatrixXd B(A);
    double tau = 0.0;
    while (!posdef)
    {
        double beta = 1e-3;
        if (tau == 0.0)
            tau = B.cwiseAbs().diagonal().minCoeff() + beta;
        else
            tau *= 2.0;
        B += tau * MatrixXd::Identity(B.rows(), B.cols());
        dec.compute(B);
        posdef = (dec.info() == Success);
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
        unsigned D;                        // Dimension of input space
        unsigned N;                        // Number of constraints 
        LinearConstraints* constraints;    // Linear inequality constraints
        Program* program;                  // Internal quadratic program to be solved at each step

    public:
        SQPOptimizer(unsigned D, unsigned N, const Ref<const MatrixXd>& A,
                     const Ref<const VectorXd>& b)
        {
            /*
             * Straightforward constructor.
             */
            this->D = D;
            this->N = N;
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new LinearConstraints(A, b);
            this->program = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
        }

        ~SQPOptimizer()
        {
            /*
             * Destructor; deallocates the LinearConstraints object.
             */
            delete this->constraints;
            delete this->program;
        }

        void setConstraints(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Update the stored linear constraints. 
             */
            if (A.rows() != this->N || A.cols() != this->D || b.size() != this->N)
                throw std::invalid_argument("Invalid input matrix dimensions");
            this->constraints = new LinearConstraints(A, b);
        }

        VectorXd gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                          const Ref<const Matrix<T, Dynamic, 1> >& x);

        std::pair<T, VectorXd> func_with_gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                                  const Ref<const Matrix<T, Dynamic, 1> >& x); 

        T lagrangian(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                     const Ref<const Matrix<T, Dynamic, 1> >& xl)
        {
            /*
             * Given a vector xl of dimension D + N, whose first D coordinates
             * specify the values of the input space and the latter N
             * coordinates specify the values of the Lagrange multipliers,
             * evaluate the Lagrangian of the objective function. 
             */
            Matrix<T, Dynamic, 1> x = xl.head(this->D);
            Matrix<T, Dynamic, 1> l = xl.tail(this->N);
            Matrix<T, Dynamic, Dynamic> A = this->constraints->getA().cast<T>();
            Matrix<T, Dynamic, 1> b = this->constraints->getb().cast<T>();
            return func(x) - l.dot(A * x - b);
        }

        std::pair<T, VectorXd> lagrangian_with_gradient(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                                        const Ref<const Matrix<T, Dynamic, 1> >& xl);

        StepData<T> step(std::function<T(const Ref<const Matrix<T, Dynamic, 1> >&)> func,
                                         const unsigned iter, const QuasiNewtonMethod quasi_newton,
                                         const StepData<T> prev_data, const bool verbose);

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
            curr_data.xl = xl_init;
            curr_data.df = df;
            curr_data.dL = dL;
            curr_data.d2L = MatrixXd::Identity(this->D, this->D);

            unsigned i = 0;
            double delta = 2 * tol;
            while (i < max_iter && delta > tol)
            {
                StepData<T> next_data = this->step(func, i, quasi_newton, curr_data, verbose); 
                delta = (curr_data.xl.head(this->D) - next_data.xl.head(this->D)).template cast<double>().norm();
                i++;
                curr_data.xl = next_data.xl;
                curr_data.df = next_data.df;
                curr_data.dL = next_data.dL;
                curr_data.d2L = next_data.d2L;
            }
            return curr_data.xl.head(this->D).template cast<double>();
        }
};

// -------------------------------------------------------------- //
//    CLASS TEMPLATE SPECIALIZATION FOR REVERSE-MODE VARIABLES    //
// -------------------------------------------------------------- //
#include <autodiff/reverse/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

template <>
std::pair<autodiff::var, VectorXd>
    SQPOptimizer<autodiff::var>::func_with_gradient(std::function<autodiff::var(const Ref<const VectorXvar>&)> func,
                                                    const Ref<const VectorXvar>& x)
{
    /*
     * Compute the given function and its gradient at the given vector.
     */
    autodiff::var f = func(x);
    VectorXd df = autodiff::gradient(f, x);
    return std::make_pair(f, df);
}

template <>
std::pair<autodiff::var, VectorXd>
    SQPOptimizer<autodiff::var>::lagrangian_with_gradient(std::function<autodiff::var(const Ref<const VectorXvar>&)> func,
                                                          const Ref<const VectorXvar>& xl)
{
    /*
     * Compute the Lagrangian of the given function and its gradient at
     * the given vector.
     */
    VectorXvar x = xl.head(this->D);
    VectorXvar l = xl.tail(this->N);
    MatrixXvar A = this->constraints->getA().cast<autodiff::var>();
    VectorXvar b = this->constraints->getb().cast<autodiff::var>();
    autodiff::var L = func(x) - l.dot(A * x - b);
    VectorXd dL = autodiff::gradient(L, xl);
    return std::make_pair(L, dL);
} 

template <>
StepData<autodiff::var> SQPOptimizer<autodiff::var>::step(std::function<autodiff::var(const Ref<const VectorXvar>&)> func,
                                                          const unsigned iter,
                                                          const QuasiNewtonMethod quasi_newton,
                                                          StepData<autodiff::var> prev_data,
                                                          const bool verbose)
{
    /*
     * Run one step of the SQP algorithm with autodiff::var scalars.
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
    VectorXvar xl = prev_data.xl;
    VectorXvar x = xl.head(this->D);
    VectorXd df = prev_data.df;
    VectorXd dL = prev_data.dL;
    MatrixXd d2L = prev_data.d2L;

    // Evaluate the constraints and their gradients
    MatrixXd A = this->constraints->getA();
    VectorXd c = -(A * x.cast<double>() - this->constraints->getb());

    // Set up the quadratic program 
    for (unsigned i = 0; i < this->D; ++i)
    {
        for (unsigned j = 0; j <= i; ++j)
        {
            this->program->set_d(i, j, d2L(i,j)); 
        }
        this->program->set_c(i, df(i));
    }
    for (unsigned i = 0; i < this->N; ++i)
    {
        for (unsigned j = 0; j < this->D; ++j)
        {
            this->program->set_a(j, i, A(i,j));
        }
        this->program->set_b(i, c(i));
    }
    this->program->set_c0(0.0); // TODO Does this matter?

    // Solve the quadratic program 
    Solution solution = CGAL::solve_quadratic_program(*this->program, ET());

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

    // Check that the solution satisfies the original constraints
    bool feasible = this->constraints->check(xl.head(this->D).cast<double>() + sol);
    if (!feasible)
    {
        // TODO Figure out what to do here 
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
    VectorXvar xl_new(this->D + this->N);
    xl_new.head(this->D) = xl.head(this->D) + sol.cast<autodiff::var>();
    xl_new.tail(this->N) = mult.cast<autodiff::var>();

    // Print the new vector and value of the objective function
    if (verbose)
    {
        std::cout << "Iteration " << iter << ": x = " << xl_new.head(this->D).transpose()
                  << "; " << "f(x) = " << func(xl_new.head(this->D)) << std::endl; 
    }

    // Evaluate the Hessian of the Lagrangian (with respect to 
    // the input space)
    VectorXvar x_new = xl_new.head(this->D);
    VectorXd df_new = this->func_with_gradient(func, x_new).second;
    VectorXvar xl_mixed(xl);
    xl_mixed.tail(this->N) = xl_new.tail(this->N);
    std::pair<autodiff::var, VectorXd> lagr_mixed = this->lagrangian_with_gradient(func, xl_mixed);
    std::pair<autodiff::var, VectorXd> lagr_new = this->lagrangian_with_gradient(func, xl_new);
    autodiff::var L_new = lagr_new.first;
    VectorXd dL_new = lagr_new.second.head(this->D);
    MatrixXd d2L_new;
    VectorXd s, y; 
    switch (quasi_newton)
    {
        case NONE:
            d2L_new = modify(autodiff::hessian(L_new, xl_new).block(0, 0, this->D, this->D));
            break;

        case BFGS:
            s = (x_new - x).cast<double>();
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = modify(updateBFGS<double>(d2L, s, y));
            break;

        case SR1:
            s = (x_new - x).cast<double>();
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = modify(updateSR1<double>(d2L, s, y));
            break;

        default:
            break;
    } 

    // Return the new data
    StepData<autodiff::var> new_data;
    new_data.xl = xl_new;
    new_data.df = df_new;
    new_data.dL = dL_new;
    new_data.d2L = d2L_new;
    return new_data;
}

// -------------------------------------------------------------- //
//    CLASS TEMPLATE SPECIALIZATION FOR FORWARD-MODE VARIABLES    //
// -------------------------------------------------------------- //
#include "duals/duals.hpp"
#include "duals/eigen.hpp"

using Duals::DualNumber;

template <>
std::pair<DualNumber, VectorXd>
    SQPOptimizer<DualNumber>::func_with_gradient(std::function<DualNumber(const Ref<const VectorXDual>&)> func,
                                                 const Ref<const VectorXDual>& x)
{
    /*
     * Compute the given function and its gradient at the given vector.
     */
    DualNumber f;
    VectorXd df = Duals::gradient(func, x, f);
    return std::make_pair(f, df);
}

template <>
std::pair<DualNumber, VectorXd>
    SQPOptimizer<DualNumber>::lagrangian_with_gradient(std::function<DualNumber(const Ref<const VectorXDual>&)> func,
                                                       const Ref<const VectorXDual>& xl)
{
    /*
     * Compute the Lagrangian of the given function and its gradient at
     * the given vector.
     */
    VectorXDual x = xl.head(this->D);
    VectorXDual l = xl.tail(this->N);
    MatrixXDual A = this->constraints->getA().cast<DualNumber>();
    VectorXDual b = this->constraints->getb().cast<DualNumber>();
    DualNumber L;
    std::function<DualNumber(const Ref<const VectorXDual>&)> lagr = [func, l, A, b](const Ref<const VectorXDual>& a)
    {
        return func(a) - l.dot(A * a - b);
    };
    VectorXd dL = Duals::gradient(lagr, x, L);
    return std::make_pair(L, dL);
} 

template <>
StepData<DualNumber> SQPOptimizer<DualNumber>::step(std::function<DualNumber(const Ref<const VectorXDual>&)> func,
                                                    const unsigned iter,
                                                    const QuasiNewtonMethod quasi_newton,
                                                    StepData<DualNumber> prev_data,
                                                    const bool verbose)
{
    /*
     * Run one step of the SQP algorithm with Duals::DualNumber scalars.
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
    VectorXDual xl = prev_data.xl;
    VectorXDual x = xl.head(this->D);
    VectorXd df = prev_data.df;
    VectorXd dL = prev_data.dL;
    MatrixXd d2L = prev_data.d2L;

    // Evaluate the constraints and their gradients
    MatrixXd A = this->constraints->getA();
    VectorXd c = -(A * x.cast<double>() - this->constraints->getb());

    // Set up the quadratic program 
    for (unsigned i = 0; i < this->D; ++i)
    {
        for (unsigned j = 0; j <= i; ++j)
        {
            this->program->set_d(i, j, d2L(i,j)); 
        }
        this->program->set_c(i, df(i));
    }
    for (unsigned i = 0; i < this->N; ++i)
    {
        for (unsigned j = 0; j < this->D; ++j)
        {
            this->program->set_a(j, i, A(i,j));
        }
        this->program->set_b(i, c(i));
    }
    this->program->set_c0(0.0); // TODO Does this matter?

    // Solve the quadratic program 
    Solution solution = CGAL::solve_quadratic_program(*this->program, ET());

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

    // Check that the solution satisfies the original constraints
    bool feasible = this->constraints->check(xl.head(this->D).cast<double>() + sol);
    if (!feasible)
    {
        // TODO Figure out what to do here 
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
    VectorXDual xl_new(this->D + this->N);
    xl_new.head(this->D) = xl.head(this->D) + sol.cast<DualNumber>();
    xl_new.tail(this->N) = mult.cast<DualNumber>();

    // Print the new vector and value of the objective function
    if (verbose)
    {
        std::cout << "Iteration " << iter << ": x = " << xl_new.head(this->D).transpose()
                  << "; " << "f(x) = " << func(xl_new.head(this->D)) << std::endl; 
    }

    // Evaluate the Hessian of the Lagrangian (with respect to 
    // the input space)
    VectorXDual x_new = xl_new.head(this->D);
    VectorXd df_new = this->func_with_gradient(func, x_new).second;
    VectorXDual xl_mixed(xl);
    xl_mixed.tail(this->N) = xl_new.tail(this->N);
    std::pair<DualNumber, VectorXd> lagr_mixed = this->lagrangian_with_gradient(func, xl_mixed);
    std::pair<DualNumber, VectorXd> lagr_new = this->lagrangian_with_gradient(func, xl_new);
    DualNumber L_new = lagr_new.first;
    VectorXd dL_new = lagr_new.second.head(this->D);
    MatrixXd d2L_new;
    VectorXd s, y; 
    switch (quasi_newton)
    {
        case BFGS:
            s = (x_new - x).cast<double>();
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = modify(updateBFGS<double>(d2L, s, y));
            break;

        case SR1:
            s = (x_new - x).cast<double>();
            dL = lagr_mixed.second.head(this->D);
            y = dL_new - dL; 
            d2L_new = modify(updateSR1<double>(d2L, s, y));
            break;

        default:
            break;
    } 

    // Return the new data
    StepData<DualNumber> new_data;
    new_data.xl = xl_new;
    new_data.df = df_new;
    new_data.dL = dL_new;
    new_data.d2L = d2L_new;
    return new_data;
}


#endif 
