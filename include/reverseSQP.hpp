#ifndef SQP_OPTIMIZER_REVERSE_MODE_HPP
#define SQP_OPTIMIZER_REVERSE_MODE_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <autodiff/reverse/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include "linearConstraints.hpp"
#include "quasiNewton.hpp"

/*
 * Class template specialization of SQPOptimizer for reverse-mode variables.
 *
 * Authors: 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */
using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program_solution<ET> Solution;

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
     * Run one step of the SQP algorithm with reverse-mode variables.
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

#endif
