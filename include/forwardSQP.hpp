#ifndef SQP_OPTIMIZER_FORWARD_MODE_HPP
#define SQP_OPTIMIZER_FORWARD_MODE_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include "linearConstraints.hpp"
#include "quasiNewton.hpp"

/*
 * Class template specialization of SQPOptimizer for forward-mode variables.
 *
 * NOTE: Any code that instantiates SQPOptimizer<autodiff::dual> is not
 * expected to compile until issues related to metaprogramming in autodiff
 * are resolved (https://github.com/autodiff/autodiff/issues/62).
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
std::pair<autodiff::dual, VectorXd>
    SQPOptimizer<autodiff::dual>::func_with_gradient(std::function<autodiff::dual(const Ref<const VectorXdual>&)> func,
                                                     const Ref<const VectorXdual>& x)
{
    /*
     * Compute the given function and its gradient at the given vector.
     */
    autodiff::dual f;
    VectorXd df = autodiff::forward::gradient(func, autodiff::forward::wrt(x), autodiff::forward::at(x), f);
    return std::make_pair(f, df);
}

template <>
std::pair<autodiff::dual, VectorXd>
    SQPOptimizer<autodiff::dual>::lagrangian_with_gradient(std::function<autodiff::dual(const Ref<const VectorXdual>&)> func,
                                                           const Ref<const VectorXdual>& xl)
{
    /*
     * Compute the Lagrangian of the given function and its gradient at
     * the given vector.
     */
    VectorXdual x = xl.head(this->D);
    VectorXdual l = xl.tail(this->N);
    MatrixXdual A = this->constraints->getA().cast<autodiff::dual>();
    VectorXdual b = this->constraints->getb().cast<autodiff::dual>();
    autodiff::dual L;
    VectorXd dL = autodiff::forward::gradient(
        [func, l, A, b](const VectorXdual& a){ func(a) - l.dot(A * a - b); }, autodiff::forward::wrt(x), autodiff::forward::at(x), L
    );
    return std::make_pair(L, dL);
} 

template <>
StepData<autodiff::dual> SQPOptimizer<autodiff::dual>::step(std::function<autodiff::dual(const Ref<const VectorXdual>&)> func,
                                                            const unsigned iter,
                                                            const QuasiNewtonMethod quasi_newton,
                                                            StepData<autodiff::dual> prev_data,
                                                            const bool verbose)
{
    /*
     * Run one step of the SQP algorithm with forward-mode variables.
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
    VectorXdual xl = prev_data.xl;
    VectorXdual x = xl.head(this->D);
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
    VectorXdual xl_new(this->D + this->N);
    xl_new.head(this->D) = xl.head(this->D) + sol.cast<autodiff::dual>();
    xl_new.tail(this->N) = mult.cast<autodiff::dual>();

    // Print the new vector and value of the objective function
    if (verbose)
    {
        std::cout << "Iteration " << iter << ": x = " << xl_new.head(this->D).transpose()
                  << "; " << "f(x) = " << func(xl_new.head(this->D)) << std::endl; 
    }

    // Evaluate the Hessian of the Lagrangian (with respect to 
    // the input space)
    VectorXdual x_new = xl_new.head(this->D);
    VectorXd df_new = this->func_with_gradient(func, x_new).second;
    VectorXdual xl_mixed(xl);
    xl_mixed.tail(this->N) = xl_new.tail(this->N);
    std::pair<autodiff::dual, VectorXd> lagr_mixed = this->lagrangian_with_gradient(func, xl_mixed);
    std::pair<autodiff::dual, VectorXd> lagr_new = this->lagrangian_with_gradient(func, xl_new);
    autodiff::dual L_new = lagr_new.first;
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
    StepData<autodiff::dual> new_data;
    new_data.xl = xl_new;
    new_data.df = df_new;
    new_data.dL = dL_new;
    new_data.d2L = d2L_new;
    return new_data;
}

#endif
