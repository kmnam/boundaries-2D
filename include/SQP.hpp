#ifndef SQP_OPTIMIZER_HPP
#define SQP_OPTIMIZER_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <autodiff/reverse/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include "linearConstraints.hpp"

/*
 * Authors: 
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/7/2019
 */
using namespace Eigen;
using namespace autodiff;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

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
        unsigned max_iter;                 // Maximum number of iterations 
        LinearConstraints* constraints;    // Linear inequality constraints
        Program* program;                  // Internal quadratic program to be solved at each step

    public:
        SQPOptimizer(unsigned D, unsigned N, unsigned max_iter,
                     const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Straightforward constructor.
             */
            this->D = D;
            this->N = N;
            this->max_iter = max_iter;
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

        var lagrangian(std::function<var(const Ref<const VectorXvar>&)> func,
                       const Ref<const VectorXvar>& xl)
        {
            /*
             * Given a vector xl of dimension D + N, whose first D coordinates
             * specify the values of the input space and the latter N
             * coordinates specify the values of the Lagrange multipliers,
             * evaluate the Lagrangian of the objective function. 
             */
            VectorXvar x = xl.head(this->D);
            VectorXvar l = xl.tail(this->N);
            MatrixXvar A = this->constraints->getA().cast<var>();
            VectorXvar b = this->constraints->getb().cast<var>();
            return func(x) - l.dot(A * x - b);
        }

        VectorXvar step(std::function<var(const Ref<const VectorXvar>&)> func,
                        const Ref<const VectorXvar>& xl, bool check_hessian_posdef,
                        unsigned iter, bool verbose)
        {
            /*
             * Run one step of the SQP algorithm: 
             * 1) Given an input vector xl = (x,l) with this->D + this->N
             *    coordinates, compute f(x) and df(x)/dx. 
             * 2) Compute the Lagrangian, L(x,l) = f(x) - l.T * A * x, where
             *    A is the constraint matrix, and its Hessian matrix of 
             *    second derivatives w.r.t. x.
             *    - If the Hessian is not positive definite, perturb by 
             *      a small multiple of the identity until it is positive
             *      definite. 
             * 3) Define the quadratic subproblem according to the above
             *    quantities and the constraints (see below). 
             * 4) Solve the quadratic subproblem, check that the new vector
             *    satisfies the constraints of the original problem, and 
             *    output the new vector.
             */
            // Check input vector dimensions
            if (xl.size() != this->D + this->N)
                throw std::invalid_argument("Invalid input vector dimensions");

            // Evaluate the objective and the Lagrangian 
            VectorXvar x = xl.head(this->D);
            var y = func(x);
            var L = lagrangian(func, xl);

            // Evaluate the gradient of the objective
            VectorXd dy = gradient(y, x);

            // Evaluate the Hessian of the Lagrangian (with respect to 
            // the input space) 
            MatrixXd d2L = hessian(L, xl).block(0, 0, this->D, this->D);

            // If desired, check that the Hessian is positive definite
            // with the Cholesky decomposition
            if (check_hessian_posdef)
            {
                LLT<MatrixXd> decomp(d2L);
                bool hessian_posdef = (decomp.info() == Success);

                // If the Hessian is not positive definite, follow the 
                // prescription by Nocedal & Wright (Alg.3.3, p.51) by
                // adding successive multiples of the identity
                double beta = 1e-3;
                double tau = 0.0;
                unsigned num_updates = 0;
                while (!hessian_posdef) // TODO Make this customizable
                {
                    if (tau == 0.0)
                        tau = std::abs(d2L.diagonal().minCoeff()) + beta;
                    else
                        tau *= 2.0;
                    d2L += tau * MatrixXd::Identity(this->D, this->D);
                    decomp.compute(d2L);
                    hessian_posdef = (decomp.info() == Success);
                }
            }

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
                this->program->set_c(i, dy(i));
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
            xl_new.head(this->D) = xl.head(this->D) + sol.cast<var>();
            xl_new.tail(this->N) = mult.cast<var>();

            // Print the new vector and value of the objective function
            if (verbose)
            {
                std::cout << "Iteration " << iter << ": x = " << xl_new.head(this->D).transpose()
                          << "; " << "f(x) = " << func(xl_new.head(this->D)) << std::endl; 
            }

            // Return the new value
            return xl_new;
        }

        VectorXd run(std::function<var(const Ref<const VectorXvar>&)> func,
                     const Ref<const VectorXvar>& xl_init, double tol,
                     bool check_hessian_posdef, bool verbose)
        {
            /*
             *
             */
            // Print the input vector and value of the objective function
            if (verbose)
            {
                std::cout << "Initial vector: x = " << xl_init.head(this->D).transpose()
                          << "; " << "f(x) = " << func(xl_init.head(this->D)) << std::endl; 
            }

            VectorXvar xl(xl_init);
            unsigned i = 0;
            double delta = 2 * tol;
            while (i < this->max_iter && delta > tol)
            {
                VectorXvar xl_new = this->step(func, xl, check_hessian_posdef, i, verbose);
                delta = (xl - xl_new).cast<double>().norm();
                i++;
                xl = xl_new;
            }
            return xl.cast<double>();
        }
};

#endif 
