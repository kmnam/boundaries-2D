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
            VectorXvar x = xl_init.head(this->D);
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

#endif 
