#ifndef LINEAR_CONSTRAINTS_HPP
#define LINEAR_CONSTRAINTS_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <CGAL/Gmpzf.h>

/*
 * Helper class for representing linear constraints of the form A * x >= b.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/17/2021
 */
using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

class LinearConstraints
{
    /*
     * A class that implements a set of linear constraints among a set
     * of variables, A * x >= b.
     */
    private:
        unsigned D;    // Number of variables
        unsigned N;    // Number of constraints
        MatrixXd A;    // Matrix of constraint coefficients
        VectorXd b;    // Matrix of constraint values

        // Quadratic program for nearest point queries
        Program* nearest_L2;

    public:
        LinearConstraints()
        {
            /*
             * Empty constructor.
             */
            this->N = 0;
            this->D = 0;
            this->A = MatrixXd::Zero(0, 0);
            this->b = VectorXd::Zero(0);
            this->nearest_L2 = new Program(CGAL::LARGER, false, 0.0, false, 0.0);
        }

        LinearConstraints(unsigned D, double lower, double upper)
        {
            /*
             * Constructor which sets each variable to between the given 
             * lower and upper bounds.
             */
            this->N = 2 * D;
            this->D = D;
            this->A.resize(this->N, this->D);
            this->b.resize(this->N);
            for (unsigned i = 0; i < this->D; ++i)
            {
                VectorXd v = VectorXd::Zero(this->D);
                v(i) = 1.0;
                this->A.row(i) = v.transpose();
                this->A.row(this->D + i) = -v.transpose();
                this->b(i) = lower;
                this->b(this->D + i) = -upper;
            }
            this->nearest_L2 = new Program(CGAL::LARGER, false, 0.0, false, 0.0);

            // Update quadratic program
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2->set_a(j, i, this->A(i,j));
                this->nearest_L2->set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2->set_d(i, i, 2.0);
                this->nearest_L2->set_c0(0.0);
                this->nearest_L2->set_c(i, 0.0);
            }
        } 

        LinearConstraints(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Constructor with supplied matrix and vector.
             */
            this->A = A;
            this->b = b;
            this->N = this->A.rows();
            this->D = this->A.cols();
            if (this->b.size() != this->N)
            {
                std::stringstream ss;
                ss << "Dimensions of A and b do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << this->b.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            this->nearest_L2 = new Program(CGAL::LARGER, false, 0.0, false, 0.0);

            // Update quadratic program
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2->set_a(j, i, this->A(i,j));
                this->nearest_L2->set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2->set_d(i, i, 2.0);
                this->nearest_L2->set_c0(0.0);
                this->nearest_L2->set_c(i, 0.0);
            }
        }

        ~LinearConstraints()
        {
            /*
             * Trivial destructor.
             */
            delete this->nearest_L2;
        }

        void parse(std::string polytope_file)
        {
            /*
             * Given a file specifying a convex polytope in terms of 
             * half-spaces (inequalities), read in the constraint matrix
             * and vector. 
             */
            unsigned D = 0;
            unsigned N = 0;
            MatrixXd A(N, D);
            VectorXd b(N);
            
            std::string line;
            std::ifstream infile(polytope_file);
            if (infile.is_open())
            {
                while (std::getline(infile, line))
                {
                    // Accumulate the entries in each line ...
                    std::stringstream ss(line);
                    std::string token;
                    std::vector<double> row;
                    N++;
                    while (std::getline(ss, token, ' '))
                        row.push_back(std::stod(token));

                    // If this is the first row being parsed, get the number 
                    // of columns in constraint matrix 
                    if (D == 0) D = row.size() - 1;

                    // Add the new constraint, with column 0 specifying the 
                    // constant term and the remaining columns specifying the
                    // linear coefficients:
                    //
                    // a0 + a1*x1 + a2*x2 + ... + aN*xN >= 0
                    //
                    A.conservativeResize(N, D);
                    b.conservativeResize(N);
                    for (unsigned i = 1; i < row.size(); ++i)
                        A(N-1, i-1) = row[i];
                    b(N-1) = -row[0];
                }
                infile.close();
            }
            else
                throw std::invalid_argument("Specified file does not exist");

            // Update this->A, this->b, this->N, this->D 
            this->A = A;
            this->b = b;
            this->N = N;
            this->D = D;

            // Update quadratic program
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2->set_a(j, i, this->A(i,j));
                this->nearest_L2->set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2->set_d(i, i, 2.0);
                this->nearest_L2->set_c0(0.0);
                this->nearest_L2->set_c(i, 0.0);
            }
        }

        void setAb(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Update this->A and this->b.
             */
            this->A = A;
            this->b = b;
            this->N = A.rows();
            this->D = A.cols();
            if (this->b.size() != this->N)
            {
                std::stringstream ss;
                ss << "Dimensions of A and b do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << this->b.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }

            // Update quadratic program
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2->set_a(j, i, this->A(i,j));
                this->nearest_L2->set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2->set_d(i, i, 2.0);
                this->nearest_L2->set_c0(0.0);
                this->nearest_L2->set_c(i, 0.0);
            }
        }

        MatrixXd getA()
        {
            /*
             * Return this->A.
             */
            return this->A;
        }

        VectorXd getb()
        {
            /*
             * Return this->b.
             */
            return this->b;
        }

        bool check(const Ref<const VectorXd>& x)
        {
            /* 
             * Return true if the constraints were satisfied or, otherwise,
             * if this->A * x >= this->b.
             */
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() >= (this->b).array()).all();
        }

        Matrix<bool, Dynamic, 1> active(const Ref<const VectorXd>& x)
        {
            /*
             * Return a boolean vector indicating which constraints 
             * are active (i.e., which constraints are satisfied as 
             * equalities).
             */
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() == (this->b).array());
        }

        VectorXd nearestL2(const Ref<const VectorXd>& x)
        {
            /*
             * Return the nearest point to x, with respect to L2 distance,
             * that satisfies the given constraints.
             */
            // First check that x itself satisfies the constraints
            if (this->check(x)) return x;

            // Otherwise, solve the quadratic program for the nearest point to x
            for (unsigned i = 0; i < this->D; ++i)
                this->nearest_L2->set_c(i, -2.0 * x(i));
            Solution solution = CGAL::solve_quadratic_program(*this->nearest_L2, ET());
            if (solution.is_infeasible())
                throw std::runtime_error("Quadratic program is infeasible");
            else if (solution.is_unbounded())
                throw std::runtime_error("Quadratic program is unbounded");
            else if (!solution.is_optimal())
                throw std::runtime_error("Failed to compute optimal solution");

            // Collect the values of the solution into a VectorXd
            VectorXd y = VectorXd::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = CGAL::to_double(*it);
                i++;
            }
            return y;
        }
};

#endif
