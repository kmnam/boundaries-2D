#ifndef BOUNDARY_FINDER_HPP
#define BOUNDARY_FINDER_HPP

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Vector_2.h>
#include "boundaries.hpp"
#include "linearConstraints.hpp"
#include "SQP.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/11/2019
 */
using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Vector_2                                            Vector_2;

class BoundaryFinder 
{
    /*
     * A class that implements an importance sampling algorithm for
     * iteratively computing the boundary of a (bounded) 2-D region,
     * arising as the image of a map from a higher-dimensional space of 
     * input points. 
     */
    private:
        unsigned D;          // Number of parameters
        unsigned N;          // Number of points
        double area_tol;     // Terminate sampling when difference in area of 
                             // region bounded by alpha shapes in successive 
                             // iterations is less than this value
        unsigned max_iter;   // Maximum number of sampling iterations
        double curr_area;    // Area enclosed by last computed boundary

        // Matrix of sampled parameters
        MatrixXd params;

        // Matrix of points in 2-D space
        MatrixX2d points;

        // Set of linear constraints that parameters must satisfy
        LinearConstraints* constraints;

        // Vectors containing indices of the boundary points and edges 
        std::vector<unsigned> vertices;
        std::vector<std::pair<unsigned, unsigned> > edges;

        // boost::random::mt19937 random number generator
        boost::random::mt19937 rng;

    public:
        BoundaryFinder(unsigned D, double area_tol, unsigned max_iter,
                       unsigned seed, const Ref<const MatrixXd>& A,
                       const Ref<const VectorXd>& b)
        {
            /*
             * Straightforward constructor.
             */
            this->D = D;
            this->N = 0;
            this->area_tol = area_tol;
            this->max_iter = max_iter;
            this->curr_area = 0.0;
            this->rng.seed(seed);
            this->constraints = new LinearConstraints(A, b);
        }

        ~BoundaryFinder()
        {
            /*
             * Empty destructor. 
             */
            delete this->constraints;
        }

        void setConstraints(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Instantiate and store a new LinearConstraints instance from
             * the given matrix and vector. 
             */
            this->constraints->setAb(A, b);
        }

        void initialize(std::function<VectorXvar(const Ref<const VectorXvar>&)> func,
                        const Ref<const MatrixXd>& params)
        {
            /*
             * Initialize the sampling run by evaluating the given function
             * for a specified number of random parameter values in the
             * given logarithmic range.
             */
            // Run through the specified parameter values
            for (unsigned i = 0; i < params.rows(); ++i)
            {
                // Evaluate the given function at a randomly generated 
                // parameter point (if it satisfies the required constraints)
                if (this->constraints->check(params.row(i).transpose()))
                {
                    VectorXvar y = func(params.row(i).cast<var>());
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = params.row(i);
                    this->points.row(this->N-1) = y.cast<double>();
                }
            }
            if (this->N == 0)
                throw std::invalid_argument("No valid parameter values given");
        }

        bool step(std::function<VectorXvar(const Ref<const VectorXvar>&)> func, 
                  std::function<VectorXvar(const Ref<const VectorXvar>&, boost::random::mt19937&, LinearConstraints*)> mutate,
                  unsigned iter, bool simplify, bool verbose, std::string write_prefix = "")
        {
            /*
             * Given a list of points (with their x- and y-coordinates
             * specified in vectors of the same length), take one step
             * in the boundary-sampling algorithm as follows: 
             *
             * 1) Get the boundary of the position/steepness points 
             *    accrued thus far. 
             * 2) "Mutate" (randomly perturb) the input points in the
             *    determined boundary by uniformly sampling along each
             *    dimension within [x-radius, x+radius] in logarithmic
             *    coordinates.
             * 3) Plug in the mutated parameter values and obtain new 
             *    values for position/steepness.
             *
             * The return value indicates whether or not the enclosed 
             * area has converged to within the specified area tolerance. 
             */
            // Convert each position/steepness value to type double
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get boundary of the points in position/steepness space
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, simplify);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "_pass" << iter << ".txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence
            double area = bound_data.area;
            double change = area - this->curr_area;
            this->curr_area = area;
            if (verbose)
            {
                std::cout << "Iteration " << iter
                          << "; enclosed area: " << area
                          << "; change: " << change << std::endl;
            }
            if (change >= 0.0 && change < this->area_tol) return true;
            
            // For each of the points in the boundary, mutate the corresponding
            // model parameters once, and evaluate the given function at these
            // mutated parameter values
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Evaluate the given function at a randomly generated 
                // parameter point
                VectorXvar p = this->params.row(this->vertices[i]).cast<var>();
                VectorXvar q = mutate(p, this->rng, this->constraints);
                VectorXvar z = func(q);
                
                // Check that the mutation did not give rise to an already 
                // computed point 
                double mindist = (this->points.rowwise() - z.cast<double>().transpose()).rowwise().squaredNorm().minCoeff();
                if (mindist > 0)
                {
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = q.cast<double>();
                    this->points.row(this->N-1) = z.cast<double>();
                }
            }
            return false;
        }

        bool pull(std::function<std::pair<double, double>(std::vector<double>)> func,
                  double delta, unsigned iter, bool simplify, bool verbose,
                  std::string write_prefix = "")
        {
            /*
             *
             */
            // Store point coordinates in two vectors
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get boundary of the points
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, simplify);

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
            this->vertices = bound_data.vertices;
            this->edges = bound_data.edges;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "_pass" << iter << ".txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence
            double area = bound_data.area;
            double change = area - this->curr_area;
            this->curr_area = area;
            if (verbose)
            {
                std::cout << "Iteration " << iter
                          << "; enclosed area: " << area
                          << "; change: " << change << std::endl;
            }
            if (change >= 0.0 && change < this->area_tol) return true;

            // Obtain the outward vertex normals along the boundary and,
            // for each vertex in the boundary, "pull" along its outward
            // normal by distance delta
            std::vector<double> x_pulled, y_pulled;
            std::vector<Vector_2> normals = bound_data.outwardVertexNormals();
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                Vector_2 v(x[this->vertices[i]], y[this->vertices[i]]);
                Vector_2 pulled = v + delta * normals[i];
                x_pulled.push_back(pulled.x());
                y_pulled.push_back(pulled.y());
            }

            // Pull out the constraint matrix and vector 
            MatrixXd A = this->constraints->getA();
            VectorXd b = this->constraints->getb();
            unsigned nc = A.rows();

            // Define an SQPOptimizer instance to be utilized 
            SQPOptimizer* optimizer = new SQPOptimizer(this->D, nc, max_iter, A, b);

            // For each vertex in the boundary, minimize the distance to the
            // pulled vertex with a feasible parameter point
            /*
            this->params.conservativeResize(this->N + this->vertices.size(), this->D);
            this->points.conservativeResize(this->N + this->vertices.size(), 2);
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Extract the i-th parameter point
                std::vector<double> p;
                p.resize(this->D);
                VectorXd::Map(&p[0], this->D) = this->params.row(this->vertices[i]);

                // Minimize the appropriate objective function
                //VectorXd q = optimizer->run(
                //    [](){
                //    },
                //    this->params.row(this->vertices[i]).cast<var>(),
                //    tol, check_hessian_posdef, verbose
                //);
                //std::pair<double, double> z = func(q);
                //for (unsigned j = 0; j < q.size(); ++j)
                //    this->params(this->N + i, j) = q[j];
                //this->points(this->N + i, 0) = z.first;
                //this->points(this->N + i, 1) = z.second;
            }
            this->N += this->vertices.size();
            */
            return false;
        }

        void run(std::function<VectorXvar(const Ref<const VectorXvar>&)> func,
                 std::function<VectorXvar(const Ref<const VectorXvar>&, boost::random::mt19937&, LinearConstraints*)> mutate,
                 const Ref<const MatrixXd>& params, bool simplify, bool verbose,
                 std::string write_prefix = "")
        {
            /*
             * Run the boundary sampling until convergence, up to the maximum
             * number of iterations. 
             */
            // Initialize the sampling run ...
            this->initialize(func, params);

            // ... then take up to the maximum number of iterations 
            unsigned i = 0;
            bool terminate = false;
            while (i < this->max_iter && !terminate)
            {
                terminate = this->step(func, mutate, i, simplify, verbose, write_prefix);
                i++;
            }

            // Did the loop terminate without achieving convergence?
            if (!terminate)
            {
                std::cout << "Reached maximum number of iterations (" << max_iter
                          << ") without convergence" << std::endl;
            }
            else
            {
                std::cout << "Reached convergence within " << i << " iterations"
                          << std::endl;
            }
        }
};

#endif
