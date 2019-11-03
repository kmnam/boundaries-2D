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
#include "boundaries.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/2/2019
 */
using namespace Eigen;

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

        // Vectors containing indices of the boundary points 
        std::vector<unsigned> vertices;

        // boost::random::mt19937 random number generator
        boost::random::mt19937 rng;

    public:
        BoundaryFinder(unsigned D, double area_tol, unsigned max_iter,
                       unsigned seed)
        {
            /*
             * Straightforward constructor.
             */
            this->D = D;
            this->area_tol = area_tol;
            this->max_iter = max_iter;
            this->curr_area = 0.0;
            this->rng.seed(seed);
        }

        ~BoundaryFinder()
        {
            /*
             * Empty destructor. 
             */
        }

        void initialize(std::function<std::pair<double, double>(std::vector<double>)> func,
                        const Ref<const MatrixXd>& params)
        {
            /*
             * Initialize the sampling run by evaluating the given function
             * for a specified number of random parameter values in the
             * given logarithmic range.
             */
            this->N = params.rows();
            this->params = params;
            this->points = MatrixX2d::Zero(this->N, 2);

            // Generate the sampled points
            for (unsigned i = 0; i < this->N; ++i)
            {
                // Evaluate the given function at a randomly generated 
                // parameter point
                std::vector<double> row;
                row.resize(this->D);
                VectorXd::Map(&row[0], this->D) = params.row(i);
                std::pair<double, double> y = func(row);
                this->points(i,0) = y.first;
                this->points(i,1) = y.second;
            }
        }

        bool step(std::function<std::pair<double, double>(std::vector<double>)> func,
                  std::function<std::vector<double>(std::vector<double>, boost::random::mt19937&)> mutate,
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
                bound_data.write(x, y, ss.str());
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
            if (change > 0.0 && change < this->area_tol) return true;
            
            // For each of the points in the boundary, mutate the corresponding
            // model parameters once, and evaluate the given function at these
            // mutated parameter values
            this->params.conservativeResize(this->N + this->vertices.size(), this->D);
            this->points.conservativeResize(this->N + this->vertices.size(), 2);
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Evaluate the given function at a randomly generated 
                // parameter point
                std::vector<double> p;
                p.resize(this->D);
                VectorXd::Map(&p[0], this->D) = this->params.row(this->vertices[i]);
                std::vector<double> q = mutate(p, this->rng);
                std::pair<double, double> z = func(q);
                for (unsigned j = 0; j < q.size(); ++j)
                    this->params(this->N + i, j) = q[j];
                this->points(this->N + i, 0) = z.first;
                this->points(this->N + i, 1) = z.second;
            }
            this->N += this->vertices.size();
            return false;
        }

        void run(std::function<std::pair<double, double>(std::vector<double>)> func,
                 std::function<std::vector<double>(std::vector<double>, boost::random::mt19937&)> mutate,
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
