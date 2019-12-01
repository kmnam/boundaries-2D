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
#include <random>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Vector_2.h>
#include "boundaries.hpp"
#include "linearConstraints.hpp"
#include "SQP.hpp"

/*
 * An implementation of a "boundary-finding" algorithm in the plane. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/30/2019
 */
using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Vector_2                                            Vector_2;

template <typename DT>
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

        // std::mt19937 random number generator
        std::mt19937 rng;

    public:
        BoundaryFinder(unsigned D, double area_tol, std::mt19937& rng,
                       const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Straightforward constructor.
             */
            this->D = D;
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
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

        MatrixXd getParams()
        {
            /*
             * Get the stored matrix of parameter values.
             */
            return this->params; 
        }

        void initialize(std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&)> func,
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
                    Matrix<DT, Dynamic, 1> y = func(params.row(i).cast<DT>());
                    
                    // Check that the point in the output space is not too 
                    // close to the others
                    if (this->points.rows() == 0 || (this->points.rowwise() - y.template cast<double>().transpose()).rowwise().squaredNorm().minCoeff() > 1e-20)
                    {
                        this->N++;
                        this->params.conservativeResize(this->N, this->D);
                        this->points.conservativeResize(this->N, 2);
                        this->params.row(this->N-1) = params.row(i);
                        this->points.row(this->N-1) = y.template cast<double>();
                    }
                }
            }
            if (this->N == 0)
                throw std::invalid_argument("No valid parameter values given");
        }

        bool step(std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&)> func, 
                  std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&, std::mt19937&)> mutate,
                  std::function<bool(const Ref<const Matrix<DT, Dynamic, 1> >&)> filter,
                  const unsigned iter, const unsigned max_edges, const bool verbose,
                  const std::string write_prefix = "")
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
            AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, max_edges);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence
            double area = bound_data.area;
            double change = area - this->curr_area;
            this->curr_area = area;
            if (verbose)
            {
                std::cout << "[STEP] Iteration " << iter
                          << "; enclosed area: " << area
                          << "; " << this->vertices.size() << " boundary points" 
                          << "; change: " << change << std::endl;
            }
            if (change >= 0.0 && change < this->area_tol)
                return true;
            
            // For each of the points in the boundary, mutate the corresponding
            // model parameters once, and evaluate the given function at these
            // mutated parameter values
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                bool filtered = true;
                double mindist = 2e-20;
                Matrix<DT, Dynamic, 1> q, z;
                while (filtered || mindist < 1e-20)
                {
                    // Evaluate the given function at a randomly generated 
                    // parameter point
                    Matrix<DT, Dynamic, 1> p = this->params.row(this->vertices[i]).template cast<DT>();
                    q = this->constraints->nearestL2(mutate(p, this->rng).template cast<double>()).template cast<DT>();
                    z = func(q);
                    filtered = filter(z);
                    
                    // Check that the mutation did not give rise to an already 
                    // computed point 
                    mindist = (this->points.rowwise() - z.template cast<double>().transpose()).rowwise().squaredNorm().minCoeff();
                }
                this->N++;
                this->params.conservativeResize(this->N, this->D);
                this->points.conservativeResize(this->N, 2);
                this->params.row(this->N-1) = q.template cast<double>();
                this->points.row(this->N-1) = z.template cast<double>();
            }
            return false;
        }

        bool pull(std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&)> func,
                  std::function<bool(const Ref<const Matrix<DT, Dynamic, 1> >&)> filter,
                  const double delta, const unsigned max_iter, const double sqp_tol,
                  const unsigned iter, const unsigned max_edges, const bool verbose,
                  const bool sqp_verbose, const std::string write_prefix = "")
        {
            /*
             * "Pull" the boundary points along their outward normal vectors
             * with sequential quadratic programming. 
             */
            using std::abs;

            // Store point coordinates in two vectors
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get boundary of the points
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, max_edges);

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
            this->vertices = bound_data.vertices;
            this->edges = bound_data.edges;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence
            double area = bound_data.area;
            double change = area - this->curr_area;
            this->curr_area = area;
            if (verbose)
            {
                std::cout << "[PULL] Iteration " << iter
                          << "; enclosed area: " << area
                          << "; " << this->vertices.size() << " boundary points" 
                          << "; change: " << change << std::endl;
            }
            if (change >= 0.0 && change < this->area_tol)
                return true;

            // Obtain the outward vertex normals along the boundary and,
            // for each vertex in the boundary, "pull" along its outward
            // normal by distance delta
            MatrixXd pulled(this->vertices.size(), 2);
            std::vector<Vector_2> normals = bound_data.outwardVertexNormals();
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                Vector_2 v(x[this->vertices[i]], y[this->vertices[i]]);
                Vector_2 v_pulled = v + delta * normals[i];
                pulled(i, 0) = CGAL::to_double(v_pulled.x());
                pulled(i, 1) = CGAL::to_double(v_pulled.y());

                // Check that the pulled point is not subject to filtering
                if (filter(pulled.row(i)))
                {
                    // If so, simply don't pull that vertex
                    pulled(i, 0) = x[this->vertices[i]];
                    pulled(i, 1) = y[this->vertices[i]];
                }
            }

            // Pull out the constraint matrix and vector 
            MatrixXd A = this->constraints->getA();
            VectorXd b = this->constraints->getb();
            unsigned nc = A.rows();

            // Define an SQPOptimizer instance to be utilized 
            SQPOptimizer<DT>* optimizer = new SQPOptimizer<DT>(this->D, nc, A, b);

            // For each vertex in the boundary, minimize the distance to the
            // pulled vertex with a feasible parameter point
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Minimize the appropriate objective function
                Matrix<DT, Dynamic, 1> target = pulled.row(i).cast<DT>();
                auto obj = [func, target](const Ref<const Matrix<DT, Dynamic, 1> >& x)
                {
                    return (target - func(x)).squaredNorm();
                };
                VectorXd x_init = this->params.row(this->vertices[i]);
                VectorXd l_init = VectorXd::Ones(nc) - this->constraints->active(x_init).template cast<double>();
                Matrix<DT, Dynamic, 1> xl_init(this->D + nc);
                xl_init << x_init.cast<DT>(), l_init.cast<DT>();
                VectorXd q = optimizer->run(obj, xl_init, max_iter, sqp_tol, BFGS, sqp_verbose);
                Matrix<DT, Dynamic, 1> z = func(q.cast<DT>());
                
                // Check that the mutation did not give rise to an already 
                // computed point
                double mindist = (this->points.rowwise() - z.template cast<double>().transpose()).rowwise().squaredNorm().minCoeff();
                if (mindist > 0)
                {
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = q;
                    this->points.row(this->N-1) = z.template cast<double>();
                }
            }
            return false;
        }

        void run(std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&)> func,
                 std::function<Matrix<DT, Dynamic, 1>(const Ref<const Matrix<DT, Dynamic, 1> >&, std::mt19937&)> mutate,
                 std::function<bool(const Ref<const Matrix<DT, Dynamic, 1> >&)> filter,
                 const Ref<const MatrixXd>& params, const unsigned min_step_iter,
                 const unsigned max_step_iter, const unsigned min_pull_iter,
                 const unsigned max_pull_iter, const unsigned max_edges, const bool verbose,
                 const unsigned sqp_max_iter, const double sqp_tol,
                 const bool sqp_verbose, const std::string write_prefix = "")
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
            unsigned converged = 0;
            while (i < min_step_iter || (i < max_step_iter && !terminate))
            {
                bool conv = this->step(func, mutate, filter, i, max_edges, verbose, write_prefix);
                if (!conv) converged = 0;
                else       converged += 1;
                terminate = (converged >= 10);
                i++;
            }

            // Pull the boundary points outward
            unsigned j = 0;
            converged = 0;
            while (j < min_step_iter || (j < max_pull_iter && !terminate))
            {
                bool conv = this->pull(
                    func, filter, 0.1 * std::sqrt(this->curr_area), sqp_max_iter, sqp_tol,
                    i + j, max_edges, verbose, sqp_verbose, write_prefix
                );
                if (!conv) converged = 0;
                else       converged += 1;
                terminate = (converged >= 3);
                j++;
            }

            // Compute the boundary one last time if the algorithm did not terminate
            if (!terminate)
            {
                std::vector<double> x, y;
                x.resize(this->N);
                y.resize(this->N);
                VectorXd::Map(&x[0], this->N) = this->points.col(0);
                VectorXd::Map(&y[0], this->N) = this->points.col(1);
                Boundary2D boundary(x, y);
                AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, max_edges);
                this->vertices = bound_data.vertices;

                // Write boundary information to file if desired
                if (write_prefix.compare(""))
                {
                    std::stringstream ss;
                    ss << write_prefix << "-pass" << i + j << ".txt";
                    bound_data.write(ss.str());
                }

                // Compute enclosed area and test for convergence
                double area = bound_data.area;
                double change = area - this->curr_area;
                this->curr_area = area;
                if (verbose)
                {
                    std::cout << "[FINAL] Iteration " << i + j
                              << "; enclosed area: " << area
                              << "; " << this->vertices.size() << " boundary points" 
                              << "; change: " << change << std::endl;
                }
                if (change >= 0.0 && change < this->area_tol)
                    terminate = true;
            }

            // Did the loop terminate without achieving convergence?
            if (!terminate)
            {
                std::cout << "Reached maximum number of iterations ("
                          << max_step_iter + max_pull_iter
                          << ") without convergence" << std::endl;
            }
            else
            {
                std::cout << "Reached convergence within " << i + j << " iterations"
                          << std::endl;
            }
        }
};

#endif
