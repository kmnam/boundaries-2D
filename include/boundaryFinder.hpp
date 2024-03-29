#ifndef BOUNDARY_FINDER_HPP
#define BOUNDARY_FINDER_HPP
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Vector_2.h>
#include <boost/random.hpp>
#include "boundaries.hpp"
#include "linearConstraints.hpp"
#include "SQP.hpp"
#include "simplex.hpp"
#include "triangulate.hpp"
#include "sample.hpp"

/*
 * An implementation of a "boundary-finding" algorithm in the plane. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/13/2021
 */
using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Vector_2                                            Vector_2;

class ParamFileCollection
{
    /*
     * A class that dumps and reads parameter values to and from a 
     * collection of files.
     */
    public:
        std::string prefix;                // Prefix for filenames
        std::vector<std::string> paths;    // Vector of filenames
        unsigned nfiles;                   // Number of files (length of paths)

        ParamFileCollection(std::string prefix)
        {
            /*
             * Trivial constructor.
             */
            this->prefix = prefix;
        }

        ~ParamFileCollection()
        {
            /*
             * Empty destructor.
             */
        }

        void dump(const Ref<const MatrixXd>& params)
        {
            /*
             * Dump parameter values to a new tab-delimited file.
             */
            std::stringstream ss;
            ss << this->prefix << "-" << this->nfiles << ".tsv";
            std::string path = ss.str();
            std::ofstream outfile(path);
            if (outfile.is_open())
            {
                outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
                for (unsigned i = 0; i < params.rows(); ++i)
                {
                    for (unsigned j = 0; j < params.cols() - 1; ++j)
                    {
                        outfile << params(i,j) << "\t";
                    }
                    outfile << params(i,params.cols()-1) << "\n";
                }
            }
            outfile.close();
            this->nfiles++;
            this->paths.push_back(path);
        }

        MatrixXd read(unsigned i)
        {
            /*
             * Read the i-th file in the collection.
             */
            if (i >= this->nfiles)
            {
                std::stringstream ss;
                ss << "File does not exist: " << this->prefix << "-" << i << ".tsv";
                throw std::runtime_error(ss.str());
            }

            MatrixXd params;
            std::stringstream ss;
            ss << this->prefix << "-" << i << ".tsv";
            std::string path = ss.str();
            std::ifstream infile(path);
            if (infile.is_open())
            {
                unsigned nrows = 0, ncols = 0;
                std::string line;
                while (std::getline(infile, line))
                {
                    std::istringstream ssl(line);
                    std::string token;
                    std::vector<double> row;
                    while (std::getline(ssl, token, '\t'))
                    {
                        row.push_back(std::stod(token));
                    }
                    if (nrows == 0) ncols = row.size();
                    nrows++;
                    params.conservativeResize(nrows, ncols);
                    for (unsigned i = 0; i < row.size(); ++i)
                    {
                        params(nrows-1, i) = row[i];
                    }
                }
            }
            return params;
        }

        void writeToFile(std::string filename)
        {
            /*
             * Write the parameters in the entire collection to a single
             * file at the given path. 
             */
            if (this->nfiles == 0)
            {
                throw std::runtime_error("Parameter file collection is empty");
            }

            // Start with the 0-th file ...
            MatrixXd params = this->read(0);
            std::ofstream outfile(filename);
            if (outfile.is_open())
            {
                outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
                for (unsigned i = 0; i < params.rows(); ++i)
                {
                    for (unsigned j = 0; j < params.cols() - 1; ++j)
                    {
                        outfile << params(i,j) << "\t";
                    }
                    outfile << params(i,params.cols()-1) << "\n";
                }
            }
            outfile.close();

            // ... then run through the remaining files
            for (unsigned k = 1; k < this->nfiles; ++k)
            {
                params = this->read(k);
                std::ofstream outfile2(filename, std::ios_base::app);
                if (outfile2.is_open())
                {
                    outfile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
                    for (unsigned i = 0; i < params.rows(); ++i)
                    {
                        for (unsigned j = 0; j < params.cols() - 1; ++j)
                        {
                            outfile2 << params(i,j) << "\t";
                        }
                        outfile2 << params(i,params.cols()-1) << "\n";
                    }
                }
                outfile2.close();
            }
        }
};

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

        // Set of simplices in the Delaunay triangulation of the 
        // convex polytope
        std::vector<Simplex> interior_simplices;

        // Set of simplices in the boundary of the Delaunay triangulation
        std::vector<Simplex> boundary_simplices; 

        // Vector containing indices of the boundary points  
        std::vector<unsigned> vertices;

        // boost::random::mt19937 random number generator
        boost::random::mt19937 rng;

    public:
        BoundaryFinder(double area_tol, boost::random::mt19937& rng, 
                       const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
        {
            /*
             * Constructor with input polytope constraints as Eigen::Matrix
             * objects.
             */
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new LinearConstraints(A, b);
            this->D = A.cols();
        }

        BoundaryFinder(double area_tol, boost::random::mt19937& rng, 
                       std::string constraints_file)
        {
            /*
             * Constructor with input polytope constraints to be parsed from
             * a text file. 
             */
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new LinearConstraints();
            this->constraints->parse(constraints_file);
            this->D = this->constraints->getA().cols();
        }

        BoundaryFinder(double area_tol, boost::random::mt19937& rng, 
                       std::string constraints_file, std::string vertices_file)
        {
            /*
             * Constructor with input polytope constraints and vertices to 
             * be parsed from text files.
             *
             * The enumerated vertices are used to triangulate the polytope.
             */
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new LinearConstraints();
            this->constraints->parse(constraints_file);

            // Parse the vertices from the given .vert file
            std::vector<std::vector<double> > vertices = parseVertices(vertices_file);

            // Triangulate the parsed polytope
            std::pair<Delaunay_triangulation, unsigned> result = triangulate(vertices);
            Delaunay_triangulation dt = result.first;
            this->D = result.second;

            // Get the simplices of the triangulation
            this->interior_simplices = getInteriorSimplices(dt, this->D);

            // Get the facets of the convex hull
            this->boundary_simplices = getConvexHullFacets(dt, this->D);
        }

        BoundaryFinder(double area_tol, boost::random::mt19937& rng, 
                       std::string constraints_file, std::string vertices_file,
                       std::string triangulation_file)
        {
            /*
             * Constructor with input polytope constraints, vertices, and 
             * pre-computed Delaunay triangulation all parsed from input 
             * text files.
             */
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new LinearConstraints();
            this->constraints->parse(constraints_file);
            this->D = this->constraints->getA().cols();

            // Parse the vertices and triangulation from the given input files
            std::vector<std::vector<double> > vertices = parseVertices(vertices_file);
            auto triangulation = parseTriangulation(triangulation_file);
            std::vector<Simplex> simplices = std::get<1>(triangulation);

            // Classify the simplices in the triangulation as either interior or boundary
            auto classified = classifySimplices(simplices, *this->constraints, this->D, rng);
            this->interior_simplices = classified.first; 
            this->boundary_simplices = classified.second;
        }

        ~BoundaryFinder()
        {
            /*
             * Trivial destructor. 
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

        void initialize(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                        std::function<bool(const Ref<const VectorXd>&)> filter,
                        const Ref<const MatrixXd>& params)
        {
            /*
             * Initialize the sampling run by evaluating the given function
             * at a given (pre-sampled) set of parameter values. 
             */
            this->N = 0;
            this->params.resize(this->N, this->D);
            this->points.resize(this->N, 2);
                
            // Evaluate the given function at a randomly generated 
            // parameter point (if it satisfies the required constraints)
            for (unsigned i = 0; i < params.rows(); ++i)
            {
                VectorXd y = func(params.row(i));
                
                // Check that the point in the output space is not too 
                // close to the others
                if (!filter(y) && (this->points.rows() == 0 || (this->points.rowwise() - y.transpose()).rowwise().norm().minCoeff() > 1e-10))
                {
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = params.row(i);
                    this->points.row(this->N-1) = y;
                }
            }
        }

        void initialize(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                        std::function<bool(const Ref<const VectorXd>&)> filter,
                        unsigned n_within, unsigned n_bound)
        {
            /*
             * Initialize the sampling run by:
             *
             *   1) sampling the given number of points from the parameter polytope
             *      via Delaunay triangulation; and
             *   2) evaluating the given function and the sampled parameter values. 
             *
             * n_within is the number of points to be sampled from the 
             * interior simplices of the polytope; n_bound is the number of
             * points to be sampled from the boundary simplices of the
             * polytope. 
             */
            this->N = 0;
            this->params.resize(this->N, this->D);
            this->points.resize(this->N, 2);

            // Check that interior/boundary simplices have been labeled as 
            // such if nonzero numbers of points are to be sampled from either 
            if (n_within > 0 && this->interior_simplices.size() == 0)
                throw std::invalid_argument("Interior simplices were not specified, cannot sample from interior"); 
            else if (n_bound > 0 && this->boundary_simplices.size() == 0)
                throw std::invalid_argument("Boundary simplices were not specified, cannot sample from boundary"); 

            // Sample until the given number of points have been sampled
            // from the interior of the polytope  
            while (this->N < n_within)
            {
                MatrixXd params = sampleFromSimplices(this->interior_simplices, n_within - this->N, this->rng);

                // Evaluate the given function at a randomly generated 
                // parameter point (if it satisfies the required constraints)
                for (unsigned i = 0; i < params.rows(); ++i)
                {
                    VectorXd y = func(params.row(i));
                    
                    // Check that the point in the output space is not too 
                    // close to the others
                    if (!filter(y) && (this->points.rows() == 0 || (this->points.rowwise() - y.transpose()).rowwise().norm().minCoeff() > 1e-10))
                    {
                        this->N++;
                        this->params.conservativeResize(this->N, this->D);
                        this->points.conservativeResize(this->N, 2);
                        this->params.row(this->N-1) = params.row(i);
                        this->points.row(this->N-1) = y;
                    }
                }
            }

            // Sample until the given number of points have been sampled
            // from the boundary of the polytope  
            while (this->N < n_bound + n_within)
            {
                MatrixXd params = sampleFromSimplices(this->boundary_simplices, n_bound + n_within - this->N, this->rng);

                // Evaluate the given function at a randomly generated 
                // parameter point (if it satisfies the required constraints)
                for (unsigned i = 0; i < params.rows(); ++i)
                {
                    VectorXd y = func(params.row(i));
                    
                    // Check that the point in the output space is not too 
                    // close to the others
                    if (!filter(y) && (this->points.rows() == 0 || (this->points.rowwise() - y.transpose()).rowwise().norm().minCoeff() > 1e-10))
                    {
                        this->N++;
                        this->params.conservativeResize(this->N, this->D);
                        this->points.conservativeResize(this->N, 2);
                        this->params.row(this->N-1) = params.row(i);
                        this->points.row(this->N-1) = y;
                    }
                }
            }
        }

        void initializeByRandomWalk(std::function<VectorXd(const Ref<const VectorXd>&)> func, 
                                    std::function<bool(const Ref<const VectorXd>&)> filter,
                                    unsigned n_sample, unsigned nchains = 5, double atol = 1e-8,
                                    double warmup = 0.5, unsigned ntrials = 50)
        {
            /*
             * Initialize the sampling run by:
             *
             *   1) sampling the given number of points from the parameter polytope
             *      via MCMC-based sampling; and 
             *   2) evaluating the given function and the sampled parameter values. 
             *
             * n_within is the number of points to be sampled from the 
             * interior simplices of the polytope; n_bound is the number of
             * points to be sampled from the boundary simplices of the
             * polytope. 
             */
            this->N = 0;
            this->params.resize(this->N, this->D);
            this->points.resize(this->N, 2);

            // Sample until the given number of points have been sampled
            // from the polytope
            MatrixXd A = this->constraints->getA();
            VectorXd b = this->constraints->getb();
            MatrixXd concat(A.rows(), A.cols() + 1); 
            concat << -b, A; 
            Polytope polytope(concat);
            while (this->N < n_sample)
            {
                unsigned n_remain = n_sample - this->N; 
                MatrixXd params = sampleFromConvexPolytopeRandomWalk(polytope, n_remain, this->rng, nchains, atol, warmup, ntrials);

                // Evaluate the given function at a randomly generated 
                // parameter point (if it satisfies the required constraints)
                for (unsigned i = 0; i < params.rows(); ++i)
                {
                    VectorXd y = func(params.row(i));
                    
                    // Check that the point in the output space is not too 
                    // close to the others
                    if (!filter(y) && (this->points.rows() == 0 || (this->points.rowwise() - y.transpose()).rowwise().norm().minCoeff() > 1e-10))
                    {
                        this->N++;
                        this->params.conservativeResize(this->N, this->D);
                        this->points.conservativeResize(this->N, 2);
                        this->params.row(this->N-1) = params.row(i);
                        this->points.row(this->N-1) = y;
                    }
                }
            }
        }

        bool step(std::function<VectorXd(const Ref<const VectorXd>&)> func, 
                  std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate,
                  std::function<bool(const Ref<const VectorXd>&)> filter,
                  const unsigned iter, const unsigned max_edges, const bool verbose,
                  const std::string write_prefix = "")
        {
            /*
             * Given a list of points (with their x- and y-coordinates
             * specified in vectors of the same length), take one step
             * in the boundary-sampling algorithm as follows: 
             *
             * 1) Get the boundary of the output points accrued thus far. 
             * 2) "Mutate" (randomly perturb) the input points in the
             *    determined boundary by uniformly sampling along each
             *    dimension within [x-radius, x+radius] in logarithmic
             *    coordinates.
             * 3) Plug in the mutated parameter values and obtain new 
             *    output points.
             *
             * The return value indicates whether or not the enclosed 
             * area has converged to within the specified area tolerance. 
             */
            // Store the output coordinates in vectors 
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get boundary of the output points in 2-D space (assume that 
            // shape is simply connected)
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data;
            try
            {
                // This line may throw:
                // - CGAL::Precondition_exception (while attempting polygon instantiation)
                // - std::runtime_error (if polygon is not simple)
                bound_data = boundary.getBoundary<true>(true, true, max_edges);
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
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
                          << "; " << this->points.rows() << " total points" 
                          << "; change in area: " << change << std::endl;
            }
            
            // For each of the points in the boundary, mutate the corresponding
            // model parameters once, and evaluate the given function at these
            // mutated parameter values
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                bool filtered = true;
                double mindist = 0.0;
                unsigned j = 0;
                VectorXd q, z;
                while ((filtered || mindist < 1e-10) && j < 20)   // Attempt 20 mutations
                {
                    // Evaluate the given function at a randomly generated 
                    // parameter point
                    VectorXd p = this->params.row(this->vertices[i]);
                    q = this->constraints->nearestL2(mutate(p, this->rng));
                    z = func(q);
                    filtered = filter(z);
                    
                    // Check that the mutation did not give rise to an already 
                    // computed point 
                    mindist = (this->points.rowwise() - z.transpose()).rowwise().norm().minCoeff();

                    j++;
                }
                if (!filtered && mindist > 1e-10)
                {
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = q;
                    this->points.row(this->N-1) = z;
                }
            }

            return (std::abs(change) < this->area_tol * (area - change));
        }

        bool pull(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                  std::function<bool(const Ref<const VectorXd>&)> filter,
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

            // Get boundary of the points (assume that shape is simply connected)
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data;
            try
            {
                // This line may throw:
                // - CGAL::Precondition_exception (while attempting polygon instantiation)
                // - std::runtime_error (if polygon is not simple)
                bound_data = boundary.getBoundary<true>(true, true, max_edges);
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
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
                std::cout << "[PULL] Iteration " << iter
                          << "; enclosed area: " << area
                          << "; " << this->vertices.size() << " boundary points" 
                          << "; " << this->points.rows() << " total points" 
                          << "; change in area: " << change << std::endl;
            }

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
            SQPOptimizer<double>* optimizer = new SQPOptimizer<double>(this->D, nc, A, b);

            // For each vertex in the boundary, minimize the distance to the
            // pulled vertex with a feasible parameter point
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Minimize the appropriate objective function
                VectorXd target = pulled.row(i);
                auto obj = [func, target](const Ref<const VectorXd>& x)
                {
                    return (target - func(x)).squaredNorm();
                };
                VectorXd x_init = this->params.row(this->vertices[i]);
                VectorXd l_init = VectorXd::Ones(nc) - this->constraints->active(x_init).cast<double>();
                VectorXd xl_init(this->D + nc);
                xl_init << x_init, l_init;
                VectorXd q = optimizer->run(obj, xl_init, max_iter, sqp_tol, BFGS, sqp_verbose);
                VectorXd z = func(q);
                
                // Check that the mutation did not give rise to an already 
                // computed point
                double mindist = (this->points.rowwise() - z.transpose()).rowwise().norm().minCoeff();
                if (!filter(z) && mindist > 1e-10)
                {
                    this->N++;
                    this->params.conservativeResize(this->N, this->D);
                    this->points.conservativeResize(this->N, 2);
                    this->params.row(this->N-1) = q;
                    this->points.row(this->N-1) = z;
                }
            }

            delete optimizer;
            return (std::abs(change) < this->area_tol * (area - change));
        }

        void run(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                 std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate,
                 std::function<bool(const Ref<const VectorXd>&)> filter,
                 const Ref<const MatrixXd>& init_params,
                 const unsigned min_step_iter, const unsigned max_step_iter,
                 const unsigned min_pull_iter, const unsigned max_pull_iter,
                 const unsigned max_edges, const bool verbose,
                 const unsigned sqp_max_iter, const double sqp_tol,
                 const bool sqp_verbose, const std::string write_prefix = "")
        {
            /*
             * Run the boundary sampling until convergence, up to the maximum
             * number of iterations, given an initial set of parameter values.  
             */
            // Initialize the sampling run ...
            this->initialize(func, filter, init_params);

            // ... then take up to the maximum number of iterations 
            unsigned i = 0;
            bool terminate = false;
            unsigned n_converged = 0;
            while (i < min_step_iter || (i < max_step_iter && !terminate))
            {
                bool result = this->step(func, mutate, filter, i, max_edges, verbose, write_prefix);
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                i++;
            }

            // Pull the boundary points outward
            unsigned j = 0;
            terminate = false;
            n_converged = 0;
            while (j < min_pull_iter || (j < max_pull_iter && !terminate))
            {
                bool result = this->pull(
                    func, filter, 0.1 * std::sqrt(this->curr_area), sqp_max_iter, sqp_tol,
                    i + j, max_edges, verbose, sqp_verbose, write_prefix
                );
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                j++;
            }

            // Compute the boundary one last time
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data;
            try
            {
                // This line may throw:
                // - CGAL::Precondition_exception (while attempting polygon instantiation)
                // - std::runtime_error (if polygon is not simple)
                bound_data = boundary.getBoundary<true>(true, true, max_edges);
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "-final.txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence if algorithm
            // did not already terminate
            if (!terminate)
            {
                double area = bound_data.area;
                double change = area - this->curr_area;
                this->curr_area = area;
                if (verbose)
                {
                    std::cout << "[FINAL] Iteration " << i + j
                              << "; enclosed area: " << area
                              << "; " << this->vertices.size() << " boundary points"
                              << "; " << this->points.rows() << " total points" 
                              << "; change in area: " << change << std::endl;
                }
                if (std::abs(change) < this->area_tol)
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

        void run(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                 std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate,
                 std::function<bool(const Ref<const VectorXd>&)> filter,
                 const unsigned n_within, const unsigned n_bound,
                 const unsigned min_step_iter, const unsigned max_step_iter,
                 const unsigned min_pull_iter, const unsigned max_pull_iter,
                 const unsigned max_edges, const bool verbose,
                 const unsigned sqp_max_iter, const double sqp_tol,
                 const bool sqp_verbose, const std::string write_prefix = "")
        {
            /*
             * Run the boundary sampling until convergence, up to the maximum
             * number of iterations, with an initial set of parameter values 
             * sampled from a Delaunay triangulation of the constraint polytope. 
             */
            // Initialize the sampling run ...
            this->initialize(func, filter, n_within, n_bound);

            // ... then take up to the maximum number of iterations 
            unsigned i = 0;
            bool terminate = false;
            unsigned n_converged = 0;
            while (i < min_step_iter || (i < max_step_iter && !terminate))
            {
                bool result = this->step(func, mutate, filter, i, max_edges, verbose, write_prefix);
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                i++;
            }

            // Pull the boundary points outward
            unsigned j = 0;
            terminate = false;
            n_converged = 0;
            while (j < min_pull_iter || (j < max_pull_iter && !terminate))
            {
                bool result = this->pull(
                    func, filter, 0.1 * std::sqrt(this->curr_area), sqp_max_iter, sqp_tol,
                    i + j, max_edges, verbose, sqp_verbose, write_prefix
                );
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                j++;
            }

            // Compute the boundary one last time
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data;
            try
            {
                // This line may throw:
                // - CGAL::Precondition_exception (while attempting polygon instantiation)
                // - std::runtime_error (if polygon is not simple)
                bound_data = boundary.getBoundary<true>(true, true, max_edges);
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "-final.txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence if algorithm
            // did not already terminate
            if (!terminate)
            {
                double area = bound_data.area;
                double change = area - this->curr_area;
                this->curr_area = area;
                if (verbose)
                {
                    std::cout << "[FINAL] Iteration " << i + j
                              << "; enclosed area: " << area
                              << "; " << this->vertices.size() << " boundary points"
                              << "; " << this->points.rows() << " total points" 
                              << "; change in area: " << change << std::endl;
                }
                if (std::abs(change) < this->area_tol)
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

        void runFromRandomWalk(std::function<VectorXd(const Ref<const VectorXd>&)> func,
                               std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate,
                               std::function<bool(const Ref<const VectorXd>&)> filter,
                               const unsigned n_sample, const unsigned min_step_iter, const unsigned max_step_iter,
                               const unsigned min_pull_iter, const unsigned max_pull_iter, const unsigned max_edges,
                               const bool verbose, const unsigned sqp_max_iter, const double sqp_tol,
                               const bool sqp_verbose, const std::string write_prefix = "",
                               const unsigned nchains = 5, const double atol = 1e-8,
                               const double warmup = 0.5, const unsigned ntrials = 50)
        {
            /*
             * Run the boundary sampling until convergence, up to the maximum
             * number of iterations, with an initial set of parameter values 
             * sampled via MCMC sampling from the constraint polytope. 
             */
            // Initialize the sampling run ...
            this->initializeByRandomWalk(func, filter, n_sample, nchains, atol, warmup, ntrials);

            // ... then take up to the maximum number of iterations 
            unsigned i = 0;
            bool terminate = false;
            unsigned n_converged = 0;
            while (i < min_step_iter || (i < max_step_iter && !terminate))
            {
                bool result = this->step(func, mutate, filter, i, max_edges, verbose, write_prefix);
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                i++;
            }

            // Pull the boundary points outward
            unsigned j = 0;
            terminate = false;
            n_converged = 0;
            while (j < min_pull_iter || (j < max_pull_iter && !terminate))
            {
                bool result = this->pull(
                    func, filter, 0.1 * std::sqrt(this->curr_area), sqp_max_iter, sqp_tol,
                    i + j, max_edges, verbose, sqp_verbose, write_prefix
                );
                if (!result) n_converged = 0;
                else         n_converged += 1;
                terminate = (n_converged >= 5);
                j++;
            }

            // Compute the boundary one last time
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data;
            try
            {
                // This line may throw:
                // - CGAL::Precondition_exception (while attempting polygon instantiation)
                // - std::runtime_error (if polygon is not simple)
                bound_data = boundary.getBoundary<true>(true, true, max_edges);
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false 
                // This may throw a CGAL::Assertion_exception (while attempting alpha 
                // shape instantiation)
                try 
                {
                    bound_data = boundary.getBoundary<false>(true, true, max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            bound_data.orient(CGAL::RIGHT_TURN);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                std::stringstream ss;
                ss << write_prefix << "-final.txt";
                bound_data.write(ss.str());
            }

            // Compute enclosed area and test for convergence if algorithm
            // did not already terminate
            if (!terminate)
            {
                double area = bound_data.area;
                double change = area - this->curr_area;
                this->curr_area = area;
                if (verbose)
                {
                    std::cout << "[FINAL] Iteration " << i + j
                              << "; enclosed area: " << area
                              << "; " << this->vertices.size() << " boundary points"
                              << "; " << this->points.rows() << " total points" 
                              << "; change in area: " << change << std::endl;
                }
                if (std::abs(change) < this->area_tol)
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
