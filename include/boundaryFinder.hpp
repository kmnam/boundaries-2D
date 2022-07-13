/**
 * An implementation of a boundary-finding algorithm in the plane. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     7/13/2022
 */

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
#include <boost/multiprecision/gmp.hpp>
#include <boost/random.hpp>
#include <polytopes.hpp>
#include <linearConstraints.hpp>
#include <vertexEnum.hpp>
#include "boundaries.hpp"
#include "SQP.hpp"

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Vector_2                                            Vector_2;

constexpr int MAX_NUM_MUTATION_ATTEMPTS = 20;
constexpr int NUM_CONSECUTIVE_ITERATIONS_SATISFYING_TOLERANCE_FOR_CONVERGENCE = 5;
constexpr int INTERNAL_PRECISION = 100;
constexpr double MINDIST_BETWEEN_POINTS = 1e-8;

/**
 * A class that implements an importance sampling algorithm for iteratively
 * computing the boundary of a (compact) 2-D region, arising as the image of
 * a map from a (possibly higher-dimensional) convex polytope. 
 */
class BoundaryFinder 
{
    private:
        unsigned N;          // Number of points
        double area_tol;     // Terminate sampling when difference in area of 
                             // region bounded by alpha shapes in successive 
                             // iterations is less than this value
        unsigned max_iter;   // Maximum number of sampling iterations
        double curr_area;    // Area enclosed by last computed boundary

        // Mapping from convex polytope to the plane 
        std::function<VectorXd(const Ref<const VectorXd>&)> func;  

        // Matrix of input points in the domain of the map 
        MatrixXd input; 

        // Matrix of output points in 2-D space
        MatrixX2d points;

        // Linear inequalities that encode the convex polytopic domain
        // with rational coordinates  
        Polytopes::LinearConstraints<mpq_rational>* constraints;

        // Delaunay triangulation of the convex polytopic domain
        //
        // Note that Delaunay_triangulation here is an alias for 
        // CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dynamic_dimension_tag> > 
        Delaunay_triangulation* tri;

        // Current boundary of output points in 2-D space
        AlphaShape2DProperties curr_bound; 

        // Indices of the output points along the boundary 
        std::vector<int> vertices;

        // Random number generator 
        boost::random::mt19937 rng;

    public:
        /**
         * Constructor with input polytope constraints given as `Eigen::Matrix`
         * instances.
         *
         * @param dim      Domain (input polytope) dimension.
         * @param area_tol Area tolerance for sampling termination. 
         * @param rng      Random number generator instance. 
         * @param A        Left-hand matrix for polytope constraints. 
         * @param b        Right-hand vector for polytope constraints.
         * @param type     Inequality type. 
         * @param func     Mapping from the input polytope into the plane.  
         */
        BoundaryFinder(const double area_tol, boost::random::mt19937& rng, 
                       const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& A,
                       const Ref<const Matrix<mpq_rational, Dynamic, 1> >& b,
                       const Polytopes::InequalityType type, 
                       std::function<VectorXd(const Ref<const VectorXd>&)>& func)
        {
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(type, A, b);
            this->tri = new Delaunay_triangulation(A.cols()); 
            this->func = func;

            // Enumerate the vertices of the input polytope
            Polytopes::PolyhedralDictionarySystem* dict = new Polytopes::PolyhedralDictionarySystem(type, A, b); 
            Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices(); 
            delete dict; 

            // Obtain the Delaunay triangulation of the input polytope 
            Polytopes::triangulate(vertices, this->tri); 
        }

        /**
         * Constructor with input polytope constraints to be parsed from
         * a text file.
         *
         * @param area_tol             Area tolerance for sampling termination. 
         * @param rng                  Random number generator instance. 
         * @param constraints_filename Name of input file of polytope constraints.
         * @param func                 Mapping from the input polytope into
         *                             the plane.  
         */
        BoundaryFinder(const double area_tol, boost::random::mt19937& rng, 
                       const std::string constraints_filename,
                       const Polytopes::InequalityType type, 
                       std::function<VectorXd(const Ref<const VectorXd>&)>& func)
        {
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(type);
            this->constraints->parse(constraints_filename);
            this->tri = new Delaunay_triangulation(this->constraints->getD());  
            this->func = func;

            // Enumerate the vertices of the input polytope
            Matrix<mpq_rational, Dynamic, Dynamic> A = this->constraints->getA(); 
            Matrix<mpq_rational, Dynamic, 1> b = this->constraints->getb(); 
            Polytopes::PolyhedralDictionarySystem* dict = new Polytopes::PolyhedralDictionarySystem(type, A, b); 
            Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices(); 
            delete dict; 

            // Obtain the Delaunay triangulation of the input polytope 
            Polytopes::triangulate(vertices, this->tri); 
        }

        /**
         * Constructor with input polytope constraints and vertices to 
         * be parsed from separate text files.
         *
         * The vertices are used to triangulate the polytope.
         *
         * @param area_tol             Area tolerance for sampling termination. 
         * @param rng                  Random number generator instance. 
         * @param constraints_filename Name of input file of polytope constraints. 
         * @param vertices_filename    Name of input file of polytope vertices.
         * @param func                 Mapping from the input polytope into
         *                             the plane.  
         */
        BoundaryFinder(const double area_tol, boost::random::mt19937& rng, 
                       const std::string constraints_filename,
                       const std::string vertices_filename,
                       const Polytopes::InequalityType type, 
                       std::function<VectorXd(const Ref<const VectorXd>&)>& func)
        {
            this->N = 0;
            this->area_tol = area_tol;
            this->curr_area = 0.0;
            this->rng = rng;
            this->constraints = new Polytopes::LinearConstraints<mpq_rational>(type); 
            this->constraints->parse(constraints_filename);
            this->tri = new Delaunay_triangulation(this->constraints->getD());  
            this->func = func;

            // Parse the vertices from the given file and obtain the Delaunay
            // triangulation of the polytope  
            Polytopes::parseVerticesFile(vertices_filename, this->tri);
        }

        /**
         * Trivial destructor.
         */
        ~BoundaryFinder()
        {
            delete this->constraints;
            delete this->tri;
        }

        /**
         * Update the stored input polytope constraints with the given matrix
         * and vector. 
         *
         * @param A Left-hand matrix of polytope constraints.
         * @param b Right-hand vector of polytope constraints. 
         */
        void setConstraints(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& A,
                            const Ref<const Matrix<mpq_rational, Dynamic, 1> >& b)
        {
            this->constraints->setAb(A, b);
        }

        /**
         * Return the stored input points. 
         */ 
        MatrixXd getInput()
        {
            return this->input; 
        }

        /**
         * Return the vertices of the input polytope, by traversing over them
         * in the stored Delaunay triangulation.  
         */
        MatrixXd getVertices()
        {
            const int d = this->constraints->getD(); 
            const int n = this->tri->number_of_vertices(); 
            MatrixXd vertices(n, d);
            int i = 0; 
            for (auto it = this->tri->finite_vertices_begin(); it != this->tri->finite_vertices_end(); ++it)
            {
                // Get the coordinates of the i-th vertex 
                Delaunay_triangulation::Point p = it->point();
                for (unsigned j = 0; j < d; ++j)
                    vertices(i, j) = CGAL::to_double(p[j]);
                i++; 
            }

            return vertices; 
        }

        /**
         * Randomly sample the given number of points from the uniform density 
         * on the input polytope.
         *
         * @param npoints Number of points to sample. 
         * @returns       Matrix of sampled points (each row a point).  
         */
        MatrixXd sampleInput(const int npoints)
        {
            return Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION, INTERNAL_PRECISION>(
                this->tri, npoints, 0, this->rng
            );  
        }

        /**
         * Return the currently stored boundary of output points.
         */
        AlphaShape2DProperties getBoundary()
        {
            return this->curr_bound; 
        }

        /**
         * Initialize the sampling run by evaluating the stored mapping 
         * at the given set of points in the input polytope, and computing
         * the initial boundary.
         *
         * @param filter       Boolean function for filtering output points in 
         *                     the plane as desired.
         * @param input        Initial set of points in the input polytope at 
         *                     which to evaluate the stored mapping.
         * @param max_edges    Maximum number of edges to be contained in the
         *                     boundary. 
         * @param write_prefix Prefix of output file name to which to write 
         *                     the boundary obtained in this iteration.
         * @param verbose      If true, output intermittent messages to `stdout`.
         * @throws std::invalid_argument if the input points do not have the 
         *                               correct dimension.  
         */
        void initialize(std::function<bool(const Ref<const VectorXd>&)> filter, 
                        const Ref<const MatrixXd>& input, const unsigned max_edges, 
                        const std::string write_prefix, const bool verbose = true)
        {
            // Check that the input points have the correct dimensionality
            const int D = this->constraints->getD();  
            if (input.cols() != D)
                throw std::invalid_argument("Input points are of incorrect dimension"); 

            this->N = 0;
            this->input.resize(this->N, D);
            this->points.resize(this->N, 2);
                
            // Evaluate the stored mapping at each given input point
            for (unsigned i = 0; i < input.rows(); ++i)
            {
                VectorXd y = this->func(input.row(i));
                
                // Check that the output point is not subject to filtering and 
                // is not too close to the others 
                if (!filter(y)
                    && (this->N == 0 || (this->points.rowwise() - y.transpose()).rowwise().norm().minCoeff() > MINDIST_BETWEEN_POINTS)
                )
                {
                    this->N++;
                    this->input.conservativeResize(this->N, D); 
                    this->points.conservativeResize(this->N, 2);
                    this->input.row(this->N-1) = input.row(i);
                    this->points.row(this->N-1) = y;
                }
            }

            // Store the output coordinates in vectors 
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get new boundary (assuming that the shape is simply connected)
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - CGAL::Precondition_exception (while instantiating the polygon 
                //   for simplification) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(max_edges);
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);
            this->vertices = this->curr_bound.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::stringstream ss;
                ss << write_prefix << "-init.txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                std::ofstream outfile;
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (unsigned i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (unsigned j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close(); 
            }

            // Compute enclosed area and test for convergence
            this->curr_area = this->curr_bound.area;
            if (verbose)
            {
                std::cout << "[INIT] Initializing; enclosed area: " << this->curr_area
                          << "; " << this->vertices.size() << " boundary points" 
                          << "; " << this->points.rows() << " total points"
                          << std::endl;  
            }
        }

        /**
         * Take one "step" in the boundary-sampling algorithm as follows: 
         *
         * 1) Compute the boundary of the output points accrued thus far. 
         * 2) "Mutate" (randomly perturb) the input points in the pre-image 
         *    of the determined boundary, according to the given `mutate` 
         *    function.
         * 3) Evaluate the stored mapping on these new input points.
         *
         * The return value indicates whether or not the area enclosed by the 
         * boundary *obtained prior to mutation* has converged to within
         * `this->area_tol`.
         *
         * Note that this method assumes that the boundary is simply connected.
         *
         * @param dist         Random distribution from which to sample increments
         *                     by which to mutate each input point coordinate.
         * @param filter       Boolean function for filtering output points in the 
         *                     plane as desired.
         * @param iter         Iteration number. 
         * @param max_edges    Maximum number of edges to be contained in the
         *                     boundary. 
         * @param write_prefix Prefix of output file name to which to write 
         *                     the boundary obtained in this iteration.
         * @param verbose      If true, output intermittent messages to `stdout`.
         * @returns True if the area enclosed by the boundary (obtained prior 
         *          to mutation) has converged to within `this->area_tol`. 
         */
        bool step(boost::random::uniform_real_distribution<double>& dist,
                  std::function<bool(const Ref<const VectorXd>&)> filter, 
                  const unsigned iter, const unsigned max_edges,
                  const std::string write_prefix, const bool verbose = true)
        {
            // Store the output coordinates in vectors 
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // For each of the points in the boundary, mutate the corresponding
            // model parameters once, and evaluate the stored mapping at these 
            // mutated parameter values
            const int D = this->constraints->getD(); 
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                bool filtered = true;
                double mindist = 0.0;
                unsigned j = 0;
                VectorXd p = this->input.row(this->vertices[i]);
                VectorXd q, z;
                while ((filtered || mindist < MINDIST_BETWEEN_POINTS) && j < MAX_NUM_MUTATION_ATTEMPTS)
                {
                    // Evaluate the given function at a randomly generated 
                    // parameter point
                    VectorXd m(p);
                    for (int k = 0; k < D; ++k)
                        m(k) += dist(this->rng); 
                    q = this->constraints->template nearestL2<double>(m).template cast<double>();
                    z = this->func(q);

                    // Check that the mutation does not give rise to a
                    // filtered point 
                    filtered = filter(z);
                    
                    // Check that the mutation does not give rise to an
                    // already encountered point 
                    mindist = (this->points.rowwise() - z.transpose()).rowwise().norm().minCoeff();
                    j++;
                }
                if (!filtered && mindist > MINDIST_BETWEEN_POINTS)
                {
                    this->N++;
                    this->input.conservativeResize(this->N, D); 
                    this->points.conservativeResize(this->N, 2);
                    this->input.row(this->N-1) = q;
                    this->points.row(this->N-1) = z;
                }
            }

            // Get new boundary (assuming that the shape is simply connected)
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - CGAL::Precondition_exception (while instantiating the polygon 
                //   for simplification) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(max_edges);
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);
            this->vertices = this->curr_bound.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                std::ofstream outfile;
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (unsigned i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (unsigned j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close(); 
            }

            // Compute enclosed area and test for convergence
            double area = this->curr_bound.area;
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

            return (std::abs(change) < this->area_tol * (area - change));
        }

        /**
         * "Pull" the boundary points along their outward normal vectors
         * with sequential quadratic programming.
         *
         * The return value indicates whether or not the enclosed area has 
         * converged to within `this->area_tol`.
         *
         * Note that this method assumes that the boundary is simply connected.
         *
         * @param optimizer               Pointer to existing `SQPOptimizer<double>`
         *                                instance. This instance is assumed to
         *                                contain correct information regarding
         *                                the constraints.  
         * @param filter                  Boolean function for filtering output
         *                                points in the plane as desired.
         * @param epsilon                 Distance by which the output points 
         *                                along the boundary should be pulled. 
         * @param max_iter                Maximum number of iterations for SQP. 
         * @param sqp_tol                 Tolerance for assessing convergence
         *                                in SQP.
         * @param iter                    Iteration number.  
         * @param max_edges               Maximum number of edges to be contained
         *                                in the boundary.
         * @param tau                     Rate at which step-sizes are decreased
         *                                during each SQP iteration.
         * @param delta                   Increment for finite-differences 
         *                                approximation during each SQP iteration.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semi-definiteness).
         * @param use_only_armijo         If true, use only the Armijo condition
         *                                to determine step-size during each 
         *                                SQP iteration.
         * @param use_strong_wolfe        If true, use the strong Wolfe conditions
         *                                to determine step-size during each 
         *                                SQP iteration. Disregarded if
         *                                `use_only_armijo == true`.  
         * @param hessian_modify_max_iter Maximum number of Hessian matrix
         *                                modification iterations (for ensuring
         *                                positive semi-definiteness).  
         * @param write_prefix            Prefix of output file name to which to  
         *                                write the boundary obtained in this
         *                                iteration.
         * @param regularize              Regularization method: `NOREG`, `L1`,
         *                                or `L2`. Set to `NOREG` by default.  
         * @param regularize_weight       Regularization weight. If `regularize`
         *                                is `NOREG`, then this value is ignored. 
         * @param c1                      Pre-factor for testing Armijo's 
         *                                condition during each SQP iteration.
         * @param c2                      Pre-factor for testing the curvature 
         *                                condition during each SQP iteration. 
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param sqp_verbose             If true, output intermittent messages 
         *                                during SQP to `stdout`.
         * @returns True if the area enclosed by the boundary (obtained prior 
         *          to mutation) has converged to within `this->area_tol`. 
         */
        bool pull(SQPOptimizer<double>* optimizer, 
                  std::function<bool(const Ref<const VectorXd>&)> filter, 
                  const double epsilon, const unsigned max_iter, const double sqp_tol,
                  const unsigned iter, const unsigned max_edges,
                  const double tau, const double delta, const double beta,
                  const bool use_only_armijo, const bool use_strong_wolfe,
                  const unsigned hessian_modify_max_iter,
                  const std::string write_prefix,
                  const RegularizationMethod regularize = NOREG,
                  const T regularize_weight = 0, const double c1 = 1e-4,
                  const double c2 = 0.9, const bool verbose = true,
                  const bool sqp_verbose = false)
        {
            // Store point coordinates in two vectors
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Obtain the outward vertex normals along the boundary and,
            // for each vertex in the boundary, "pull" along its outward
            // normal by distance epsilon
            MatrixXd pulled(this->vertices.size(), 2);
            std::vector<Vector_2> normals = this->curr_bound.outwardVertexNormals();
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                Vector_2 v(x[this->vertices[i]], y[this->vertices[i]]);
                Vector_2 v_pulled = v + epsilon * normals[i];
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

            // For each vertex in the boundary, minimize the distance to the
            // pulled vertex with a feasible parameter point
            for (unsigned i = 0; i < this->vertices.size(); ++i)
            {
                // Minimize the appropriate objective function
                VectorXd target = pulled.row(i);
                auto obj = [this, target](const Ref<const VectorXd>& x)
                {
                    return (target - this->func(x)).squaredNorm();
                };
                VectorXd x_init = this->input.row(this->vertices[i]);
                VectorXd l_init = VectorXd::Ones(this->constraints->getN())
                    - this->constraints->active(x_init.cast<mpq_rational>()).template cast<double>();
                VectorXd q = optimizer->run(
                    obj, x_init, l_init, tau, delta, beta, max_iter, sqp_tol,
                    sqp_tol, BFGS, regularize, regularize_weight, use_only_armijo,
                    use_strong_wolfe, hessian_modify_max_iter, c1, c2, sqp_verbose
                );
                VectorXd z = this->func(q); 
                
                // Check that the mutation did not give rise to an already 
                // computed point
                double mindist = (this->points.rowwise() - z.transpose()).rowwise().norm().minCoeff();
                if (!filter(z) && mindist > MINDIST_BETWEEN_POINTS)
                {
                    this->N++;
                    this->input.conservativeResize(this->N, this->constraints->getD()); 
                    this->points.conservativeResize(this->N, 2);
                    this->input.row(this->N-1) = q;
                    this->points.row(this->N-1) = z;
                }
            }

            // Get new boundary (assuming that the shape is simply connected)
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - CGAL::Precondition_exception (while instantiating the polygon 
                //   for simplification) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(max_edges);
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (CGAL::Precondition_exception& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }
            catch (std::runtime_error& e)
            {
                // Try with tag == false
                //
                // This may throw a CGAL::Assertion_exception (while instantiating
                // the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(max_edges);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);
            this->vertices = this->curr_bound.vertices;

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                std::ofstream outfile;
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (unsigned i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (unsigned j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close(); 
            }

            // Compute enclosed area and test for convergence
            double area = this->curr_bound.area;
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

            return (std::abs(change) < this->area_tol * (area - change));
        }

        /**
         * Run the full boundary-sampling algorithm until convergence, up to
         * the maximum number of iterations.
         *
         * @param mutate_delta            Maximum increment by which any input 
         *                                point coordinate may be mutated.
         * @param filter                  Boolean function for filtering output
         *                                points in the plane as desired.
         * @param init_input              Initial set of points in the input
         *                                polytope at which to evaluate the
         *                                stored mapping.
         * @param min_step_iter           Minimum number of step iterations. 
         * @param max_step_iter           Maximum number of step iterations. 
         * @param min_pull_iter           Minimum number of pull iterations. 
         * @param max_pull_iter           Maximum number of pull iterations.
         * @param sqp_max_iter            Maximum number of SQP iterations per
         *                                pull iteration.
         * @param sqp_tol                 Tolerance for assessing convergence
         *                                in SQP.
         * @param max_edges               Maximum number of edges to be contained
         *                                in the boundary. 
         * @param tau                     Rate at which step-sizes are decreased
         *                                during each SQP iteration.
         * @param delta                   Increment for finite-differences 
         *                                approximation during each SQP iteration.
         * @param beta                    Increment for Hessian matrix modification
         *                                (for ensuring positive semi-definiteness).
         * @param use_only_armijo         If true, use only the Armijo condition
         *                                to determine step-size during each 
         *                                SQP iteration. 
         * @param use_strong_wolfe        If true, use the strong Wolfe conditions
         *                                to determine step-size during each 
         *                                SQP iteration. Disregarded if 
         *                                `use_only_armijo == true`. 
         * @param hessian_modify_max_iter Maximum number of Hessian matrix
         *                                modification iterations (for ensuring
         *                                positive semi-definiteness).  
         * @param write_prefix            Prefix of output file name to which to  
         *                                write the boundary obtained in each
         *                                iteration.
         * @param regularize              Regularization method: `NOREG`, `L1`,
         *                                or `L2`. Set to `NOREG` by default.  
         * @param regularize_weight       Regularization weight. If `regularize`
         *                                is `NOREG`, then this value is ignored. 
         * @param c1                      Pre-factor for testing Armijo's 
         *                                condition during each SQP iteration.
         * @param c2                      Pre-factor for testing the curvature 
         *                                condition during each SQP iteration. 
         * @param verbose                 If true, output intermittent messages
         *                                to `stdout`.
         * @param sqp_verbose             If true, output intermittent messages 
         *                                during SQP to `stdout`.
         */
        void run(const double mutate_delta,
                 std::function<bool(const Ref<const VectorXd>&)> filter, 
                 const Ref<const MatrixXd>& init_input, 
                 const unsigned min_step_iter, const unsigned max_step_iter,
                 const unsigned min_pull_iter, const unsigned max_pull_iter,
                 const unsigned sqp_max_iter, const double sqp_tol,
                 const unsigned max_edges, const double tau, const double delta,
                 const double beta, const bool use_only_armijo, 
                 const bool use_strong_wolfe,
                 const unsigned hessian_modify_max_iter,
                 const std::string write_prefix,
                 const RegularizationMethod regularize = NOREG, 
                 const T regularize_weight = 0, const double c1 = 1e-4,
                 const double c2 = 0.9, const bool verbose = true,
                 const bool sqp_verbose = false)
        {
            // Initialize the sampling run ...
            this->initialize(filter, init_input, max_edges, write_prefix, verbose);

            // ... then step through the boundary-finding algorithm up to the
            // maximum number of iterations ...
            unsigned i = 1;
            bool terminate = false;
            unsigned n_converged = 0;
            boost::random::uniform_real_distribution<double> dist(-mutate_delta, mutate_delta);  
            while (i - 1 < min_step_iter || (i - 1 < max_step_iter && !terminate))
            {
                bool result = this->step(dist, filter, i, max_edges, write_prefix, verbose);
                if (!result)
                    n_converged = 0;
                else
                    n_converged++; 
                terminate = (n_converged >= NUM_CONSECUTIVE_ITERATIONS_SATISFYING_TOLERANCE_FOR_CONVERGENCE);
                i++;
            }

            // ... then turn to pulling the boundary points outward
            unsigned j = 0;
            terminate = false;
            n_converged = 0;
            double epsilon = 0.1 * std::sqrt(this->curr_area);
            SQPOptimizer<double>* optimizer = new SQPOptimizer<double>(this->constraints); 
            while (j < min_pull_iter || (j < max_pull_iter && !terminate))
            {
                if (verbose) std::cout << "Pulling by epsilon = " << epsilon << std::endl;  
                bool result = this->pull(
                    optimizer, filter, epsilon, sqp_max_iter, sqp_tol, i + j,
                    max_edges, tau, delta, beta, use_only_armijo, use_strong_wolfe, 
                    hessian_modify_max_iter, write_prefix, regularize, 
                    regularize_weight, c1, c2, verbose, sqp_verbose
                );
                if (!result)
                    n_converged = 0;
                else
                    n_converged++; 
                terminate = (n_converged >= NUM_CONSECUTIVE_ITERATIONS_SATISFYING_TOLERANCE_FOR_CONVERGENCE);
                j++;
                epsilon = 0.1 * std::sqrt(this->curr_area);
            }
            delete optimizer;

            // Write final boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::stringstream ss;
                ss << write_prefix << "-final.txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                std::ofstream outfile;
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (unsigned i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (unsigned j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close(); 
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
