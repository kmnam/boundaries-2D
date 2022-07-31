/**
 * An implementation of a boundary-finding algorithm in the plane. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     7/28/2022
 */

#ifndef BOUNDARY_FINDER_HPP
#define BOUNDARY_FINDER_HPP
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
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
 * Sample `k` items from the range from `0` to `n - 1` (inclusive) without
 * replacement.
 *
 * @param n   Length of input range. Must be positive. 
 * @param k   Number of items to sample. Must be non-negative.
 * @param rng Random number generator.  
 * @returns   Another `std::vector` instance containing the subsample.
 * @throws std::invalid_argument If `n < k`. 
 */
std::vector<int> sampleWithoutReplacement(const int n, const int k,
                                          boost::random::mt19937& rng)
{
    // Check that n > 0, k >= 0 and n >= k
    if (n < k)
        throw std::invalid_argument("Cannot sample k items from a space of size n < k");
    else if (n == 0)
        throw std::invalid_argument("Cannot sample from empty space");
    else if (n < 0)
        throw std::invalid_argument("Sample space has size n < 0"); 
    else if (k < 0)
        throw std::invalid_argument("Cannot sample k < 0 items"); 

    // If n == k, simply return the range from 0 to n-1
    std::vector<int> sample; 
    if (n == k)
    {
        for (int i = 0; i < n; ++i)
            sample.push_back(i); 
        return sample; 
    }
    // Otherwise, if k == 0, simply return an empty vector
    else if (k == 0)
    {
        return sample;
    }

    // Populate a priority queue with 0..n-1, with priority defined by randomly
    // generated weights between 0 and 1
    // 
    // Note that pairs are compared lexicographically, so placing the weights 
    // first allows for the weights to be used as priorities
    boost::random::uniform_01<double> dist;
    std::priority_queue<std::pair<double, int> > queue;
    for (int i = 0; i < n; ++i)
        queue.emplace(std::make_pair(dist(rng), i));
    
    // Return the first k items in the queue
    for (int i = 0; i < k; ++i)
    {
        sample.push_back(queue.top().second);
        queue.pop();
    }

    return sample;  
}

/**
 * A class that implements an importance sampling algorithm for iteratively
 * computing the boundary of a (compact) 2-D region, arising as the image of
 * a map from a (possibly higher-dimensional) convex polytope. 
 */
class BoundaryFinder 
{
    private:
        int N;               // Number of points
        double area_tol;     // Terminate sampling when difference in area of 
                             // region bounded by alpha shapes in successive 
                             // iterations is less than this value
        int max_iter;        // Maximum number of sampling iterations
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

        // Current *simplified* boundary of output points in 2-D space
        // (only defined if simplification is desired)
        AlphaShape2DProperties curr_simplified;

        // Flag indicating whether the current boundary has been simplified
        bool simplified;  

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
            this->simplified = false; 

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
            this->simplified = false; 

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
            this->simplified = false; 

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
                for (int j = 0; j < d; ++j)
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
         * Return the currently stored boundary of the output points.
         */
        AlphaShape2DProperties getBoundary()
        {
            return this->curr_bound; 
        }

        /**
         * Return the simplified boundary of the output points. 
         */
        AlphaShape2DProperties getSimplifiedBoundary()
        {
            return this->curr_simplified; 
        }

        /**
         * Initialize the sampling run by evaluating the stored mapping 
         * at the given set of points in the input polytope, and computing
         * the initial boundary.
         *
         * @param filter           Boolean function for filtering output points
         *                         in the plane as desired.
         * @param input            Initial set of points in the input polytope 
         *                         at which to evaluate the stored mapping.
         * @param max_edges        Maximum number of edges to be contained in
         *                         the boundary. If zero, the boundary is kept
         *                         unsimplified.
         * @param n_keep_interior  Number of interior points to keep from the
         *                         unsimplified boundary. If this number exceeds
         *                         the total number of interior points, then all 
         *                         interior points are kept.
         * @param write_prefix     Prefix of output file name to which to write 
         *                         the boundary obtained in this iteration.
         * @param verbose          If true, output intermittent messages to `stdout`.
         * @throws std::invalid_argument If the input points do not have the 
         *                               correct dimension.  
         */
        void initialize(std::function<bool(const Ref<const VectorXd>&)> filter, 
                        const Ref<const MatrixXd>& input, const int max_edges, 
                        const int n_keep_interior, const std::string write_prefix,
                        const bool verbose = true)
        {
            // Check that the input points have the correct dimensionality
            const int D = this->constraints->getD();  
            if (input.cols() != D)
                throw std::invalid_argument("Input points are of incorrect dimension"); 

            this->N = 0;
            this->input.resize(this->N, D);
            this->points.resize(this->N, 2);
                
            // Evaluate the stored mapping at each given input point
            for (int i = 0; i < input.rows(); ++i)
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

            // Get new boundary (assuming that the shape is simply connected)
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(verbose); 
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose); 
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
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);

            // Remove as many points from the interior as desired
            std::unordered_set<int> boundary_indices(
                this->curr_bound.vertices.begin(), this->curr_bound.vertices.end()
            );
            std::vector<int> interior_indices; 
            for (int i = 0; i < this->curr_bound.np; ++i)
            {
                if (boundary_indices.find(i) == boundary_indices.end())
                    interior_indices.push_back(i); 
            }
            std::vector<int> interior_indices_to_delete;
            int n_to_delete = this->curr_bound.np - this->curr_bound.nv - n_keep_interior;
            if (n_to_delete < 0)
            {   // Check that the number of points to be deleted is not negative 
                n_to_delete = 0; 
            }
            std::vector<int> idx = sampleWithoutReplacement(
                this->curr_bound.np - this->curr_bound.nv, n_to_delete, this->rng
            );
            for (const int i : idx)
                interior_indices_to_delete.push_back(interior_indices[i]);  
            this->curr_bound.deleteInteriorPoints(interior_indices_to_delete);

            // Update this->N, this->input, this->points accordingly 
            std::vector<int> indices_to_keep; 
            std::unordered_set<int> interior_indices_to_delete_set(
                interior_indices_to_delete.begin(), interior_indices_to_delete.end()
            ); 
            for (int i = 0; i < this->N; ++i)
            {
                if (interior_indices_to_delete_set.find(i) == interior_indices_to_delete_set.end())
                    indices_to_keep.push_back(i); 
            }
            this->input = this->input(indices_to_keep, Eigen::all).eval(); 
            this->points = this->points(indices_to_keep, Eigen::all).eval(); 
            this->N = this->curr_bound.np;
            if (verbose)
            {
                std::cout << "- removed " << interior_indices_to_delete.size()
                          << " interior points" << std::endl;
            } 

            // If desired, simplify the current boundary
            if (max_edges > 0 && this->curr_bound.edges.size() > max_edges)
            {
                try
                { 
                    // This line may raise a CGAL::Precondition_exception while 
                    // instantiating the polygon for simplification
                    this->curr_simplified = simplifyAlphaShape(this->curr_bound, max_edges, verbose);  
                }
                catch (CGAL::Precondition_exception& e)
                {
                    // TODO Perhaps a better option exists here 
                    throw; 
                }
                // Re-orient the points so that the boundary is traversed 
                // clockwise
                this->curr_simplified.orient(CGAL::RIGHT_TURN); 
                this->simplified = true; 
            }

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges
                std::ofstream outfile;  
                std::stringstream ss;
                ss << write_prefix << "-init.txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (int i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (int j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close();
                ss.clear(); 
                ss.str(std::string()); 

                // Also write the simplified boundary to file, if simplification
                // was performed ... 
                if (this->simplified)
                {
                    // Write the boundary points, vertices, and edges 
                    ss << write_prefix << "-init-simplified.txt";
                    this->curr_simplified.write(ss.str());

                    // Write the input vectors passed into the given function to
                    // yield the boundary points
                    outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                    if (outfile.is_open())
                    {
                        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                        for (int i = 0; i < this->input.rows(); ++i)
                        {
                            outfile << "INPUT\t"; 
                            for (int j = 0; j < this->input.cols() - 1; ++j)
                                outfile << this->input(i, j) << '\t'; 
                            outfile << this->input(i, this->input.cols() - 1) << std::endl;
                        }
                    }
                    outfile.close(); 
                }
            }

            // Compute enclosed area and test for convergence
            if (verbose)
            {
                std::cout << "[INIT] Initializing; "
                          << this->curr_bound.nv << " boundary points; " 
                          << this->curr_bound.np << " total points; "
                          << "enclosed area: " << this->curr_bound.area
                          << std::endl; 
                if (this->simplified)
                {
                    std::cout << ">>>>>> Simplified to "
                              << this->curr_simplified.nv
                              << " boundary points; enclosed area: " 
                              << this->curr_simplified.area << std::endl;
                    this->curr_area = this->curr_simplified.area; 
                }
                else
                {
                    this->curr_area = this->curr_bound.area; 
                }
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
         * @param dist               Random distribution from which to sample
         *                           increments by which to mutate each input
         *                           point coordinate.
         * @param filter             Boolean function for filtering output points
         *                           in the plane as desired.
         * @param iter               Iteration number. 
         * @param max_edges          Maximum number of edges to be contained in
         *                           the boundary. If zero, the boundary is kept
         *                           unsimplified.
         * @param n_keep_interior    Number of interior points to keep in the
         *                           unsimplified boundary. If this number exceeds
         *                           the total number of interior points, then all 
         *                           interior points are kept. 
         * @param n_keep_origbound   If the boundary is simplified, the number of
         *                           points in the *original (unsimplified)
         *                           boundary* that should be randomly sampled
         *                           (without replacement) to be preserved. 
         * @param n_mutate_origbound If the boundary is simplified, the number of
         *                           points *among those chosen to be preserved*
         *                           that should be further randomly sampled 
         *                           (without replacement) to be mutated. 
         * @param write_prefix       Prefix of output file name to which to write 
         *                           the boundary obtained in this iteration.
         * @param verbose            If true, output intermittent messages to
         *                           `stdout`.
         * @returns True if the area enclosed by the boundary (obtained prior 
         *          to mutation) has converged to within `this->area_tol`. 
         */
        bool step(boost::random::uniform_real_distribution<double>& dist,
                  std::function<bool(const Ref<const VectorXd>&)> filter, 
                  const int iter, const int max_edges, int n_keep_interior,
                  int n_keep_origbound, int n_mutate_origbound, 
                  const std::string write_prefix, const bool verbose = true)
        {
            // Reset input parameters if necessary 
            int n_interior = this->curr_bound.np - this->curr_bound.nv; 
            int origbound_nv = this->curr_bound.nv - this->curr_simplified.nv;
            if (n_keep_origbound > origbound_nv)
                n_keep_origbound = origbound_nv; 
            if (n_mutate_origbound > n_keep_origbound)
                n_mutate_origbound = n_keep_origbound; 

            // First, if the boundary was simplified, identify the subset of 
            // boundary vertices in the unsimplified boundary and not in 
            // the simplified boundary to keep; otherwise, keep every point 
            // in the (unsimplified) boundary
            std::vector<int> origbound_indices, origbound_indices_to_keep;
            std::unordered_set<int> boundary_indices(
                this->curr_bound.vertices.begin(),
                this->curr_bound.vertices.end()
            );
            std::unordered_set<int> simplified_indices; 
            if (!this->simplified)
            {
                for (const int i : this->curr_bound.vertices)
                    origbound_indices_to_keep.push_back(i); 
            }
            else
            {
                simplified_indices.insert(
                    this->curr_simplified.vertices.begin(),
                    this->curr_simplified.vertices.end()
                );

                // Get the complement of vertices in the unsimplified boundary 
                // that do not lie in the simplified boundary
                for (const int i : boundary_indices)
                {
                    if (simplified_indices.find(i) == simplified_indices.end())
                        origbound_indices.push_back(i); 
                }

                // Sample a subset of these vertices 
                std::vector<int> idx = sampleWithoutReplacement(
                    origbound_indices.size(), n_keep_origbound, this->rng
                ); 
                for (const int i : idx)
                    origbound_indices_to_keep.push_back(origbound_indices[i]); 
            }

            // Keep only the vertices in the unsimplified boundary chosen above
            //
            // Note that this slicing is only necessary if the boundary was 
            // simplified  
            if (this->simplified)
            {
                int n_keep_prior = this->curr_simplified.nv + n_interior + n_keep_origbound;
                VectorXi indices_to_keep_prior(n_keep_prior); 
                int j = 0;

                // Keep all vertices in the simplified boundary, all points
                // in the interior, and the vertices in the unsimplified 
                // boundary chosen above
                for (int i = 0; i < this->curr_bound.np; ++i)
                {
                    if (boundary_indices.find(i) == boundary_indices.end())
                    {
                        indices_to_keep_prior(j) = i;
                        j++;
                    }
                }
                for (const int i : this->curr_simplified.vertices) 
                {
                    indices_to_keep_prior(j) = i; 
                    j++;
                }
                for (const int i : origbound_indices_to_keep)
                {
                    indices_to_keep_prior(j) = i;
                    j++; 
                }
                this->N = n_keep_prior; 
                this->input = this->input(indices_to_keep_prior, Eigen::all).eval(); 
                this->points = this->points(indices_to_keep_prior, Eigen::all).eval();
            }
            if (verbose)
            {
                if (!this->simplified)
                {
                    std::cout << "- preserved " << this->N << " points: "
                              << n_interior << " interior, "
                              << this->curr_bound.nv << " boundary"
                              << std::endl; 
                } 
                else 
                {
                    std::cout << "- preserved " << this->N << " points: "
                              << n_interior << " interior, "
                              << this->curr_simplified.nv + n_keep_origbound
                              << " boundary" << std::endl;
                }
            }

            const int D = this->constraints->getD();
            VectorXi to_mutate;

            // If the boundary was not simplified, then mutate every point in
            // the boundary
            if (!this->simplified)
            {
                to_mutate.resize(this->curr_bound.nv);
                for (int i = 0; i < this->curr_bound.nv; ++i)
                    to_mutate(i) = this->curr_bound.vertices[i];
            }
            // Otherwise, then mutate every point in the *simplified* boundary,
            // plus the desired number of points in the unsimplified boundary 
            else
            {
                const int n_mutate = this->curr_simplified.nv + n_mutate_origbound;
                to_mutate.resize(n_mutate);

                // The vertices in the simplified boundary are now in one 
                // contiguous chunk in this->input / this->points (see above)
                for (int i = 0; i < this->curr_simplified.nv; ++i)
                    to_mutate(i) = n_interior + i;
                
                // Choose n_mutate_origbound number of vertices among the 
                // vertices in the unsimplified boundary *that were chosen to 
                // be kept above*
                std::vector<int> idx = sampleWithoutReplacement(
                    n_keep_origbound, n_mutate_origbound, this->rng
                );
                for (int i = 0; i < n_mutate_origbound; ++i)
                    to_mutate(this->curr_simplified.nv + i) = n_interior + this->curr_simplified.nv + idx[i];
            }
            if (verbose)
            {
                std::cout << "- mutating " << to_mutate.size()
                          << " boundary points" << std::endl; 
            } 

            // Now proceed to mutate each point ... 
            for (int i = 0; i < to_mutate.size(); ++i)
            {
                bool filtered = true;
                double mindist = 0.0;
                int j = 0;
                VectorXd p = this->input.row(to_mutate(i));
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
            if (verbose)
            {
                std::cout << "- mutations complete; augmented point-set contains "
                          << this->N << " points" << std::endl;
            }

            // Get new boundary (assuming that the shape is simply connected)
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(verbose);
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose);
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
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);

            // Update input parameters as necessary
            n_interior = this->curr_bound.np - this->curr_bound.nv;   // New number of interior points 
            if (n_keep_interior > n_interior)
                n_keep_interior = n_interior; 

            // Remove as many points from the interior as desired
            boundary_indices.clear(); 
            boundary_indices.insert(
                this->curr_bound.vertices.begin(), this->curr_bound.vertices.end()
            );
            std::vector<int> interior_indices; 
            for (int i = 0; i < this->curr_bound.np; ++i)
            {
                if (boundary_indices.find(i) == boundary_indices.end())
                    interior_indices.push_back(i); 
            }
            std::vector<int> interior_indices_to_delete;
            int n_to_delete = this->curr_bound.np - this->curr_bound.nv - n_keep_interior;
            if (n_to_delete < 0)
            {   // Check that the number of points to be deleted is not negative 
                n_to_delete = 0; 
            }
            std::vector<int> idx = sampleWithoutReplacement(
                this->curr_bound.np - this->curr_bound.nv, n_to_delete, this->rng
            );
            for (const int i : idx)
                interior_indices_to_delete.push_back(interior_indices[i]);  
            this->curr_bound.deleteInteriorPoints(interior_indices_to_delete);

            // Update this->N, this->input, this->points accordingly 
            std::vector<int> indices_to_keep; 
            std::unordered_set<int> interior_indices_to_delete_set(
                interior_indices_to_delete.begin(), interior_indices_to_delete.end()
            ); 
            for (int i = 0; i < this->N; ++i)
            {
                if (interior_indices_to_delete_set.find(i) == interior_indices_to_delete_set.end())
                    indices_to_keep.push_back(i); 
            }
            this->input = this->input(indices_to_keep, Eigen::all).eval(); 
            this->points = this->points(indices_to_keep, Eigen::all).eval(); 
            this->N = this->curr_bound.np;
            if (verbose)
            {
                std::cout << "- removed " << interior_indices_to_delete.size()
                          << " interior points" << std::endl;
            } 

            // If desired, simplify the current boundary
            if (max_edges > 0 && this->curr_bound.edges.size() > max_edges)
            {
                try
                { 
                    // This line may raise a CGAL::Precondition_exception while 
                    // instantiating the polygon for simplification
                    this->curr_simplified = simplifyAlphaShape(this->curr_bound, max_edges, verbose);  
                }
                catch (CGAL::Precondition_exception& e)
                {
                    // TODO Perhaps a better option exists here 
                    throw; 
                }
                // Re-orient the points so that the boundary is traversed 
                // clockwise
                this->curr_simplified.orient(CGAL::RIGHT_TURN); 
                this->simplified = true; 
            }

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::ofstream outfile;  
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (int i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (int j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close();
                ss.clear(); 
                ss.str(std::string()); 

                // Also write the simplified boundary to file, if simplification
                // was performed ... 
                if (this->simplified)
                {
                    // Write the boundary points, vertices, and edges 
                    ss << write_prefix << "-pass" << iter << "-simplified.txt";
                    this->curr_simplified.write(ss.str());

                    // Write the input vectors passed into the given function to
                    // yield the boundary points
                    outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                    if (outfile.is_open())
                    {
                        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                        for (int i = 0; i < this->input.rows(); ++i)
                        {
                            outfile << "INPUT\t"; 
                            for (int j = 0; j < this->input.cols() - 1; ++j)
                                outfile << this->input(i, j) << '\t'; 
                            outfile << this->input(i, this->input.cols() - 1) << std::endl;
                        }
                    }
                    outfile.close(); 
                }
            }

            // Compute enclosed area and test for convergence
            double change = (
                this->simplified
                ? this->curr_simplified.area - this->curr_area
                : this->curr_bound.area - this->curr_area
            );
            if (verbose)
            {
                std::cout << "[STEP] Iteration " << iter << "; "
                          << this->curr_bound.nv << " boundary points; " 
                          << this->curr_bound.np << " total points; "
                          << "enclosed area: " << this->curr_bound.area << "; "
                          << "change: " << this->curr_bound.area - this->curr_area
                          << std::endl;
                if (this->simplified)
                {
                    std::cout << ">>>>>> Simplified to "
                              << this->curr_simplified.nv
                              << " boundary points; enclosed area: "
                              << this->curr_simplified.area << "; "
                              << "change: " << this->curr_simplified.area - this->curr_area
                              << std::endl;
                }
            }
            bool converged = (std::abs(change) < this->area_tol * this->curr_area); 
            this->curr_area = (
                this->simplified ? this->curr_simplified.area : this->curr_bound.area
            ); 

            return converged;
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
         *                                in the boundary. If zero, the boundary
         *                                is kept unsimplified.
         * @param n_keep_interior         Number of interior points to keep in
         *                                the unsimplified boundary. If this
         *                                number exceeds the total number of
         *                                interior points, then all interior
         *                                points are kept. 
         * @param n_keep_origbound        If the boundary is simplified, the
         *                                number of points in the *original
         *                                (unsimplified) boundary* that should
         *                                be randomly sampled (without replacement)
         *                                to be preserved.
         * @param n_pull_origbound        If the boundary is simplified, the
         *                                number of points *among those chosen
         *                                to be preserved* that should be further
         *                                randomly sampled (without replacement)
         *                                to be pulled. 
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
         * @param write_pulled_points     If true, write the points that were 
         *                                pulled, their corresponding normal 
         *                                vectors, and the resulting points 
         *                                to file. 
         * @returns True if the area enclosed by the boundary (obtained prior 
         *          to pulling) has converged to within `this->area_tol`. 
         */
        bool pull(SQPOptimizer<double>* optimizer, 
                  std::function<bool(const Ref<const VectorXd>&)> filter, 
                  const double epsilon, const int max_iter,
                  const double sqp_tol, const int iter, const int max_edges,
                  int n_keep_interior, int n_keep_origbound, int n_pull_origbound,
                  const double tau, const double delta, const double beta,
                  const bool use_only_armijo, const bool use_strong_wolfe,
                  const int hessian_modify_max_iter,
                  const std::string write_prefix,
                  const RegularizationMethod regularize = NOREG,
                  const double regularize_weight = 0, const double c1 = 1e-4,
                  const double c2 = 0.9, const bool verbose = true,
                  const bool sqp_verbose = false,
                  const bool write_pulled_points = false)
        {
            // Reset input parameters if necessary 
            int n_interior = this->curr_bound.np - this->curr_bound.nv; 
            int origbound_nv = this->curr_bound.nv - this->curr_simplified.nv;
            if (n_keep_origbound > origbound_nv)
                n_keep_origbound = origbound_nv; 
            if (n_pull_origbound > n_keep_origbound)
                n_pull_origbound = n_keep_origbound; 

            // First, if the boundary was simplified, identify the subset of 
            // boundary vertices in the unsimplified boundary and not in 
            // the simplified boundary to keep; otherwise, keep every point 
            // in the (unsimplified) boundary
            std::vector<int> origbound_indices, origbound_indices_to_keep;
            std::unordered_set<int> boundary_indices(
                this->curr_bound.vertices.begin(),
                this->curr_bound.vertices.end()
            );
            std::unordered_set<int> simplified_indices; 
            if (!this->simplified)
            {
                for (const int i : this->curr_bound.vertices)
                    origbound_indices_to_keep.push_back(i); 
            }
            else
            {
                simplified_indices.insert(
                    this->curr_simplified.vertices.begin(),
                    this->curr_simplified.vertices.end()
                );

                // Get the complement of vertices in the unsimplified boundary 
                // that do not lie in the simplified boundary
                for (const int i : boundary_indices)
                {
                    if (simplified_indices.find(i) == simplified_indices.end())
                        origbound_indices.push_back(i); 
                }

                // Sample a subset of these vertices to keep ... 
                std::vector<int> idx = sampleWithoutReplacement(
                    origbound_indices.size(), n_keep_origbound, this->rng
                ); 
                for (const int i : idx)
                    origbound_indices_to_keep.push_back(origbound_indices[i]);
            }

            // Keep only the vertices in the unsimplified boundary chosen above
            //
            // Note that this slicing is only necessary if the boundary was 
            // simplified  
            if (this->simplified)
            {
                int n_keep_prior = this->curr_simplified.nv + n_interior + n_keep_origbound;
                VectorXi indices_to_keep_prior(n_keep_prior); 
                int j = 0;

                // Keep all vertices in the simplified boundary, all points
                // in the interior, and the vertices in the unsimplified 
                // boundary chosen above
                for (int i = 0; i < this->curr_bound.np; ++i)
                {
                    if (boundary_indices.find(i) == boundary_indices.end())
                    {
                        indices_to_keep_prior(j) = i;
                        j++;
                    }
                }
                for (const int i : this->curr_simplified.vertices) 
                {
                    indices_to_keep_prior(j) = i; 
                    j++;
                }
                for (const int i : origbound_indices_to_keep)
                {
                    indices_to_keep_prior(j) = i;
                    j++; 
                }
                this->N = n_keep_prior; 
                this->input = this->input(indices_to_keep_prior, Eigen::all).eval(); 
                this->points = this->points(indices_to_keep_prior, Eigen::all).eval();
            }
            if (verbose)
            {
                if (!this->simplified)
                {
                    std::cout << "- preserved " << this->N << " points: "
                              << n_interior << " interior, "
                              << this->curr_bound.nv << " boundary"
                              << std::endl; 
                } 
                else 
                {
                    std::cout << "- preserved " << this->N << " points: "
                              << n_interior << " interior, "
                              << this->curr_simplified.nv + n_keep_origbound
                              << " boundary" << std::endl;
                }
            }

            const int D = this->constraints->getD();
            VectorXi to_pull;
            std::vector<Vector_2> normals;

            // If the boundary was not simplified, then pull every point in
            // the boundary
            if (!this->simplified)
            {
                to_pull.resize(this->curr_bound.nv);
                for (int i = 0; i < this->curr_bound.nv; ++i)
                    to_pull(i) = this->curr_bound.vertices[i];
                normals = this->curr_bound.getOutwardVertexNormals();  
            }
            // Otherwise, then pull every point in the *simplified* boundary,
            // plus the desired number of points in the unsimplified boundary 
            else
            {
                const int n_pull = this->curr_simplified.nv + n_pull_origbound;
                std::cout << "n_pull = " << n_pull << std::endl;  
                to_pull.resize(n_pull);

                // The vertices in the simplified boundary are now in one 
                // contiguous chunk in this->input / this->points (see above)
                for (int i = 0; i < this->curr_simplified.nv; ++i)
                    to_pull(i) = n_interior + i;
                std::cout << "added vertices in simplified boundary to to_pull\n"; 
                
                // Choose n_pull_origbound number of vertices among the 
                // vertices in the unsimplified boundary *that were chosen to 
                // be kept above*
                std::vector<int> idx = sampleWithoutReplacement(
                    n_keep_origbound, n_pull_origbound, this->rng
                );
                for (int i = 0; i < n_pull_origbound; ++i)
                    to_pull(this->curr_simplified.nv + i) = n_interior + this->curr_simplified.nv + idx[i];
                std::cout << "added vertices in unsimplified boundary to to_pull\n"; 

                // Rely on the old indexing of points to locate each vertex 
                // to be pulled in the current unsimplified and simplified
                // boundaries and determine its outward normal vector
                std::vector<Vector_2> normals_simplified = this->curr_simplified.getOutwardVertexNormals(); 
                for (auto&& v : normals_simplified)
                    normals.push_back(v);
                std::cout << "found outward normal vectors for vertices in simplified boundary\n"; 
                std::vector<Vector_2> normals_origbound; 
                for (int i = 0; i < n_pull_origbound; ++i)
                {
                    int q = origbound_indices_to_keep[idx[i]];
                    std::vector<int>::iterator qit = std::find(
                        this->curr_bound.vertices.begin(), this->curr_bound.vertices.end(), q
                    );
                    std::vector<int>::iterator pit = std::prev(qit); 
                    std::vector<int>::iterator rit = std::next(qit);  
                    normals.push_back(this->curr_bound.getOutwardVertexNormal(*pit, q, *rit)); 
                    std::cout << ".. found outward normal vector for " << i << "-th vertex in unsimplified boundary\n";  
                }
                std::cout << "found outward normal vectors for vertices in unsimplified boundary\n"; 
            }
            if (verbose)
            {
                std::cout << "- pulling " << to_pull.size()
                          << " boundary points" << std::endl; 
            } 

            // For each vertex in the boundary, pull along its outward normal
            // vector by distance epsilon
            MatrixXd pulled(to_pull.size(), 2); 
            for (int i = 0; i < to_pull.size(); ++i)
            {
                Vector_2 v(this->points(to_pull(i), 0), this->points(to_pull(i), 1));
                Vector_2 v_pulled = v + epsilon * normals[i];
                pulled(i, 0) = CGAL::to_double(v_pulled.x());
                pulled(i, 1) = CGAL::to_double(v_pulled.y());

                // Check that the pulled point is not subject to filtering
                if (filter(pulled.row(i)))
                {
                    // If so, simply don't pull that vertex
                    pulled.row(i) = this->points.row(to_pull(i));
                }
            }

            // For each vertex in the boundary, minimize the distance to the
            // pulled vertex with a feasible parameter point
            MatrixXd pull_results_in(to_pull.size(), D);
            MatrixXd pull_results_out(to_pull.size(), 2); 
            for (int i = 0; i < to_pull.size(); ++i)
            {
                // Minimize the appropriate objective function
                VectorXd target = pulled.row(i);
                auto obj = [this, target](const Ref<const VectorXd>& x)
                {
                    return (target - this->func(x)).squaredNorm();
                };
                VectorXd x_init = this->input.row(to_pull(i));
                VectorXd l_init = VectorXd::Ones(this->constraints->getN())
                    - this->constraints->active(x_init.cast<mpq_rational>()).template cast<double>();
                VectorXd q = optimizer->run(
                    obj, x_init, l_init, tau, delta, beta, max_iter, sqp_tol,
                    sqp_tol, BFGS, regularize, regularize_weight, use_only_armijo,
                    use_strong_wolfe, hessian_modify_max_iter, c1, c2, sqp_verbose
                );
                pull_results_in.row(i) = q;
                pull_results_out.row(i) = this->func(q);
            }  
            
            // Now check proximity of the resulting points (in output 2-D space)
            // to the existing points
            Matrix<bool, Dynamic, 1> added = Matrix<bool, Dynamic, 1>::Zero(to_pull.size());  
            for (int i = 0; i < to_pull.size(); ++i)
            {
                double mindist = (this->points.rowwise() - pull_results_out.row(i)).rowwise().norm().minCoeff();

                // If the resulting point is not filtered and is far enough
                // away from the points already in the point-set, then add to
                // the point-set 
                if (!filter(pull_results_out.row(i)) && mindist > MINDIST_BETWEEN_POINTS)
                {
                    added(i) = true; 
                    this->N++;
                    this->input.conservativeResize(this->N, D); 
                    this->points.conservativeResize(this->N, 2);
                    this->input.row(this->N-1) = pull_results_in.row(i);
                    this->points.row(this->N-1) = pull_results_out.row(i);
                }
            }
            if (verbose)
            {
                std::cout << "- pulling complete; augmented point-set contains "
                          << this->N << " points" << std::endl;
            }

            // Write the results of the pulling procedure to file, if desired 
            if (write_pulled_points)
            {
                std::ofstream outfile; 
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << "-pull.txt";
                outfile.open(ss.str());
                outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                outfile << "START_X\tSTART_Y\tNORMAL_X\tNORMAL_Y\tTARGET_X\tTARGET_Y\t"; 
                for (int i = 0; i < D; ++i)
                    outfile << "RESULT_IN_" << i << '\t'; 
                outfile << "RESULT_OUT_X\tRESULT_OUT_Y\tADDED\n"; 
                for (int i = 0; i < to_pull.size(); ++i)
                {
                    outfile << this->points(to_pull(i), 0) << '\t'
                            << this->points(to_pull(i), 1) << '\t'
                            << CGAL::to_double(normals[i].x()) << '\t'
                            << CGAL::to_double(normals[i].y()) << '\t'
                            << pulled(i, 0) << '\t'
                            << pulled(i, 1) << '\t';
                    for (int j = 0; j < D; ++j)
                        outfile << pull_results_in(i, j) << '\t'; 
                    outfile << pull_results_out(i, 0) << '\t'
                            << pull_results_out(i, 1) << '\t'
                            << added(i) << std::endl; 
                }
                outfile.close();  
            }

            // Get new boundary (assuming that the shape is simply connected)
            std::vector<double> x, y; 
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);
            Boundary2D boundary(x, y);
            try
            {
                // This line may throw:
                // - CGAL::Assertion_exception (while instantiating the alpha shape) 
                // - std::runtime_error (if polygon is not simple)
                this->curr_bound = boundary.getSimplyConnectedBoundary<true>(verbose);
            }
            catch (CGAL::Assertion_exception& e) 
            {
                // Try with tag == false
                //
                // This may throw (another) CGAL::Assertion_exception (while
                // instantiating the alpha shape) 
                try 
                {
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose);
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
                    this->curr_bound = boundary.getSimplyConnectedBoundary<false>(verbose);
                }
                catch (CGAL::Assertion_exception& e)
                {
                    throw; 
                }
            }

            // Re-orient the points so that the boundary is traversed clockwise
            this->curr_bound.orient(CGAL::RIGHT_TURN);

            // Update input parameters as necessary
            n_interior = this->curr_bound.np - this->curr_bound.nv;   // New number of interior points 
            if (n_keep_interior > n_interior)
                n_keep_interior = n_interior; 

            // Remove as many points from the interior as desired
            boundary_indices.clear(); 
            boundary_indices.insert(
                this->curr_bound.vertices.begin(), this->curr_bound.vertices.end()
            );
            std::vector<int> interior_indices; 
            for (int i = 0; i < this->curr_bound.np; ++i)
            {
                if (boundary_indices.find(i) == boundary_indices.end())
                    interior_indices.push_back(i); 
            }
            std::vector<int> interior_indices_to_delete;
            int n_to_delete = this->curr_bound.np - this->curr_bound.nv - n_keep_interior;
            if (n_to_delete < 0)
            {   // Check that the number of points to be deleted is not negative 
                n_to_delete = 0; 
            }
            std::vector<int> idx = sampleWithoutReplacement(
                this->curr_bound.np - this->curr_bound.nv, n_to_delete, this->rng
            );
            for (const int i : idx)
                interior_indices_to_delete.push_back(interior_indices[i]);  
            this->curr_bound.deleteInteriorPoints(interior_indices_to_delete);

            // Update this->N, this->input, this->points accordingly 
            std::vector<int> indices_to_keep; 
            std::unordered_set<int> interior_indices_to_delete_set(
                interior_indices_to_delete.begin(), interior_indices_to_delete.end()
            ); 
            for (int i = 0; i < this->N; ++i)
            {
                if (interior_indices_to_delete_set.find(i) == interior_indices_to_delete_set.end())
                    indices_to_keep.push_back(i); 
            }
            this->input = this->input(indices_to_keep, Eigen::all).eval(); 
            this->points = this->points(indices_to_keep, Eigen::all).eval(); 
            this->N = this->curr_bound.np;
            if (verbose)
            {
                std::cout << "- removed " << interior_indices_to_delete.size()
                          << " interior points" << std::endl;
            } 
            
            // If desired, simplify the current boundary
            if (max_edges > 0 && this->curr_bound.edges.size() > max_edges)
            {
                try
                { 
                    // This line may raise a CGAL::Precondition_exception while 
                    // instantiating the polygon for simplification
                    this->curr_simplified = simplifyAlphaShape(this->curr_bound, max_edges, verbose);  
                }
                catch (CGAL::Precondition_exception& e)
                {
                    // TODO Perhaps a better option exists here
                    throw; 
                }
                // Re-orient the points so that the boundary is traversed 
                // clockwise
                this->curr_simplified.orient(CGAL::RIGHT_TURN); 
                this->simplified = true; 
            }

            // Write boundary information to file if desired
            if (write_prefix.compare(""))
            {
                // Write the boundary points, vertices, and edges 
                std::ofstream outfile; 
                std::stringstream ss;
                ss << write_prefix << "-pass" << iter << ".txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (int i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (int j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close();
                ss.clear(); 
                ss.str(std::string()); 

                // Also write the simplified boundary to file, if simplification
                // was performed ... 
                if (this->simplified)
                {
                    // Write the boundary points, vertices, and edges 
                    ss << write_prefix << "-pass" << iter << "-simplified.txt";
                    this->curr_simplified.write(ss.str());

                    // Write the input vectors passed into the given function to
                    // yield the boundary points
                    outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                    if (outfile.is_open())
                    {
                        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                        for (int i = 0; i < this->input.rows(); ++i)
                        {
                            outfile << "INPUT\t"; 
                            for (int j = 0; j < this->input.cols() - 1; ++j)
                                outfile << this->input(i, j) << '\t'; 
                            outfile << this->input(i, this->input.cols() - 1) << std::endl;
                        }
                    }
                    outfile.close(); 
                }
            }

            // Compute enclosed area and test for convergence
            double change = (
                this->simplified
                ? this->curr_simplified.area - this->curr_area
                : this->curr_bound.area - this->curr_area
            );
            if (verbose)
            {
                std::cout << "[PULL] Iteration " << iter << "; "
                          << this->curr_bound.nv << " boundary points; " 
                          << this->curr_bound.np << " total points; "
                          << "enclosed area: " << this->curr_bound.area << "; "
                          << "change: " << this->curr_bound.area - this->curr_area
                          << std::endl;
                if (this->simplified)
                {
                    std::cout << ">>>>>> Simplified to "
                              << this->curr_simplified.nv
                              << " boundary points; enclosed area: "
                              << this->curr_simplified.area << "; "
                              << "change: " << this->curr_simplified.area - this->curr_area
                              << std::endl;
                }
            }
            bool converged = (std::abs(change) < this->area_tol * this->curr_area); 
            this->curr_area = (
                this->simplified ? this->curr_simplified.area : this->curr_bound.area
            ); 

            return converged; 
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
         *                                in the boundary. If zero, the boundary
         *                                is kept unsimplified.
         * @param n_keep_interior         Number of interior points to keep in
         *                                the unsimplified boundary. If this
         *                                number exceeds the total number of
         *                                interior points, then all interior
         *                                points are kept. 
         * @param n_keep_origbound        If the boundary is simplified, the
         *                                number of points in the *original
         *                                (unsimplified) boundary* that should
         *                                be randomly sampled (without replacement)
         *                                to be preserved.
         * @param n_mutate_origbound      If the boundary is simplified, the
         *                                number of points *among those chosen
         *                                to be preserved* that should be further
         *                                randomly sampled (without replacement)
         *                                to be mutated. 
         * @param n_pull_origbound        If the boundary is simplified, the
         *                                number of points *among those chosen
         *                                to be preserved* that should be further
         *                                randomly sampled (without replacement)
         *                                to be pulled. 
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
         * @param write_pulled_points     If true, write the points that were 
         *                                pulled, their corresponding normal 
         *                                vectors, and the resulting points 
         *                                to file. 
         */
        void run(const double mutate_delta,
                 std::function<bool(const Ref<const VectorXd>&)> filter, 
                 const Ref<const MatrixXd>& init_input, 
                 const int min_step_iter, const int max_step_iter,
                 const int min_pull_iter, const int max_pull_iter,
                 const int sqp_max_iter, const double sqp_tol,
                 const int max_edges, int n_keep_interior, int n_keep_origbound,
                 int n_mutate_origbound, int n_pull_origbound,
                 const double tau, const double delta, const double beta,
                 const bool use_only_armijo, const bool use_strong_wolfe,
                 const int hessian_modify_max_iter,
                 const std::string write_prefix,
                 const RegularizationMethod regularize = NOREG, 
                 const double regularize_weight = 0, const double c1 = 1e-4,
                 const double c2 = 0.9, const bool verbose = true,
                 const bool sqp_verbose = false, 
                 const bool write_pulled_points = false)
        {
            // Initialize the sampling run ...
            this->initialize(
                filter, init_input, max_edges, n_keep_interior, write_prefix,
                verbose
            );

            // ... then step through the boundary-finding algorithm up to the
            // maximum number of iterations ...
            int i = 1;
            bool terminate = false;
            int n_converged = 0;
            boost::random::uniform_real_distribution<double> dist(-mutate_delta, mutate_delta);  
            while (i - 1 < min_step_iter || (i - 1 < max_step_iter && !terminate))
            {
                bool result = this->step(
                    dist, filter, i, max_edges, n_keep_interior, n_keep_origbound,
                    n_mutate_origbound, write_prefix, verbose
                );
                if (!result)
                    n_converged = 0;
                else
                    n_converged++; 
                terminate = (n_converged >= NUM_CONSECUTIVE_ITERATIONS_SATISFYING_TOLERANCE_FOR_CONVERGENCE);
                i++;
            }

            // ... then turn to pulling the boundary points outward
            int j = 0;
            terminate = false;
            n_converged = 0;
            double epsilon = 0.1 * std::sqrt(this->curr_area);
            SQPOptimizer<double>* optimizer = new SQPOptimizer<double>(this->constraints); 
            while (j < min_pull_iter || (j < max_pull_iter && !terminate))
            {
                if (verbose)
                    std::cout << "- pulling by epsilon = " << epsilon << std::endl;  
                bool result = this->pull(
                    optimizer, filter, epsilon, sqp_max_iter, sqp_tol, i + j,
                    max_edges, n_keep_interior, n_keep_origbound, n_pull_origbound,
                    tau, delta, beta, use_only_armijo, use_strong_wolfe,
                    hessian_modify_max_iter, write_prefix, regularize,
                    regularize_weight, c1, c2, verbose, sqp_verbose,
                    write_pulled_points
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
                std::ofstream outfile; 
                std::stringstream ss;
                ss << write_prefix << "-final.txt";
                this->curr_bound.write(ss.str());

                // Write the input vectors passed into the given function to
                // yield the boundary points
                outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                if (outfile.is_open())
                {
                    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                    for (int i = 0; i < this->input.rows(); ++i)
                    {
                        outfile << "INPUT\t"; 
                        for (int j = 0; j < this->input.cols() - 1; ++j)
                            outfile << this->input(i, j) << '\t'; 
                        outfile << this->input(i, this->input.cols() - 1) << std::endl;
                    }
                }
                outfile.close();
                ss.clear(); 
                ss.str(std::string()); 

                // Also write the simplified boundary to file, if simplification
                // was performed ... 
                if (this->simplified)
                {
                    // Write the boundary points, vertices, and edges 
                    ss << write_prefix << "-final-simplified.txt";
                    this->curr_simplified.write(ss.str());

                    // Write the input vectors passed into the given function to
                    // yield the boundary points
                    outfile.open(ss.str(), std::ofstream::out | std::ofstream::app);
                    if (outfile.is_open())
                    {
                        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
                        for (int i = 0; i < this->input.rows(); ++i)
                        {
                            outfile << "INPUT\t"; 
                            for (int j = 0; j < this->input.cols() - 1; ++j)
                                outfile << this->input(i, j) << '\t'; 
                            outfile << this->input(i, this->input.cols() - 1) << std::endl;
                        }
                    }
                    outfile.close(); 
                }
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
