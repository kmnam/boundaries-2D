#ifndef BOUNDARY_MAPPER_HPP
#define BOUNDARY_MAPPER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "boundaries.hpp"
#include "simplex.hpp"
#include "triangulate.hpp"

/*
 * An implementation of a "boundary-mapping" algorithm in the plane. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     1/14/2020
 */
using namespace Eigen;

template <typename T>
class BoundaryMapper 
{
    /*
     * A class that computes the boundary of a (bounded) 2-D region
     * as the continuous image of the boundary of an input convex polytope.
     */
    private:
        unsigned D;          // Number of parameters
        unsigned N;          // Number of points

        // Matrix of sampled parameters
        MatrixXd params;

        // Matrix of points in 2-D space
        MatrixX2d points;

        // Vectors containing indices of the boundary points and edges 
        std::vector<unsigned> vertices;
        std::vector<std::pair<unsigned, unsigned> > edges;

        // boost::random::mt19937 random number generator
        boost::random::mt19937 rng;

    public:
        BoundaryMapper(boost::random::mt19937& rng)
        {
            /*
             * Straightforward constructor.
             */
            this->D = 0;
            this->N = 0;
            this->params = MatrixXd::Zero(0, 0);
            this->rng = rng;
        }

        ~BoundaryMapper()
        {
            /*
             * Empty destructor. 
             */
        }

        MatrixXd getParams()
        {
            /*
             * Get the stored matrix of parameter values.
             */
            return this->params; 
        }

        void run(std::function<Matrix<T, Dynamic, 1>(const Ref<const Matrix<T, Dynamic, 1> >&)> func
                 const std::string vertices_file, const unsigned n_within,
                 const unsigned n_bound, const unsigned max_edges = -1,
                 const std::string write_file = "")
        {
            /*
             * Run the boundary-mapping algorithm. 
             */
            // Parse the vertices from the given .vert file 
            std::vector<std::vector<double> > vertices = parseVertices(vertices_file);

            // Triangulate the parsed polytope 
            std::pair<Delaunay_triangulation, unsigned> result = triangulate(vertices);
            Delaunay_triangulation dt = result.first;
            this->D = result.second;

            // Get the simplices of the triangulation
            std::vector<Simplex> interior = getSimplices(dt, this->D);

            // Get the facets of the convex hull
            std::vector<Simplex> boundary = getConvexHullFacets(dt, this->D);

            // Sample the given number of points from within the triangulation
            // and from its boundary
            MatrixXd params_within = sampleFromSimplices(interior, n_within);
            MatrixXd params_bound = sampleFromSimplices(boundary, n_bound);
            this->N = n_within + n_bound;
            this->params = MatrixXd::Zero(this->N, this->D);
            this->params.block(0, 0, n_within, this->D) = params_within;
            this->params.block(n_within, 0, n_bound, this->D) = params_bound;
            this->points = MatrixXd::Zero(this->N, 2);

            for (unsigned i = 0; i < params.rows(); ++i)
            {
                // Evaluate the given function at each parameter point 
                Matrix<T, Dynamic, 1> y = func(this->params.row(i).cast<T>());
                this->points.row(i) = y.template cast<double>();
            }

            // Store the output points in vectors
            std::vector<double> x, y;
            x.resize(this->N);
            y.resize(this->N);
            VectorXd::Map(&x[0], this->N) = this->points.col(0);
            VectorXd::Map(&y[0], this->N) = this->points.col(1);

            // Get boundary of the points in 2-D space
            Boundary2D boundary(x, y);
            AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, max_edges);
            this->vertices = bound_data.vertices;

            // Write boundary information to file if desired
            if (write_file.compare("")) bound_data.write(write_file);
        }
};

#endif
