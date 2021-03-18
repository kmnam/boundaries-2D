#ifndef DELAUNAY_TRIANGULATE_CONVEX_HULL_HPP
#define DELAUNAY_TRIANGULATE_CONVEX_HULL_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <CGAL/Epick_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Origin.h>
#include <CGAL/Kernel_d/Point_d.h>
#include <CGAL/Kernel_d/Vector_d.h>
#include <CGAL/Kernel_d/Hyperplane_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include "simplex.hpp"

/*
 * Functions for computing Delaunay triangulations in n dimensions.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/17/2021
 */
using namespace Eigen;
typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag>          Kd;
typedef CGAL::Delaunay_triangulation<Kd>                    Delaunay_triangulation;
typedef Delaunay_triangulation::Full_cell_handle            Full_cell_handle;
typedef Delaunay_triangulation::Facet                       Facet;
typedef Delaunay_triangulation::Vertex_handle               Vertex_handle;
typedef Delaunay_triangulation::Vertex                      Vertex;
typedef Delaunay_triangulation::Point                       Point;

std::vector<std::vector<double> > parseVertices(std::string vertices_file)
{
    /*
     * Parse the vertices given in a .vert file. 
     */
    // Vector of vertex coordinates
    std::vector<std::vector<double> > vertices;

    // Parse the vertices of the input polytope
    std::string line;
    std::ifstream infile(vertices_file);
    if (!infile.is_open()) 
    {
        std::cerr << "File not found" << std::endl;
        throw std::exception();
    }
    while (std::getline(infile, line))
    {
        // Each vertex is specified as a space-delimited line
        std::istringstream iss(line);
        std::vector<double> vertex;
        std::string token;
        while (std::getline(iss, token, ' '))
            vertex.push_back(std::stod(token));
        vertices.push_back(vertex);
    }
    infile.close();

    return vertices;
}

std::pair<Delaunay_triangulation, unsigned> triangulate(std::vector<std::vector<double> > vertices)
{
    /*
     * Given a vector of vertices, compute its Delaunay triangulation. 
     */
    unsigned dim = vertices[0].size();
    Delaunay_triangulation dt(dim);
    std::vector<Point> points;
    for (unsigned i = 0; i < vertices.size(); ++i)
    {
        Point p(vertices[i].begin(), vertices[i].end());
        points.push_back(p);
    }
    for (auto it = points.begin(); it != points.end(); ++it) dt.insert(*it);

    return std::make_pair(dt, dim); 
}

std::vector<Simplex> getInteriorSimplices(Delaunay_triangulation dt, unsigned dim)
{
    /*
     * Given a Delaunay triangulation, return a vector of Simplex objects
     * for each of the finite simplices in the triangulation.  
     */
    // Run through the finite simplices in the triangulation
    std::vector<Simplex> simplices;
    for (auto it = dt.finite_full_cells_begin(); it != dt.finite_full_cells_end(); ++it)
    {
        // Write the vertices of each simplex to a matrix
        MatrixXd vertices(dim + 1, dim);
        for (unsigned i = 0; i < dim + 1; ++i)
        {
            Point p = it->vertex(i)->point();
            for (unsigned j = 0; j < p.dimension(); ++j) vertices(i,j) = p[j];
        }
        simplices.emplace_back(Simplex(vertices));
    }
    return simplices; 
}

std::vector<Simplex> getConvexHullFacets(Delaunay_triangulation dt, unsigned dim)
{
    /*
     * Given a Delaunay triangulation, return the vector of facets of the
     * covered convex hull.  
     */
    // Run through the facets of the covered convex hull
    std::vector<Simplex> boundary_simplices;
    for (auto it = dt.full_cells_begin(); it != dt.full_cells_end(); ++it)
    {
        if (!dt.is_infinite(it)) continue;
        Facet ft(it, it->index(dt.infinite_vertex()));

        // Get the full cell containing the facet and its co-vertex
        Full_cell_handle c = dt.full_cell(ft);
        int j = dt.index_of_covertex(ft);

        // Write the facet coordinates to a matrix 
        MatrixXd facet(dim, dim);
        unsigned k = 0;
        for (unsigned i = 0; i < dim + 1; ++i)
        {
            if (i != j)
            {
                Point p = c->vertex(i)->point();
                for (unsigned l = 0; l < p.dimension(); ++l) facet(k,l) = p[l];
                k++;
            }
        }
        boundary_simplices.emplace_back(Simplex(facet));
    }
    return boundary_simplices;
}

std::pair<std::vector<Simplex>, std::vector<Simplex> > classifySimplices(std::vector<Simplex> simplices,
                                                                         LinearConstraints constraints,
                                                                         unsigned dim, boost::random::mt19937& rng,
                                                                         const double epsilon = 1e-8,
                                                                         const bool verbose = true)
{
    /*
     * Given a vector of full-dimensional Simplex objects encoding a
     * triangulation of a convex polytope encoded by the given linear
     * constraints, return two vectors, each containing either the 
     * interior or boundary simplices.
     *
     * This is done by taking, for each codimension-one face of each simplex,
     * a randomly sampled point from the face, perturbing it along its normal
     * vector in both directions, and testing if both perturbed vectors lie 
     * within the covering convex polytope. 
     */
    std::vector<Simplex> interior_simplices; 
    std::vector<Simplex> boundary_simplices;

    for (auto&& simplex : simplices)
    {
        bool is_interior = true;
        Simplex face; 
        for (unsigned i = 0; i < simplex.points.rows(); ++i)
        {
            // Instantiate the codimension-one face obtained by removing each vertex 
            MatrixXd A = simplex.points.topRows(i);
            MatrixXd B = simplex.points.bottomRows(simplex.points.rows() - i - 1);
            MatrixXd face_vertices(simplex.points.rows() - 1, simplex.points.cols());
            face_vertices << A, B; 
            face.setPoints(face_vertices);

            // Sample a point on the face 
            VectorXd point = face.sample(1, rng).row(0);

            // Get the normal vector to the face in the ambient space (face 
            // should have codimension one)
            MatrixXd C = face_vertices.topRows(face_vertices.rows() - 1).rowwise() - face_vertices.row(face_vertices.rows() - 1);
            Eigen::JacobiSVD<MatrixXd> svd(C, Eigen::ComputeFullV);
            MatrixXd V = svd.matrixV();
            VectorXd normal = V.col(V.cols() - 1); 

            // Perturb the point by a little bit in either direction along 
            // the normal vector
            VectorXd perturb1 = point + (epsilon * normal / normal.norm()); 
            VectorXd perturb2 = point - (epsilon * normal / normal.norm());

            // Do either one of the perturbations not satisfy the constraints?
            bool eval1 = ((constraints.getA() * perturb1 - constraints.getb()).array() >= 0).all();
            bool eval2 = ((constraints.getA() * perturb2 - constraints.getb()).array() >= 0).all();
            if ((eval1 && !eval2) || (!eval1 && eval2))
            {
                boundary_simplices.push_back(face);
                is_interior = false; 
                break; 
            }
            else if (!eval1 && !eval2)    // Raise an error if neither satisfies the constraints
            {
                throw std::runtime_error("Convex hull facets were not successfully identified"); 
            }
        }

        // If the test was satisfied for every single face of the simplex, 
        // the simplex is interior 
        if (is_interior) interior_simplices.push_back(face);

        // Print current progress (if desired) every 1000 simplices
        if (verbose && (interior_simplices.size() + boundary_simplices.size()) % 1000 == 0)
        {
            std::cout << "- classified " << (interior_simplices.size() + boundary_simplices.size()) << " simplices ("
                      << interior_simplices.size() << " interior, "
                      << boundary_simplices.size() << " boundary)\n";
        }
    }
    return std::make_pair(interior_simplices, boundary_simplices); 
}

#endif 
