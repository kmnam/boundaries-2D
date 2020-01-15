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
#include <CGAL/Delaunay_triangulation.h>
#include "simplex.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     1/14/2020
 */
using namespace Eigen;
typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag> Kd;
typedef CGAL::Delaunay_triangulation<Kd>           Delaunay_triangulation;
typedef Delaunay_triangulation::Full_cell_handle   Full_cell_handle;
typedef Delaunay_triangulation::Facet              Facet;
typedef Delaunay_triangulation::Vertex_handle      Vertex_handle;
typedef Delaunay_triangulation::Vertex             Vertex;
typedef Delaunay_triangulation::Point              Point;

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

std::vector<Simplex> getSimplices(Delaunay_triangulation dt, unsigned dim)
{
    /*
     * Given a Delaunay triangulation, return a vector of Simplex objects
     * for each of the simplices in the triangulation.  
     */
    // Run through the (finite) simplices in the triangulation
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
    std::vector<Simplex> simplices;
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
        simplices.emplace_back(Simplex(facet));
    }
    return simplices;
}

#endif 
