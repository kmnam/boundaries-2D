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
 *     1/11/2020
 */
using namespace Eigen;
typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag> K;
typedef CGAL::Delaunay_triangulation<K>            Delaunay_triangulation;
typedef Delaunay_triangulation::Full_cell_handle   Full_cell_handle;
typedef Delaunay_triangulation::Facet              Facet;
typedef Delaunay_triangulation::Vertex_handle      Vertex_handle;
typedef Delaunay_triangulation::Vertex             Vertex;
typedef Delaunay_triangulation::Point              Point;

std::pair<Delaunay_triangulation, unsigned> triangulate(std::string vertices_file)
{
    /*
     * Given a .vert file specifying a convex polytope in terms of its
     * vertices, compute its Delaunay triangulation. 
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

    // Compute the Delaunay triangulation of the vertices
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

std::vector<Simplex> convexHullFacets(std::string vertices_file)
{
    /*
     * Given a .vert file specifying a convex polytope in terms of its
     * vertices, compute its Delaunay triangulation and return the 
     * vector of facets of the covered convex hull.  
     */
    std::pair<Delaunay_triangulation, unsigned> data = triangulate(vertices_file);
    Delaunay_triangulation dt = data.first;
    unsigned dim = data.second; 

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
