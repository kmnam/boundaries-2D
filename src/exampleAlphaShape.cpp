#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include "../include/boundaries.hpp"

/*
 * Demonstrate alpha shape computation on a polygonal region.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     5/12/2022
 */
using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K; 
typedef K::FT                                                FT;
typedef K::Point_2                                           Point;
typedef K::Segment_2                                         Segment;
typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>         Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>               Delaunay_triangulation_2;
typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2>        Alpha_shape;
typedef typename Delaunay_triangulation_2::Face_handle       Face_handle_2;
typedef typename Delaunay_triangulation_2::Vertex_handle     Vertex_handle_2;

/** 
 * Traverse the given alpha shape and write its information to the given 
 * output file. 
 */
std::tuple<std::vector<int>, std::vector<std::pair<int, int> >, double>
    getBoundary(Alpha_shape& shape, const Ref<const MatrixXd>& A)
{
    // Establish an ordering of the vertices in the alpha shape 
    // (this varies as we change alpha) 
    std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
    std::vector<Vertex_handle_2> indices_to_vertices;
    int nvertices = 0;

    // Set up a sparse adjacency matrix for the alpha shape 
    // (this varies as we change alpha) 
    SparseMatrix<int, RowMajor> adj;
    int nedges = 0;

    // Set up a dictionary that maps each vertex in the alpha shape 
    // to the corresponding point in the point-set
    std::unordered_map<Vertex_handle_2, std::pair<int, double> > vertices_to_points;

    // Establish an ordering for the vertices in the alpha shape
    nvertices = 0; 
    for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
    {
        vertices_to_indices[*it] = nvertices;
        indices_to_vertices.push_back(*it); 
        nvertices++;
    }

    // Iterate through the edges in the alpha shape and fill in
    // the adjacency matrix 
    adj.resize(nvertices, nvertices); 
    std::vector<Triplet<int> > triplets;
    for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
    {
        // Add the two corresponding entries to the adjacency 
        // matrix 
        int j = it->second; 
        Vertex_handle_2 source = (it->first)->vertex(Delaunay_triangulation_2::cw(j));
        Vertex_handle_2 target = (it->first)->vertex(Delaunay_triangulation_2::ccw(j));
        int source_i = vertices_to_indices[source]; 
        int target_i = vertices_to_indices[target]; 
        triplets.emplace_back(Triplet<int>(source_i, target_i, 1)); 
        triplets.emplace_back(Triplet<int>(target_i, source_i, 1)); 
    }
    adj.setFromTriplets(triplets.begin(), triplets.end());

    // Identify, for each vertex in the alpha shape, the point corresponding to it 
    for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
    {
        // Find the point being pointed to by the boundary vertex 
        Point_2 point = (*it)->point();
        double x = point.x(); 
        double y = point.y(); 
        
        // Find the nearest input point to this point
        double nearest_sqdist = std::numeric_limits<double>::infinity(); 
        int nearest_index = 0; 
        for (int i = 0; i < A.rows(); ++i)
        {
            double sqdist = std::pow(A(i, 0) - x, 2) + std::pow(A(i, 1) - y, 2);
            if (sqdist < nearest_sqdist)
            {
                nearest_sqdist = sqdist;
                nearest_index = i; 
            }
        }
        vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
    }

    // Count the edges in the alpha shape
    VectorXi ones = VectorXi::Ones(nvertices); 
    nedges = (adj.triangularView<Upper>() * ones).sum(); 

    // Get the area of the region enclosed by the alpha shape 
    // with the optimum value of alpha
    double total_area = 0.0;
    for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
    {
        Face_handle_2 face = Tds::Face_range::s_iterator_to(*it);
        auto type = shape.classify(face);
        if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
        {
            Point_2 p = it->vertex(0)->point();
            Point_2 q = it->vertex(1)->point();
            Point_2 r = it->vertex(2)->point();
            total_area += CGAL::area(p, q, r);
        }
    }
    std::cout << "- alpha = " << shape.get_alpha() << std::endl; 
    std::cout << "- enclosed area = " << total_area << std::endl;  
    std::cout << "- number of vertices = " << nvertices << std::endl; 
    std::cout << "- number of edges = " << nedges << std::endl; 

    // Finally collect the boundary vertices and edges in arbitrary order 
    std::vector<std::pair<int, int> > edges; 
    for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
    {
        Face_handle_2 f = it->first;
        int i = it->second;
        Vertex_handle_2 s = f->vertex(f->cw(i));
        Vertex_handle_2 t = f->vertex(f->ccw(i));
        edges.emplace_back(std::make_pair(vertices_to_points[s].first, vertices_to_points[t].first));
    }
    std::vector<int> vertices;
    for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
        vertices.push_back(vertices_to_points[*it].first);

    return std::make_tuple(vertices, edges, total_area); 
}

std::pair<int, bool> traverseAlphaShape(Alpha_shape& shape, 
                                        std::vector<Point_2>& points, 
                                        std::vector<int>& traversal,
                                        std::unordered_map<Vertex_handle_2, int>& vertices_to_indices,
                                        std::vector<Vertex_handle_2>& indices_to_vertices,
                                        std::unordered_map<Vertex_handle_2, std::pair<int, double> >&
                                            vertices_to_points, 
                                        SparseMatrix<int, RowMajor>& adj, 
                                        const int alpha_index, const bool verbose)
{
    auto alpha = shape.get_nth_alpha(alpha_index);  
    shape.set_alpha(alpha);
    if (verbose)
        std::cout << "- setting alpha = " << CGAL::to_double(alpha) << std::endl; 

    // Establish an ordering for the vertices in the alpha shape
    vertices_to_indices.clear();
    indices_to_vertices.clear();  
    int nvertices = 0; 
    for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
    {
        vertices_to_indices[*it] = nvertices;
        indices_to_vertices.push_back(*it);
        nvertices++;
    }
   
    // Iterate through the edges in the alpha shape and fill in the adjacency matrix 
    adj.resize(nvertices, nvertices); 
    std::vector<Triplet<int> > triplets;
    for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
    {
        // Add the two corresponding entries to the adjacency matrix 
        int j = it->second; 
        Vertex_handle_2 source = (it->first)->vertex(Delaunay_triangulation_2::cw(j));
        Vertex_handle_2 target = (it->first)->vertex(Delaunay_triangulation_2::ccw(j));
        int source_i = vertices_to_indices[source]; 
        int target_i = vertices_to_indices[target]; 
        triplets.emplace_back(Triplet<int>(source_i, target_i, 1)); 
        triplets.emplace_back(Triplet<int>(target_i, source_i, 1)); 
    }
    adj.setFromTriplets(triplets.begin(), triplets.end());
    VectorXi ones = VectorXi::Ones(nvertices); 
    int nedges = (adj.triangularView<Upper>() * ones).sum(); 

    // Starting from an arbitrary vertex, identify the incident edges on the
    // vertex and "travel" along the boundary, checking that:
    // 1) Each vertex has exactly two incident edges
    // 2) Each vertex has one unvisited neighbor and one visited neighbor
    //    (except at the start and end of the traversal)
    // 3) Iteratively choosing the visited neighbor returns us to the
    //    starting vertex after *every* vertex has been visited
    // 
    // Start from the zeroth vertex ...
    traversal.clear(); 
    int curr = 0;
    VectorXi visited = VectorXi::Zero(nvertices); 
    bool returned = false;  
    
    while (!returned)
    {
        // Mark the current vertex as having been visited
        traversal.push_back(curr); 
        visited(curr) = 1; 

        // Iterate over the nonzero entries in the current
        // vertex's row
        SparseMatrix<int, RowMajor>::InnerIterator row_it(adj, curr);
        
        if (!row_it) break;    // If the vertex has no neighbor, then break 
        int first = row_it.col();
        ++row_it; 
        if (!row_it) break;    // If the vertex has no second neighbor, then break 
        int second = row_it.col();
        ++row_it;  
        if (row_it)  break;    // If the vertex has more than two neighbors, then break

        // If both vertices have been visited *and* one of them is the zeroth
        // vertex, then we have returned
        if (visited(first) == 1 && visited(second) == 1 && (first == 0 || second == 0))
            returned = true;  
        // Otherwise, if both vertices have been visited, then the alpha shape
        // contains a more complicated structure
        else if (visited(first) == 1 && visited(second) == 1)
            break; 
        // Otherwise, if only the first vertex has been visited, then jump to
        // the second vertex
        else if (visited(first) == 1 && visited(second) == 0)
            curr = second;
        // Otherwise, if only the second vertex has been visited, then jump to
        // the first vertex
        else if (visited(second) == 1 && visited(first) == 0)
            curr = first;
        // Otherwise, if we are at the zeroth vertex (we are just starting our
        // traversal), then choose either vertex
        else if (visited(first) == 0 && visited(second) == 0 && curr == 0)
            curr = first; 
        // Otherwise, if neither vertex has been visited, then there is
        // something wrong (this is supposed to be impossible)
        else 
            throw std::runtime_error("This is not supposed to happen!");  
    }

    // Identify, for each vertex in the alpha shape, the point corresponding to it 
    vertices_to_points.clear(); 
    for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
    {
        // Find the point being pointed to by the boundary vertex 
        Point_2 point = (*it)->point();
        double x = point.x(); 
        double y = point.y(); 
        
        // Find the nearest point in the point-set to this boundary vertex
        double nearest_sqdist = std::numeric_limits<double>::infinity(); 
        int nearest_index = 0; 
        for (int i = 0; i < points.size(); ++i)
        {
            double sqdist = std::pow(points[i].x() - x, 2) + std::pow(points[i].y() - y, 2);
            if (sqdist < nearest_sqdist)
            {
                nearest_sqdist = sqdist;
                nearest_index = i; 
            }
        }
        vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
    }

    // Have we traversed the entire alpha shape and returned to the starting 
    // vertex, and does every vertex lie either along or within the simple cycle? 
    int nvisited = visited.sum(); 
    if (nvisited == nvertices && returned)
    {
        if (verbose)
        { 
            std::cout << "- ... traversed " << nvisited << "/" << nvertices
                      << " boundary vertices in a simple cycle" << std::endl;
        }
        return std::make_pair(nvertices, true); 
    }
    else
    {
        if (verbose)
        {
            std::cout << "- ... traversed " << nvisited << "/" << nvertices 
                      << " boundary vertices; boundary contains "
                      << nedges << " edges" << std::endl;
        }
        return std::make_pair(nvertices, false);  
    }
}

std::tuple<std::vector<int>, std::vector<std::pair<int, int> >, double, double>
    getSimplyConnectedBoundary(Alpha_shape& shape, std::vector<Point_2>& points,
                               int max_edges = 0)
{
    // Establish an ordering of the vertices in the alpha shape 
    // (this varies as we change alpha) 
    std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
    std::vector<Vertex_handle_2> indices_to_vertices;
    int nvertices = 0;

    // Set up a sparse adjacency matrix for the alpha shape 
    // (this varies as we change alpha) 
    SparseMatrix<int, RowMajor> adj;
    int nedges = 0;

    // Set up a dictionary that maps each vertex in the alpha shape 
    // to the corresponding point in the point-set 
    std::unordered_map<Vertex_handle_2, std::pair<int, double> > vertices_to_points;

    /* ------------------------------------------------------------------ //
     * Determine the optimal value of alpha such that the boundary encloses
     * a simply connected region (i.e., the boundary consists of one 
     * simple cycle of vertices) 
     * ------------------------------------------------------------------ */
    double opt_alpha = 0.0;
    int opt_alpha_index = 0;
    const int INVALID_ALPHA_INDEX = shape.number_of_alphas();

    // Set the lower value of alpha for which the region is connected
    int low = 0; 
    int last_valid = INVALID_ALPHA_INDEX;

    // Try to find the smallest value of alpha for which the boundary
    // consists of one connected component
    //
    // Alpha_shape_2::find_optimal_alpha() can throw an Assertion_exception
    // if there are collinear points within the alpha shape
    try
    {
        low = std::distance(shape.alpha_begin(), shape.find_optimal_alpha(1));
    } 
    catch (CGAL::Assertion_exception& e)
    {
        low = 0;
    }
    int high = shape.number_of_alphas() - 1; 
    std::cout << "- ... searching between alpha = "
              << CGAL::to_double(shape.get_nth_alpha(low))
              << " and "
              << CGAL::to_double(shape.get_nth_alpha(high))
              << " inclusive ("
              << high - low + 1
              << " values of alpha)" << std::endl;

    // Also keep track of the vertices and edges in the order in which 
    // they are traversed 
    std::vector<int> traversal;

    // For each larger value of alpha, test that the boundary is 
    // a simple cycle (every vertex has only two incident edges,
    // and traveling along the cycle in one direction gets us 
    // back to the starting point)
    for (int mid = low; mid <= high; ++mid)
    {
        std::pair<int, bool> result = traverseAlphaShape(
            shape, points, traversal, vertices_to_indices,
            indices_to_vertices, vertices_to_points, adj, mid, false
        );
        nvertices = result.first; 
        if (result.second)
        { 
            last_valid = mid; 
            break; 
        }
    }
    
    // Have we found a value of alpha for which the boundary is a
    // simple cycle? 
    bool is_simple_cycle = (last_valid != INVALID_ALPHA_INDEX);
    if (is_simple_cycle)
        opt_alpha_index = last_valid; 
    else
        throw std::runtime_error("Could not find any simple-cycle boundary");  
    opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
    shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

    // Count the edges in the alpha shape
    VectorXi ones = VectorXi::Ones(nvertices); 
    nedges = (adj.triangularView<Upper>() * ones).sum();

    // Define vectors that specify the boundary vertices in the 
    // order in which they were traversed, *in terms of their 
    // indices in the point-set*, as well as the edges in the 
    // order in which they were traversed
    std::vector<int> vertex_indices_in_order; 
    std::vector<std::pair<int, int> > edge_indices_in_order; 

    /* ------------------------------------------------------------------ //
     * If simplification of the detected boundary is desired, then simplify
     * the alpha shape using Dyken et al.'s polyline simplification algorithm:
     * - The cost of decimating a vertex is measured using maximum distance
     *   between the remaining vertices and the new line segment formed
     * - The simplification is terminated once the total cost of decimation
     *   exceeds 1e-5
     * ------------------------------------------------------------------ */
    if (max_edges != 0 && nedges > max_edges)
    {
        std::cout << "- ... simplifying the boundary" << std::endl; 

        // Instantiate a Polygon object with the vertices given in the
        // order in which they were traversed
        std::vector<Point_2> traversed_points;
        for (auto it = traversal.begin(); it != traversal.end(); ++it)
        {
            int nearest_index = vertices_to_points[indices_to_vertices[*it]].first;
            Point_2 p(points[nearest_index].x(), points[nearest_index].y());  
            traversed_points.push_back(p);
        }
        Polygon_2 polygon(traversed_points.begin(), traversed_points.end());
        
        // Simplify the polygon ...  
        Polygon_2 simplified_polygon = CGAL::Polyline_simplification_2::simplify(polygon, Cost(), Stop(max_edges));

        for (auto it = simplified_polygon.vertices_begin(); it != simplified_polygon.vertices_end(); ++it)
        {
            double x = it->x(); 
            double y = it->y(); 

            // ... and identify the index of each vertex in the polygon
            // with respect to the entire point-set 
            double sqdist_to_nearest_point = std::numeric_limits<double>::infinity(); 
            auto it_nearest_point = points.begin(); 
            for (auto it2 = points.begin(); it2 != points.end(); ++it2) 
            {
                double xv = it2->x(); 
                double yv = it2->y(); 
                double sqdist = std::pow(x - xv, 2) + std::pow(y - yv, 2);
                if (sqdist_to_nearest_point > sqdist)
                {
                    sqdist_to_nearest_point = sqdist; 
                    it_nearest_point = it2; 
                }
            } 
            vertex_indices_in_order.push_back(std::distance(points.begin(), it_nearest_point)); 
        }

        // The edges in the polygon are then easy to determine
        for (int i = 0; i < vertex_indices_in_order.size() - 1; ++i)
        {
            int vi = vertex_indices_in_order[i];
            int vj = vertex_indices_in_order[i+1]; 
            edge_indices_in_order.emplace_back(std::make_pair(vi, vj)); 
        }
        int vi = *vertex_indices_in_order.end();
        int vj = *vertex_indices_in_order.begin();
        edge_indices_in_order.emplace_back(std::make_pair(vi, vj)); 
        nvertices = vertex_indices_in_order.size(); 
        nedges = edge_indices_in_order.size(); 

        // Compute the area of the polygon formed by the simplified 
        // boundary
        double total_area = std::abs(CGAL::to_double(polygon.area())); 
        std::cout << "- optimal value of alpha = " << opt_alpha << std::endl;
        std::cout << "- enclosed area = " << total_area << std::endl;  
        std::cout << "- number of vertices = " << nvertices << std::endl; 
        std::cout << "- number of edges = " << nedges << std::endl; 
        
        return std::make_tuple(
            vertex_indices_in_order, edge_indices_in_order, opt_alpha, total_area
        );
    }
    // If the boundary is a simple cycle but was *not* simplified,
    // then accumulate the indices of the boundary vertices in the
    // order in which they were traversed
    else
    {
        auto it = traversal.begin();
        int curr = vertices_to_points[indices_to_vertices[*it]].first; 
        vertex_indices_in_order.push_back(curr);
        ++it;
        while (it != traversal.end())
        {
            int next = vertices_to_points[indices_to_vertices[*it]].first;
            edge_indices_in_order.emplace_back(std::make_pair(curr, next));
            curr = next;  
            vertex_indices_in_order.push_back(curr);
            ++it;  
        }
        edge_indices_in_order.emplace_back(
            std::make_pair(curr, vertices_to_points[indices_to_vertices[*(traversal.begin())]].first)
        );
        
        // Get the area of the region enclosed by the alpha shape 
        // with the optimum value of alpha
        double total_area = 0.0;
        for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
        {
            Face_handle_2 face = Tds::Face_range::s_iterator_to(*it);
            auto type = shape.classify(face);
            if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
            {
                Point_2 p = it->vertex(0)->point();
                Point_2 q = it->vertex(1)->point();
                Point_2 r = it->vertex(2)->point();
                total_area += CGAL::area(p, q, r);
            }
        }
        std::cout << "- optimal value of alpha = " << opt_alpha << std::endl;
        std::cout << "- number of vertices = " << nvertices << std::endl; 
        std::cout << "- number of edges = " << nedges << std::endl;
        std::cout << "- enclosed area = " << total_area << std::endl;  

        return std::make_tuple(
            vertex_indices_in_order, edge_indices_in_order, opt_alpha, total_area
        ); 
    }
}


int main(int argc, char** argv)
{
    std::string filename = "example/figure_eight/figure_eight.tsv";
    std::ifstream infile(filename);
    std::string line;
    int n = 0; 
    MatrixXd A(n + 1, 2); 
    while (std::getline(infile, line))
    {
        std::stringstream ss; 
        std::string token;
        ss << line; 
        std::getline(ss, token, '\t');
        A(n, 0) = std::stod(token); 
        std::getline(ss, token, '\t');
        A(n, 1) = std::stod(token);  
        n++;
        A.conservativeResize(n + 1, 2); 
    }

    // Instantiate a vector of Point objects
    std::vector<double> points_x, points_y;
    std::vector<Point_2> points;
    for (int i = 0; i < n; ++i)
    {
        double xi = A(i, 0); 
        double yi = A(i, 1);
        points_x.push_back(xi);
        points_y.push_back(yi); 
        points.emplace_back(Point_2(xi, yi)); 
    }
    
    // Compute the alpha shape from the Delaunay triangulation
    Alpha_shape shape; 
    try
    {
        shape.make_alpha_shape(points.begin(), points.end());
    }
    catch (CGAL::Assertion_exception& e)
    {
        throw; 
    }
    shape.set_mode(Alpha_shape::REGULARIZED);

    // For each larger value of alpha, test that every vertex in the 
    // point-set lies along either along the alpha shape or within 
    // its interior
    FT alpha; 
    std::tuple<std::vector<int>, std::vector<std::pair<int, int> >, double> bound; 
    for (int alpha_idx = 0; alpha_idx < shape.number_of_alphas(); alpha_idx += 5)
    {
        alpha = CGAL::to_double(shape.get_nth_alpha(alpha_idx));
        shape.set_alpha(alpha);
        bound = getBoundary(shape, A);  
        AlphaShape2DProperties prop(
            points_x, points_y, std::get<0>(bound), std::get<1>(bound), alpha,
            std::get<2>(bound), false
        );
        std::stringstream ss; 
        ss << "example/figure_eight/figure_eight_alpha" << alpha_idx << ".txt";
        prop.write(ss.str()); 
    }

    // Do the same for the very last boundary (with the largest alpha) ...
    alpha = CGAL::to_double(shape.get_nth_alpha(shape.number_of_alphas() - 1));
    shape.set_alpha(alpha);
    bound = getBoundary(shape, A);
    AlphaShape2DProperties prop_final(
        points_x, points_y, std::get<0>(bound), std::get<1>(bound), alpha,
        std::get<2>(bound), false
    );
    std::stringstream ss;
    ss << "example/figure_eight/figure_eight_alpha" << shape.number_of_alphas() - 1 << ".txt";
    prop_final.write(ss.str()); 

    // Now set alpha to the least value for which the boundary is a simple cycle
    std::tuple<std::vector<int>, std::vector<std::pair<int, int> >, double, double> scbound = 
        getSimplyConnectedBoundary(shape, points, 0);
    AlphaShape2DProperties prop_sc(
        points_x, points_y, std::get<0>(scbound), std::get<1>(scbound),
        std::get<2>(scbound), std::get<3>(scbound), true
    );
    ss.clear();
    ss.str(std::string()); 
    ss << "example/figure_eight/figure_eight_alpha_simple.txt";
    prop_sc.write(ss.str()); 

    return 0;
}
