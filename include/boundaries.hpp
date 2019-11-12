#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/algorithm.h>

/*
 * An implementation of boundary-finding algorithms in the plane. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/12/2019
 */

// CGAL convenience typedefs, adapted from the CGAL docs
typedef CGAL::Exact_predicates_inexact_constructions_kernel        K;
typedef K::FT                                                      FT;
typedef K::Point_2                                                 Point;
typedef K::Segment_2                                               Segment;
typedef K::Vector_2                                                Vector_2;
typedef CGAL::Aff_transformation_2<K>                              Transformation;
typedef CGAL::Orientation                                          Orientation;
typedef CGAL::Polygon_2<K>                                         Polygon;
typedef CGAL::Alpha_shape_vertex_base_2<K>                         Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                           Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>               Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                     Delaunay_triangulation;
typedef CGAL::Alpha_shape_2<Delaunay_triangulation>                Alpha_shape;
typedef Delaunay_triangulation::Face_handle                        Face_handle;
typedef Delaunay_triangulation::Vertex_handle                      Vertex_handle;
typedef CGAL::Polyline_simplification_2::Squared_distance_cost     Cost;
typedef CGAL::Polyline_simplification_2::Stop_above_cost_threshold Stop;

const double two_pi = 2 * std::acos(-1);

struct Grid2DProperties
{
    /*
     * A struct that stores the indices of the points within the grid-based
     * boundary, along with the grid's meshsize.
     */
    public:
        unsigned npoints;
        std::vector<unsigned> vertices;
        double meshsize;

        Grid2DProperties(unsigned npoints, std::vector<unsigned> vertices, 
                         double meshsize)
        {
            /* 
             * Trivial constructor.
             */
            this->npoints = npoints;
            this->vertices = vertices;
            this->meshsize = meshsize;
        }

        ~Grid2DProperties()
        {
            /*
             * Empty destructor.
             */
        }

        void write(std::vector<double> x, std::vector<double> y, std::string outfile)
        {
            /*
             * Write the boundary information in tab-delimited format, as follows:
             *
             * - The first line contains the meshsize.  
             * - The next block of lines contains the coordinates of the points
             *   in the region.
             * - The next block of lines contains the indices of the vertices
             *   in the boundary.
             */
            std::ofstream of;
            of.open(outfile);
            of << std::setprecision(std::numeric_limits<double>::max_digits10);
            of << "MESHSIZE\t" << this->meshsize << std::endl;
            for (unsigned i = 0; i < x.size(); i++)
                of << "POINT\t" << x[i] << "\t" << y[i] << std::endl;
            for (auto&& v : this->vertices)
                of << "VERTEX\t" << v << std::endl;
            of.close();
        }
};

struct AlphaShape2DProperties
{
    /*
     * A struct that stores the indices of the points within an alpha
     * shape, along with the value of alpha and the area of the enclosed
     * region. 
     */
    private:
        Vector_2 outwardVertexNormal(unsigned p, unsigned q, unsigned r)
        {
            /*
             * Given the indices of three vertices in the boundary, 
             * return the outward normal vector at the middle vertex, q.
             * It is assumed that the boundary contains edges between
             * (p,q) and (q,r). 
             */
            using std::sin;
            using std::cos;
            using std::acos;
            using std::sqrt;

            Vector_2 v, w, normal;
            double angle;
            Orientation v_to_w;
            v = Vector_2(this->x[p] - this->x[q], this->y[p] - this->y[q]);
            w = Vector_2(this->x[r] - this->x[q], this->y[r] - this->y[q]);
            angle = acos(CGAL::scalar_product(v, w) / sqrt(v.squared_length() * w.squared_length()));
            v_to_w = CGAL::orientation(-v, w);

            // Case 1: The boundary is oriented by right turns and -v and w
            // form a right turn
            if (this->orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::RIGHT_TURN)
            {
                // Rotate w by (2*pi - angle) / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, sin((two_pi - angle) / 2.0), cos((two_pi - angle) / 2.0)); 
                normal = rotate(w);
            }
            // Case 2: The boundary is oriented by right turns and -v and w
            // form a left turn
            else if (this->orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::LEFT_TURN)
            {
                // Rotate w by angle / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, sin(angle / 2.0), cos(angle / 2.0));
                normal = rotate(w);
            }
            // Case 3: The boundary is oriented by left turns and -v and w
            // form a right turn
            else if (this->orientation == CGAL::LEFT_TURN && v_to_w == CGAL::RIGHT_TURN)
            {
                // Rotate v by angle / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, sin(angle / 2.0), cos(angle / 2.0));
                normal = rotate(v);
            }
            // Case 4: The boundary is oriented by left turns and -v and w 
            // form a left turn
            else if (this->orientation == CGAL::LEFT_TURN && v_to_w == CGAL::LEFT_TURN)
            {
                // Rotate v by (2*pi - angle) / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, sin((two_pi - angle) / 2.0), cos((two_pi - angle) / 2.0));
                normal = rotate(v);
            }
            // Case 5: -v and w are collinear
            else
            {
                // If the interior of the boundary is to the right, rotate
                // w by 90 degrees counterclockwise
                if (this->orientation == CGAL::RIGHT_TURN)
                {
                    Transformation rotate(CGAL::ROTATION, 1.0, 0.0);
                    normal = rotate(w);
                }
                // Otherwise, rotate v by 90 degrees counterclockwise
                else
                {
                    Transformation rotate(CGAL::ROTATION, 1.0, 0.0);
                    normal = rotate(v);
                }
            }

            // Normalize the vector by its length and return
            normal /= sqrt(normal.squared_length());
            return normal;
        }

    public:
        std::vector<double> x;
        std::vector<double> y;
        std::vector<unsigned> vertices;
        std::vector<std::pair<unsigned, unsigned> > edges;
        unsigned np;
        unsigned nv;
        double alpha;
        double area;
        bool connected;
        bool simply_connected;
        bool simplified;
        unsigned min;               // Index of point with minimum y-coordinate
        Orientation orientation;    // Orientation of edges

        AlphaShape2DProperties(std::vector<double> x, std::vector<double> y,
                               std::vector<unsigned> vertices,
                               std::vector<std::pair<unsigned, unsigned> > edges,
                               double alpha, double area, bool connected, 
                               bool simply_connected, bool simplified,
                               bool check_order = false)
        {
            /*
             * Trivial constructor.
             *
             * check_order == true enforces an explicit check that the vertices
             * and edges are specified "in order," as in edges[0] lies between
             * vertices[0] and vertices[1], edges[1] between vertices[1] and 
             * vertices[2], and so on. If check_order == false, then this 
             * ordering is assumed. 
             */
            if (!(x.size() == y.size() && y.size() >= vertices.size() && vertices.size() == edges.size()))
                throw std::invalid_argument("Invalid dimensions for input arguments");

            this->x = x;
            this->y = y;
            this->vertices = vertices;
            this->edges = edges;
            this->np = x.size();
            this->nv = vertices.size();
            this->alpha = alpha;
            this->area = area;
            this->connected = connected;
            this->simply_connected = simply_connected;
            this->simplified = simplified;

            // Find the vertex with minimum y-coordinate, breaking any 
            // ties with whichever point has the smallest x-coordinate
            this->min = 0;
            double xmin = this->x[this->vertices[0]];
            double ymin = this->y[this->vertices[0]];
            for (unsigned i = 1; i < this->nv; ++i)
            {
                if (this->y[this->vertices[i]] < ymin)
                {
                    this->min = i;
                    xmin = this->x[this->vertices[i]];
                    ymin = this->y[this->vertices[i]];
                }
                else if (this->y[this->vertices[i]] == ymin && this->x[this->vertices[i]] < xmin)
                {
                    this->min = i;
                    xmin = this->x[this->vertices[i]];
                    ymin = this->y[this->vertices[i]];
                }
            }

            // Check that the edges and vertices were specified in order
            if (check_order)
            {
                unsigned i = 0;
                // Check the first vertex first 
                bool ordered = (vertices[0] == edges[0].first && vertices[0] == edges[this->nv-1].second);
                while (ordered && i < this->nv - 1)
                {
                    i++;
                    ordered = (vertices[i] == edges[i-1].second && vertices[i] == edges[i].first);
                }
                if (!ordered)
                    throw std::invalid_argument("Vertices and edges were not specified in order");
            }

            // Find the orientation of the edges
            Point p, q, r;
            unsigned nv = this->vertices.size();
            if (this->min == 0)
            {
                p = Point(this->x[this->vertices[this->nv-1]], this->y[this->vertices[this->nv-1]]);
                q = Point(this->x[this->vertices[0]], this->y[this->vertices[0]]);
                r = Point(this->x[this->vertices[1]], this->y[this->vertices[1]]);
            }
            else
            {
                p = Point(this->x[this->vertices[(this->min-1) % this->nv]], this->y[this->vertices[(this->min-1) % this->nv]]);
                q = Point(this->x[this->vertices[this->min]], this->y[this->vertices[this->min]]);
                r = Point(this->x[this->vertices[(this->min+1) % this->nv]], this->y[this->vertices[(this->min+1) % this->nv]]);
            }
            this->orientation = CGAL::orientation(p, q, r);
        }

        ~AlphaShape2DProperties()
        {
            /*
             * Empty destructor.
             */
        }

        void orient(Orientation orientation)
        {
            /*
             * Re-direct the edges so that they exhibit the given orientation. 
             */
            if (orientation != CGAL::LEFT_TURN && orientation != CGAL::RIGHT_TURN)
                throw std::invalid_argument("Invalid orientation specified");

            // If the given orientation is the opposite of the current orientation ...
            if (orientation != this->orientation)
            {
                std::vector<unsigned> vertices;
                std::vector<std::pair<unsigned, unsigned> > edges;

                vertices.push_back(this->vertices[0]);
                edges.push_back(std::make_pair(this->vertices[0], this->vertices[this->nv-1]));
                for (unsigned i = this->nv - 1; i > 0; --i)
                {
                    vertices.push_back(this->vertices[i]);
                    edges.push_back(std::make_pair(this->vertices[i], this->vertices[i-1]));
                }
                this->vertices = vertices;
                this->edges = edges;
                this->orientation = orientation;
            }
        }

        std::vector<Vector_2> outwardVertexNormals()
        {
            /*
             * Return the outward normal vectors from the vertices in the
             * alpha shape. 
             */
            unsigned p, q, r;
            std::vector<Vector_2> normals;

            // Obtain the outward normal vector at each vertex 
            p = this->vertices[this->nv-1];
            q = this->vertices[0];
            r = this->vertices[1];
            normals.push_back(this->outwardVertexNormal(p, q, r));
            for (unsigned i = 1; i < this->nv; ++i)
            {
                p = this->vertices[(i-1) % this->nv];
                q = this->vertices[i];
                r = this->vertices[(i+1) % this->nv];
                normals.push_back(this->outwardVertexNormal(p, q, r));
            }

            return normals;
        }

        void write(std::string outfile)
        {
            /*
             * Write the boundary information in tab-delimited format, as follows:
             *
             * - The first line contains the value of alpha. 
             * - The second line contains the area of the enclosed region.
             * - The next block of lines contains the coordinates of the points
             *   in the region.
             * - The next block of lines contains the indices of the vertices
             *   in the alpha shape.
             * - The final block of lines contains the indices of the endpoints
             *   of the edges in the alpha shape.
             */
            std::ofstream of;
            of.open(outfile);
            of << std::setprecision(std::numeric_limits<double>::max_digits10);
            of << "ALPHA\t" << this->alpha << std::endl;
            of << "AREA\t" << this->area << std::endl;
            for (unsigned i = 0; i < this->np; ++i)
                of << "POINT\t" << this->x[i] << "\t" << this->y[i] << std::endl;
            for (auto&& v : this->vertices)
                of << "VERTEX\t" << v << std::endl;
            for (auto&& e : this->edges)
                of << "EDGE\t" << e.first << "\t" << e.second << std::endl;
            of.close();
        }
};

class Boundary2D 
{
    /*
     * A class that computes a 2-D boundary from a set of input points
     * in the plane. 
     */
    private:
        std::vector<double> x;    // x-coordinates
        std::vector<double> y;    // y-coordinates

    public:
        Boundary2D()
        {
            /*
             * Empty constructor.
             */
        }

        Boundary2D(std::vector<double> x_, std::vector<double> y_)
        {
            /* 
             * Constructor with input points specified in vectors.
             */
            // Check that x and y have the same number of coordinates
            if (x_.size() != y_.size())
                throw std::invalid_argument("Invalid input vectors");

            this->x = x_;
            this->y = y_;
        }

        ~Boundary2D()
        {
            /*
             * Empty destructor.
             */
        }

        void fromFile(std::string path, char delim = '\t', bool clear = true)
        {
            /*
             * Parse input points from a delimited file. 
             */
            // Check that a file exists at the given path
            std::ifstream f(path);
            if (!f.good())
                throw std::invalid_argument("Specified input file does not exist");

            // Clear all stored points if desired
            if (clear)
            {
                this->x.clear();
                this->y.clear();
            }

            // Parse one line at a time
            std::string line;
            while (std::getline(f, line))
            {
                std::istringstream iss(line);
                double x_, y_;
                char c;
                if (!(iss >> x_ >> c >> y_ && c == delim)) break;
                this->x.push_back(x_);
                this->y.push_back(y_);
            }
        }

        Grid2DProperties getGridBoundary(double meshsize)
        {
            /*
             * Return the coordinates of the points lying along the grid-based
             * boundary of the stored points.
             */
            using std::ceil;
            using std::floor;

            // Divide the two axes into meshes of the given meshsize
            double epsilon = 1e-5;
            double min_x = *std::min_element(this->x.begin(), this->x.end()) - epsilon;
            double min_y = *std::min_element(this->y.begin(), this->y.end()) - epsilon;
            double max_x = *std::max_element(this->x.begin(), this->x.end()) + epsilon;
            double max_y = *std::max_element(this->y.begin(), this->y.end()) + epsilon;
            unsigned nmesh_x = static_cast<unsigned>(ceil((max_x - min_x) / meshsize));
            unsigned nmesh_y = static_cast<unsigned>(ceil((max_y - min_y) / meshsize));
            std::vector<double> mesh_x;
            std::vector<double> mesh_y;
            double curr_x = min_x, curr_y = min_y;
            unsigned i = 0, j = 0;
            while (curr_x < max_x)
            {
                mesh_x.push_back(curr_x);
                curr_x += meshsize;
                i += 1;
            }
            mesh_x.push_back(max_x);
            while (curr_y < max_y)
            {
                mesh_y.push_back(curr_y);
                curr_y += meshsize;
                j += 1;
            }
            mesh_y.push_back(max_y);

            // Keep track of maximum and minimum values of x and y within
            // each row and column of the grid
            std::vector<unsigned> min_per_col, max_per_col, min_per_row, max_per_row;
            for (unsigned i = 0; i < nmesh_x - 1; i++)
            {
                min_per_col.push_back(-1);
                max_per_col.push_back(-1);
            }
            for (unsigned i = 0; i < nmesh_y - 1; i++)
            {
                min_per_row.push_back(-1);
                max_per_row.push_back(-1);
            }
            
            // For each of the stored points, find the row/column in the
            // grid that contains it and update its max/min value accordingly
            unsigned npoints = this->x.size();
            for (unsigned i = 0; i < npoints; i++)
            {
                double xval = this->x[i], yval = this->y[i];
                
                // Find the column to which the point belongs with binary search
                unsigned low = 0, high = nmesh_x - 2;
                unsigned mid = static_cast<unsigned>(floor((low + high) / 2.0));
                while (low < high)
                {
                    if (mesh_x[mid] < xval)          // x-value falls to left of column
                        low = mid + 1;
                    else if (mesh_x[mid+1] >= xval)  // x-value falls to right of column
                        high = mid - 1;
                    else                             // x-value falls within column
                        break;
                    mid = static_cast<unsigned>(floor((low + high) / 2.0));
                }
                unsigned col = mid;

                // Find the row to which the point belongs with binary search
                low = 0; high = nmesh_y - 2;
                mid = static_cast<unsigned>(floor((low + high) / 2.0));
                while (low < high)
                {
                    if (mesh_y[mid] < yval)          // y-value falls below row
                        low = mid + 1;
                    else if (mesh_y[mid+1] >= yval)  // y-value falls above row
                        high = mid - 1;
                    else                             // y-value falls within row
                        break;
                    mid = static_cast<unsigned>(floor((low + high) / 2.0));
                }
                unsigned row = mid;

                // Update max/min values per column and row
                if (min_per_col[col] == -1 || this->y[min_per_col[col]] > yval)
                    min_per_col[col] = i;
                if (max_per_col[col] == -1 || this->y[max_per_col[col]] < yval)
                    max_per_col[col] = i;
                if (min_per_row[row] == -1 || this->x[min_per_row[row]] > xval)
                    min_per_row[row] = i;
                if (max_per_row[row] == -1 || this->x[max_per_row[row]] < xval)
                    max_per_row[row] = i;
            }
            for (unsigned i = 0; i < nmesh_x - 1; i++)
                assert((min_per_col[i] == -1 && max_per_col[i] == -1) || (min_per_col[i] != -1 && max_per_col[i] != -1));
            for (unsigned i = 0; i < nmesh_y - 1; i++)
                assert((min_per_row[i] == -1 && max_per_row[i] == -1) || (min_per_row[i] != -1 && max_per_row[i] != -1));

            // Run through the max/min values per column and row and remove 
            // all occurrences of -1 (which indicate empty columns/rows)
            auto empty_col = std::find(min_per_col.begin(), min_per_col.end(), -1);
            while (empty_col != min_per_col.end())
            {
                min_per_col.erase(empty_col);
                nmesh_x--;
                empty_col = std::find(min_per_col.begin(), min_per_col.end(), -1);
            }
            empty_col = std::find(max_per_col.begin(), max_per_col.end(), -1);
            while (empty_col != max_per_col.end())
            {
                max_per_col.erase(empty_col);
                empty_col = std::find(max_per_col.begin(), max_per_col.end(), -1);
            }
            auto empty_row = std::find(min_per_row.begin(), min_per_row.end(), -1);
            while (empty_row != min_per_row.end())
            {
                min_per_row.erase(empty_row);
                nmesh_y--;
                empty_row = std::find(min_per_row.begin(), min_per_row.end(), -1);
            }
            empty_row = std::find(max_per_row.begin(), max_per_row.end(), -1);
            while (empty_row != max_per_row.end())
            {
                max_per_row.erase(empty_row);
                empty_row = std::find(max_per_row.begin(), max_per_row.end(), -1);
            }

            // Collect the boundary vertices and edges in order
            std::unordered_set<unsigned> vertex_set;
            for (auto&& v : min_per_row) vertex_set.insert(v);
            for (auto&& v : min_per_col) vertex_set.insert(v);
            for (auto&& v : max_per_row) vertex_set.insert(v);
            for (auto&& v : max_per_col) vertex_set.insert(v);
            std::vector<unsigned> vertices;
            for (auto&& v : vertex_set) vertices.push_back(v);
            
            return Grid2DProperties(npoints, vertices, meshsize);
        }

        AlphaShape2DProperties getBoundary(bool connected = true, bool simply_connected = false,
                                           bool simplify = true)
        {
            /*
             * Return an AlphaShape2DProperties object containing the indices 
             * of the points lying in the alpha shape of the stored points, 
             * with the smallest value of alpha such that the indicated 
             * topological properties are satisfied. 
             */
            using std::abs;
            using std::pow;
            using std::floor;

            // Instantiate a vector of Point objects
            std::vector<Point> points;
            unsigned npoints = this->x.size();
            for (unsigned i = 0; i < npoints; ++i)
                points.emplace_back(Point(this->x[i], this->y[i]));

            // Compute the alpha shape from the Delaunay triangulation
            Alpha_shape shape(points.begin(), points.end(), FT(1.0), Alpha_shape::REGULARIZED);

            /* ------------------------------------------------------------------ //
             * Determine the optimal value of alpha that satisfies the following 
             * properties:
             *
             * - Every point in the cloud lies either in the boundary or interior
             *   of the regularized alpha shape (i.e., calling classify() returns
             *   either AlphaShape::REGULAR or AlphaShape::INTERIOR)
             * - If connected is true, the boundary encloses a connected region
             * - If simply_connected is true, the boundary encloses a simply 
             *   connected region (i.e., the boundary consists of one simple 
             *   cycle of vertices). 
             * ------------------------------------------------------------------ */
            double opt_alpha = 0.0;
            if (simply_connected)
            {
                // Begin with the value of alpha for which the region is connected
                unsigned low = shape.find_optimal_alpha(1) - shape.alpha_begin();
                unsigned high = shape.alpha_end() - shape.alpha_begin();

                // For each larger value of alpha, test that the boundary is 
                // a simple cycle (every vertex has only two incident edges,
                // and traveling along the cycle in one direction gets us 
                // back to the starting point)
                while (low <= high)
                {
                    unsigned mid = static_cast<unsigned>(floor((low + high) / 2.0));
                    shape.set_alpha(shape.get_nth_alpha(mid));

                    // Get the number of vertices in the alpha shape
                    unsigned nvertices = 0;
                    for (auto itv = shape.alpha_shape_vertices_begin(); itv != shape.alpha_shape_vertices_end(); ++itv)
                        nvertices++;
                    
                    // Starting from an arbitrary vertex, travel along the edges
                    // spanning the alpha shape, checking that:
                    // 1) Each vertex has exactly two incident edges
                    // 2) Iteratively choosing the unvisited neighbor (traveling
                    //    along the cycle in one direction) at each vertex returns
                    //    us to the starting vertex after visiting every vertex
                    //    in the alpha shape
                    std::unordered_set<Vertex_handle> bound;
                    Vertex_handle curr = *(shape.alpha_shape_vertices_begin());
                    bool traversed = false;
                    while (!traversed)
                    {
                        bound.insert(curr);
                        std::vector<Vertex_handle> neighbors;
                        for (auto ite = shape.alpha_shape_edges_begin(); ite != shape.alpha_shape_edges_end(); ite++)
                        {
                            Face_handle f = ite->first;
                            auto type = shape.classify(f);
                            if (type != Alpha_shape::REGULAR && type != Alpha_shape::INTERIOR)
                                throw std::runtime_error(
                                    "Regular triangulation should not have any singular/exterior faces"
                                );
                            int i = ite->second;
                            Vertex_handle source = f->vertex(f->cw(i));
                            Vertex_handle target = f->vertex(f->ccw(i));
                            if (source == curr || target == curr)
                            {
                                Vertex_handle next = (source == curr ? target : source);
                                neighbors.push_back(next);
                            }
                        }
                        if (neighbors.size() == 2)
                        {
                            if (bound.find(neighbors[0]) == bound.end())
                                curr = neighbors[0];
                            else if (bound.find(neighbors[1]) == bound.end())
                                curr = neighbors[1];
                            else
                                traversed = true;
                        }
                        else break;
                    }
                    if (bound.size() == nvertices)
                        high = mid - 1;
                    else
                        low = mid + 1;
                }
                opt_alpha = shape.get_nth_alpha(low);
            }
            else if (connected)
            {
                opt_alpha = *(shape.find_optimal_alpha(1));
            }
            else
            {
                // Begin with the smallest value of alpha 
                auto alpha_it = shape.alpha_begin();

                // For each larger value of alpha, test that every vertex 
                // in the point cloud falls within either the alpha shape 
                // (i.e., the boundary) or its interior
                for (auto it = alpha_it; it != shape.alpha_end(); it++)
                {
                    shape.set_alpha(*it);

                    // Run through the vertices in the underlying Delaunay
                    // triangulation, and classify each according to its
                    // relationship with the alpha shape
                    bool regular = true;
                    for (auto itv = shape.finite_vertices_begin(); itv != shape.finite_vertices_end(); itv++)
                    {
                        Vertex_handle v = Tds::Vertex_range::s_iterator_to(*itv);
                        auto type = shape.classify(v);
                        if (type != Alpha_shape::REGULAR && type != Alpha_shape::INTERIOR)
                        {
                            regular = false;
                            break;
                        }
                    }
                    if (regular)
                    {
                        opt_alpha = *it;
                        break;
                    }
                }
            }

            // Get the area of the region enclosed by the alpha shape 
            // with the optimum value of alpha
            shape.set_alpha(opt_alpha);
            double total_area = 0.0;
            for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
            {
                Face_handle face = Tds::Face_range::s_iterator_to(*it);
                auto type = shape.classify(face);
                if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
                {
                    Point p = it->vertex(0)->point();
                    Point q = it->vertex(1)->point();
                    Point r = it->vertex(2)->point();
                    total_area += CGAL::area(p, q, r);
                }
            }

            // Identify, for each vertex in the alpha shape, the point corresponding to it 
            std::unordered_map<Vertex_handle, std::pair<unsigned, double> > vertices_to_points;
            for (unsigned i = 0; i < points.size(); ++i)
            {
                // Find the nearest vertex to each point 
                Vertex_handle nearest = shape.nearest_vertex(points[i]);

                // Look for whether the vertex was identified as the nearest 
                // to any previous point
                auto found = vertices_to_points.find(nearest);

                // Update the vertex's nearest point if the point is indeed
                // nearer to the vertex than any previously considered point
                double sqnorm = CGAL::to_double((nearest->point() - points[i]).squared_length());
                if (found == vertices_to_points.end() || sqnorm < (found->second).second)
                {
                    vertices_to_points[nearest] = std::make_pair(i, sqnorm);
                }
            }

            // Count the number of vertices and edges in the alpha shape
            std::unordered_set<unsigned> vertex_set;
            unsigned nvertices = 0;
            unsigned nedges = 0;
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                vertex_set.insert(vertices_to_points[*it].first);
                nvertices++;
            }
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                nedges++;
            }

            // If the alpha shape is simply connected, organize the vertices 
            // and edges so that they are given in order
            if (simply_connected)
            {
                // Run through the edges of the alpha shape in order from an arbitrary
                // vertex, keeping track of their squared lengths
                std::vector<unsigned> vertices_in_order;
                std::vector<std::pair<unsigned, unsigned> > edges_in_order;
                std::vector<Point> points_in_order;
                std::vector<double> edge_lengths;
                std::unordered_map<unsigned, std::unordered_set<unsigned> > visited_edges;
                for (auto&& v : vertex_set) visited_edges[v] = {};
                unsigned source = *(vertex_set.begin());
                while (points_in_order.size() < nvertices)
                {
                    vertices_in_order.push_back(source);
                    points_in_order.push_back(points[source]);
                    auto next = std::find_if(
                        shape.alpha_shape_edges_begin(), shape.alpha_shape_edges_end(),
                        [source, vertices_to_points, visited_edges](std::pair<Face_handle, int> e)
                        {
                            // Find the source and target vertices of each edge
                            Face_handle f = e.first;
                            int i = e.second;
                            Vertex_handle s = f->vertex(f->cw(i));
                            Vertex_handle t = f->vertex(f->ccw(i));
                            unsigned si = (vertices_to_points.find(s)->second).first;
                            unsigned ti = (vertices_to_points.find(t)->second).first;
                            bool visited = (visited_edges.find(si)->second).count(ti);
                            return (si == source && !visited);
                        }
                    );
                    Face_handle f = next->first;
                    int i = next->second;
                    Vertex_handle s = f->vertex(f->cw(i));
                    Vertex_handle t = f->vertex(f->ccw(i));
                    unsigned si = vertices_to_points[s].first;
                    unsigned ti = vertices_to_points[t].first;
                    visited_edges[si].insert(ti);
                    visited_edges[ti].insert(si);
                    edges_in_order.push_back(std::make_pair(si, ti));
                    edge_lengths.push_back((points[si] - points[ti]).squared_length());
                    source = ti;
                }

                // Instantiate a Polygon object with the given vertex order
                Polygon polygon(points_in_order.begin(), points_in_order.end());
                assert(polygon.is_simple());

                /* ------------------------------------------------------------------ //
                 * Simplify the alpha shape using Dyken et al.'s polyline simplification
                 * algorithm:
                 * - The cost of decimating a vertex is measured using maximum distance
                 *   between the remaining vertices and the new line segment formed
                 * - The simplification is terminated once the total cost of decimation
                 *   exceeds 1e-5
                 * ------------------------------------------------------------------ */
                if (simplify)
                {
                    // Sort the squared edge lengths and get their median
                    std::sort(edge_lengths.begin(), edge_lengths.end());
                    double median;
                    if (edge_lengths.size() % 2 == 1)
                    {
                        unsigned median_idx = static_cast<unsigned>(floor(edge_lengths.size() / 2.0));
                        median = edge_lengths[median_idx];
                    }
                    else
                    {
                        unsigned median_idx = static_cast<unsigned>(edge_lengths.size() / 2.0);
                        median = (edge_lengths[median_idx-1] + edge_lengths[median_idx]) / 2.0;
                    }

                    // Simplify the Polygon object
                    polygon = CGAL::Polyline_simplification_2::simplify(polygon, Cost(), Stop(median));

                    // Collect the vertices and edges of the simplified Polygon object
                    vertices_in_order.clear();
                    edges_in_order.clear();
                    for (auto it = polygon.vertices_begin(); it != polygon.vertices_end(); ++it)
                    {
                        // Find the vertex in the alpha shape
                        Point p = *it;
                        auto p_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [p, points](std::pair<Vertex_handle, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((p - points[i]).squared_length()) <= sqnorm); 
                            }
                        );
                        vertices_in_order.push_back(p_it->second.first);
                    }
                    for (auto it = polygon.edges_begin(); it != polygon.edges_end(); ++it)
                    {
                        // Find the source and target vertices in the alpha shape
                        Point source = it->source();
                        Point target = it->target();
                        auto source_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [source, points](std::pair<Vertex_handle, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((source - points[i]).squared_length()) <= sqnorm);
                            }
                        );
                        auto target_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [target, points](std::pair<Vertex_handle, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((target - points[i]).squared_length()) <= sqnorm);
                            }
                        );
                        edges_in_order.push_back(
                            std::make_pair(source_it->second.first, target_it->second.first)
                        );
                    }

                    // Compute the area of the simplified Polygon
                    total_area = abs(CGAL::to_double(polygon.area()));
                }

                return AlphaShape2DProperties(
                    x, y, vertices_in_order, edges_in_order, opt_alpha, total_area,
                    connected, simply_connected, simplify
                );
            }

            // Otherwise, simply collect the vertices and edges in arbitrary order
            std::vector<std::pair<unsigned, unsigned> > edges;
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                Face_handle f = it->first;
                int i = it->second;
                Vertex_handle s = f->vertex(f->cw(i));
                Vertex_handle t = f->vertex(f->ccw(i));
                edges.push_back(std::make_pair(vertices_to_points[s].first, vertices_to_points[t].first));
            }
            std::vector<unsigned> vertices;
            for (auto&& v : vertex_set) vertices.push_back(v);

            return AlphaShape2DProperties(
                x, y, vertices, edges, opt_alpha, total_area, connected,
                simply_connected, simplify
            );
        }
};

#endif
