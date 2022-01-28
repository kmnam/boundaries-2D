/**
 * An implementation of boundary-finding algorithms in the plane. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     1/27/2022
 */

#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
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
#include <CGAL/exceptions.h>
#include <CGAL/tags.h>
#include <Eigen/Sparse>

using namespace Eigen; 

// CGAL convenience typedefs, adapted from the CGAL docs
typedef CGAL::Exact_predicates_inexact_constructions_kernel               K;
typedef K::FT                                                             FT;
typedef K::Point_2                                                        Point_2;
typedef K::Vector_2                                                       Vector_2;
typedef CGAL::Aff_transformation_2<K>                                     Transformation;
typedef CGAL::Orientation                                                 Orientation;
typedef CGAL::Polygon_2<K>                                                Polygon_2;
typedef CGAL::Polyline_simplification_2::Squared_distance_cost            Cost;
typedef CGAL::Polyline_simplification_2::Stop_below_count_threshold       Stop;

struct Grid2DProperties
{
    /*
     * A struct that stores the indices of the points within the grid-based
     * boundary, along with the grid's meshsize.
     */
    public:
        unsigned n;
        std::vector<unsigned> vertices;
        double meshsize;

        Grid2DProperties(unsigned n, std::vector<unsigned> vertices, 
                         double meshsize)
        {
            /* 
             * Trivial constructor.
             */
            this->n = n; 
            this->vertices = vertices;
            this->meshsize = meshsize;
        }

        ~Grid2DProperties()
        {
            /*
             * Empty destructor.
             */
        }

        void write(std::vector<double> x, std::vector<double> y, std::string filename)
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
            std::ofstream outfile;
            outfile.open(filename);
            outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
            outfile << "MESHSIZE\t" << this->meshsize << std::endl;
            for (unsigned i = 0; i < x.size(); i++)
                outfile << "POINT\t" << x[i] << "\t" << y[i] << std::endl;
            for (auto&& v : this->vertices)
                outfile << "VERTEX\t" << v << std::endl;
            outfile.close();
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
            const double two_pi = 2 * acos(-1);

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
        bool is_simple_cycle;
        unsigned max_edges;
        unsigned min;               // Index of point with minimum y-coordinate
        Orientation orientation;    // Orientation of edges

        AlphaShape2DProperties()
        {
            /**
             * Trivial constructor. 
             */
        }

        AlphaShape2DProperties(std::vector<double> x, std::vector<double> y,
                               std::vector<unsigned> vertices,
                               std::vector<std::pair<unsigned, unsigned> > edges,
                               double alpha, double area, bool is_simple_cycle,
                               unsigned max_edges, bool check_order = false)
        {
            /**
             * Constructor with input alpha shape. 
             *
             * Setting `check_order = true` enforces an explicit check that
             * the boundary is a simple cycle and the vertices and edges are
             * specified "in order," as in `edges[0]` lies between `vertices[0]`
             * and `vertices[1]`, `edges[1]` between `vertices[1]` and 
             * `vertices[2]`, and so on.
             */
            // The number of x- and y-coordinates should be the same  
            if (!(x.size() == y.size() && y.size() >= vertices.size()))
                throw std::invalid_argument("Invalid dimensions for input points");

            // If the boundary is assumed to be a simple cycle, then the 
            // number of vertices and edges should be the same
            if (is_simple_cycle && vertices.size() != edges.size())
                throw std::invalid_argument(
                    "Simple-cycle boundary should have the same number of vertices and edges"
                ); 
             
            this->x = x;
            this->y = y;
            this->vertices = vertices;
            this->edges = edges;
            this->np = x.size();
            this->nv = vertices.size();
            this->alpha = alpha;
            this->area = area;
            this->is_simple_cycle = is_simple_cycle; 
            this->max_edges = max_edges;

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
            // *given that the boundary is a simple cycle*
            if (this->is_simple_cycle && check_order)
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
                    throw std::invalid_argument(
                        "Vertices and edges were not specified in order in given simple-cycle boundary"
                    );
            }

            // Find the orientation of the edges
            Point_2 p, q, r;
            unsigned nv = this->vertices.size();
            if (this->min == 0)
            {
                p = Point_2(this->x[this->vertices[this->nv-1]], this->y[this->vertices[this->nv-1]]);
                q = Point_2(this->x[this->vertices[0]], this->y[this->vertices[0]]);
                r = Point_2(this->x[this->vertices[1]], this->y[this->vertices[1]]);
            }
            else
            {
                p = Point_2(
                    this->x[this->vertices[(this->min-1) % this->nv]],
                    this->y[this->vertices[(this->min-1) % this->nv]]
                );
                q = Point_2(
                    this->x[this->vertices[this->min]],
                    this->y[this->vertices[this->min]]
                );
                r = Point_2(
                    this->x[this->vertices[(this->min+1) % this->nv]],
                    this->y[this->vertices[(this->min+1) % this->nv]]
                );
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

        void write(std::string filename)
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
            std::ofstream outfile;
            outfile.open(filename);
            outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
            outfile << "ALPHA\t" << this->alpha << std::endl;
            outfile << "AREA\t" << this->area << std::endl;
            for (unsigned i = 0; i < this->np; ++i)
                outfile << "POINT\t" << this->x[i] << "\t" << this->y[i] << std::endl;
            for (auto&& v : this->vertices)
                outfile << "VERTEX\t" << v << std::endl;
            for (auto&& e : this->edges)
                outfile << "EDGE\t" << e.first << "\t" << e.second << std::endl;
            outfile.close();
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
        int n;                    // Number of stored points

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
            this->n = x_.size(); 
            if (this->n != y_.size())
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
            const int X_INVALID_INDEX = this->x.size();
            const int Y_INVALID_INDEX = this->y.size();  

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

            // Keep track of the indices of the maximum and minimum values of
            // x and y within each row and column of the grid
            std::vector<int> ymin_per_col, ymax_per_col, xmin_per_row, xmax_per_row;
            for (unsigned i = 0; i < nmesh_x - 1; i++)
            {
                ymin_per_col.push_back(Y_INVALID_INDEX);
                ymax_per_col.push_back(Y_INVALID_INDEX);
            }
            for (unsigned i = 0; i < nmesh_y - 1; i++)
            {
                xmin_per_row.push_back(X_INVALID_INDEX);
                xmax_per_row.push_back(X_INVALID_INDEX);
            }
            
            // For each of the stored points, find the row/column in the
            // grid that contains it and update its max/min value accordingly
            for (unsigned i = 0; i < this->n; i++)
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
                if (ymin_per_col[col] == Y_INVALID_INDEX || this->y[ymin_per_col[col]] > yval)
                    ymin_per_col[col] = i;
                if (ymax_per_col[col] == Y_INVALID_INDEX || this->y[ymax_per_col[col]] < yval)
                    ymax_per_col[col] = i;
                if (xmin_per_row[row] == X_INVALID_INDEX || this->x[xmin_per_row[row]] > xval)
                    xmin_per_row[row] = i;
                if (xmax_per_row[row] == X_INVALID_INDEX || this->x[xmax_per_row[row]] < xval)
                    xmax_per_row[row] = i;
            }
            for (unsigned i = 0; i < nmesh_x - 1; i++)
                assert(
                    (ymin_per_col[i] == Y_INVALID_INDEX && ymax_per_col[i] == Y_INVALID_INDEX) ||
                    (ymin_per_col[i] != Y_INVALID_INDEX && ymax_per_col[i] != Y_INVALID_INDEX)
                );
            for (unsigned i = 0; i < nmesh_y - 1; i++)
                assert(
                    (xmin_per_row[i] == X_INVALID_INDEX && xmax_per_row[i] == X_INVALID_INDEX) ||
                    (xmin_per_row[i] != X_INVALID_INDEX && xmax_per_row[i] != X_INVALID_INDEX)
                );

            // Run through the max/min values per column and row and remove 
            // all invalid indices (which indicate empty columns/rows)
            auto empty_col = std::find(ymin_per_col.begin(), ymin_per_col.end(), Y_INVALID_INDEX);
            while (empty_col != ymin_per_col.end())
            {
                ymin_per_col.erase(empty_col);
                nmesh_x--;
                empty_col = std::find(ymin_per_col.begin(), ymin_per_col.end(), Y_INVALID_INDEX);
            }
            empty_col = std::find(ymax_per_col.begin(), ymax_per_col.end(), Y_INVALID_INDEX);
            while (empty_col != ymax_per_col.end())
            {
                ymax_per_col.erase(empty_col);
                empty_col = std::find(ymax_per_col.begin(), ymax_per_col.end(), Y_INVALID_INDEX);
            }
            auto empty_row = std::find(xmin_per_row.begin(), xmin_per_row.end(), X_INVALID_INDEX);
            while (empty_row != xmin_per_row.end())
            {
                xmin_per_row.erase(empty_row);
                nmesh_y--;
                empty_row = std::find(xmin_per_row.begin(), xmin_per_row.end(), X_INVALID_INDEX);
            }
            empty_row = std::find(xmax_per_row.begin(), xmax_per_row.end(), X_INVALID_INDEX);
            while (empty_row != xmax_per_row.end())
            {
                xmax_per_row.erase(empty_row);
                empty_row = std::find(xmax_per_row.begin(), xmax_per_row.end(), X_INVALID_INDEX);
            }

            // Collect the boundary vertices and edges in order (removing 
            // duplicate vertices)
            std::unordered_set<unsigned> vertex_set;
            for (auto&& v : xmin_per_row) vertex_set.insert(v);
            for (auto&& v : ymin_per_col) vertex_set.insert(v);
            for (auto&& v : xmax_per_row) vertex_set.insert(v);
            for (auto&& v : ymax_per_col) vertex_set.insert(v);
            std::vector<unsigned> vertices;
            for (auto&& v : vertex_set) vertices.push_back(v);
            
            return Grid2DProperties(this->n, vertices, meshsize);
        }

        template <bool tag = true>
        AlphaShape2DProperties getBoundary(bool connected = true, bool simply_connected = false,
                                           unsigned max_edges = 0)
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
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            // Print warning if tag == false 
            if (!tag)
                std::cout << "[WARN] Computing alpha shape with tag == false\n";

            // Instantiate a vector of Point objects, keeping track of the 
            // smallest and largest x- and y-coordinate values 
            std::vector<Point_2> points;
            double xmin = std::numeric_limits<double>::infinity(); 
            double ymin = std::numeric_limits<double>::infinity();
            double xmax = -std::numeric_limits<double>::infinity();
            double ymax = -std::numeric_limits<double>::infinity();
            for (unsigned i = 0; i < this->n; ++i)
            {
                double xi = this->x[i]; 
                double yi = this->y[i]; 
                points.emplace_back(Point_2(xi, yi)); 
                if (xmin > xi) xmin = xi; 
                if (xmax < xi) xmax = xi; 
                if (ymin > yi) ymin = yi; 
                if (ymax < yi) ymax = yi;
            }
            double maxdist = std::sqrt(std::pow(xmax - xmin, 2) + std::pow(ymax - ymin, 2));

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

            // Establish an ordering of the vertices in the alpha shape 
            // (this varies as we change alpha) 
            std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
            std::vector<Vertex_handle_2> indices_to_vertices;
            unsigned nvertices = 0;

            // Set up a sparse adjacency matrix for the alpha shape 
            // (this varies as we change alpha) 
            SparseMatrix<int, RowMajor> adj;
            unsigned nedges = 0;

            // Set up a dictionary that maps each vertex in the alpha shape 
            // to the corresponding point in the stored input set 
            std::unordered_map<Vertex_handle_2, std::pair<unsigned, double> > vertices_to_points;

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
            unsigned opt_alpha_index = 0;
            const unsigned INVALID_ALPHA_INDEX = shape.number_of_alphas(); 
            if (simply_connected)
            {
                // -------------------------------------------------------------- // 
                //         IF A SIMPLY CONNECTED BOUNDARY IS DESIRED ...          //
                // -------------------------------------------------------------- //
                
                // Set the lower value of alpha for which the region is connected
                unsigned low = 0; 
                unsigned high; 
                unsigned last_valid = INVALID_ALPHA_INDEX;
                unsigned mid = INVALID_ALPHA_INDEX; 

                // Find the least value of alpha that is greater than the lower bound 
                // on the inter-point distance found through the minimum and maximum
                // x- and y-coordinate values 
                high = std::distance(shape.alpha_begin(), shape.alpha_lower_bound(maxdist));
                std::cout << "- searching between alpha = "
                          << CGAL::to_double(shape.get_nth_alpha(low))
                          << " and "
                          << CGAL::to_double(shape.get_nth_alpha(high)) << std::endl;

                // Try to find a value of alpha for which the boundary consists of
                // more than one connected component
                //
                // Alpha_shape_2::find_optimal_alpha() can throw an Assertion_exception
                // if there are collinear points within the alpha shape
                try
                {
                    low = std::distance(shape.alpha_begin(), shape.find_optimal_alpha(2));
                } 
                catch (CGAL::Assertion_exception& e)
                {
                    low = 0;
                }

                // If low >= high, decrease low to high - 1
                if (low >= high) 
                    low = high - 1; 

                // Also keep track of the vertices and edges in the order in which 
                // they are traversed 
                std::vector<int> traversal;

                // For each larger value of alpha, test that the boundary is 
                // a simple cycle (every vertex has only two incident edges,
                // and traveling along the cycle in one direction gets us 
                // back to the starting point)
                while (low <= high)
                {
                    mid = (low + high) / 2;
                    auto alpha = shape.get_nth_alpha(mid);  
                    shape.set_alpha(alpha);
                    std::cout << "- setting alpha = " << CGAL::to_double(alpha) << std::endl; 

                    // Establish an ordering for the vertices in the alpha shape
                    vertices_to_indices.clear();
                    indices_to_vertices.clear();  
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
                    VectorXi ones = VectorXi::Ones(nvertices); 
                    nedges = (adj.triangularView<Upper>() * ones).sum(); 

                    // Starting from an arbitrary vertex, identify the incident 
                    // edges on the vertex and "travel" along the boundary,
                    // checking that:
                    // 1) Each vertex has exactly two incident edges
                    // 2) Each vertex has one unvisited neighbor and one visited 
                    //    neighbor (except at the start and end of the traversal)
                    // 3) Iteratively choosing the visited neighbor returns us 
                    //    to the starting vertex after *every* vertex has been 
                    //    visited
                    // 
                    // Start from the zeroth vertex ...
                    traversal.clear(); 
                    int curr = 0;
                    Matrix<int, Dynamic, 1> visited = Matrix<int, Dynamic, 1>::Zero(nvertices);
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

                        // If both vertices have been visited *and* one of them 
                        // is the zeroth vertex, then we have returned
                        if (visited(first) == 1 && visited(second) == 1 && (first == 0 || second == 0))
                            returned = true;  
                        // Otherwise, if both vertices have been visited, then 
                        // the alpha shape contains a more complicated structure
                        else if (visited(first) == 1 && visited(second) == 1)
                            break; 
                        // Otherwise, if only the first vertex has been visited, 
                        // then jump to the second vertex
                        else if (visited(first) == 1 && visited(second) == 0)
                            curr = second;
                        // Otherwise, if only the second vertex has been visited, 
                        // then jump to the first vertex
                        else if (visited(second) == 1 && visited(first) == 0)
                            curr = first;
                        // Otherwise, if we are at the zeroth vertex (we are just
                        // starting our traversal), then choose either vertex
                        else if (visited(first) == 0 && visited(second) == 0 && curr == 0)
                            curr = first; 
                        // Otherwise, if neither vertex has been visited, then 
                        // there is something wrong (this is supposed to be 
                        // impossible)
                        else 
                            throw std::runtime_error("This is not supposed to happen!");  
                    }

                    // Have we traversed the entire alpha shape and returned to 
                    // the starting vertex? 
                    int nvisited = visited.sum(); 
                    if (nvisited == nvertices && returned)
                    {
                        std::cout << "- ... traversed " << nvisited << "/" << nvertices
                                  << " boundary vertices in a simple cycle" << std::endl;
                        last_valid = mid;
                        high = mid - 1;  
                    }
                    // Otherwise, if the number of edges is *greater than* the 
                    // number of vertices, then the alpha shape is too detailed
                    // and so we need to lower the value of alpha 
                    else if (nedges > nvertices)
                    {
                        std::cout << "- ... traversed " << nvisited << "/" << nvertices 
                                  << " boundary vertices; boundary contains "
                                  << nedges << " edges" << std::endl; 
                        high = mid - 1; 
                    }
                    // Otherwise, if the number of edges is *lower than* the 
                    // number of vertices, then the alpha shape is not detailed 
                    // enough and so we need to increase the value of alpha
                    else 
                    {
                        std::cout << "- ... traversed " << nvisited << "/" << nvertices
                                  << " boundary vertices; boundary contains "
                                  << nedges << " edges" << std::endl; 
                        low = mid + 1; 
                    }
                }
                bool is_simple_cycle = (last_valid != INVALID_ALPHA_INDEX);
                if (is_simple_cycle)
                    opt_alpha_index = last_valid; 
                else 
                    opt_alpha_index = mid; 
                opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
                shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

                // Re-compute the alpha shape for the last value of alpha for 
                // which the alpha shape is a simple cycle, if necessary  
                if (is_simple_cycle && mid != last_valid)
                {
                    // Establish an ordering for the vertices in the alpha shape
                    vertices_to_indices.clear();
                    indices_to_vertices.clear();  
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
                    VectorXi ones = VectorXi::Ones(nvertices); 
                    nedges = (adj.triangularView<Upper>() * ones).sum(); 

                    // Starting from an arbitrary vertex, identify the incident 
                    // edges on the vertex and "travel" along the boundary,
                    // checking that:
                    // 1) Each vertex has exactly two incident edges
                    // 2) Each vertex has one unvisited neighbor and one visited 
                    //    neighbor (except at the start and end of the traversal)
                    // 3) Iteratively choosing the visited neighbor returns us 
                    //    to the starting vertex after *every* vertex has been 
                    //    visited
                    // 
                    // Start from the zeroth vertex ...
                    traversal.clear(); 
                    int curr = 0;
                    Matrix<int, Dynamic, 1> visited = Matrix<int, Dynamic, 1>::Zero(nvertices);
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

                        // If both vertices have been visited *and* one of them 
                        // is the zeroth vertex, then we have returned
                        if (visited(first) == 1 && visited(second) == 1 && (first == 0 || second == 0))
                            returned = true;  
                        // Otherwise, if only the first vertex has been visited, 
                        // then jump to the second vertex
                        else if (visited(first) == 1 && visited(second) == 0)
                            curr = second;
                        // Otherwise, if only the second vertex has been visited, 
                        // then jump to the first vertex
                        else if (visited(second) == 1 && visited(first) == 0)
                            curr = first;
                        // Otherwise, if we are at the zeroth vertex (we are just
                        // starting our traversal), then choose either vertex
                        else if (visited(first) == 0 && visited(second) == 0 && curr == 0)
                            curr = first; 
                        // Otherwise, if neither vertex has been visited or both 
                        // vertices have been visited, then the alpha shape contains
                        // a more complicated structure (this is supposed to be
                        // impossible)
                        else
                            throw std::runtime_error("This is not supposed to happen!");  
                    }
                }

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
                    for (unsigned i = 0; i < this->n; ++i)
                    {
                        double sqdist = std::pow(this->x[i] - x, 2) + std::pow(this->y[i] - y, 2);
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

                // Define vectors that specify the boundary vertices in the 
                // order in which they were traversed, *in terms of their 
                // indices in the stored input set*, as well as the edges 
                // in the order in which they were traversed   
                std::vector<unsigned> vertex_indices_in_order; 
                std::vector<std::pair<unsigned, unsigned> > edge_indices_in_order; 

                /* ------------------------------------------------------------------ //
                 * If the detected boundary is a simple cycle and simplification is desired, 
                 * simplify the alpha shape using Dyken et al.'s polyline simplification
                 * algorithm:
                 * - The cost of decimating a vertex is measured using maximum distance
                 *   between the remaining vertices and the new line segment formed
                 * - The simplification is terminated once the total cost of decimation
                 *   exceeds 1e-5
                 * ------------------------------------------------------------------ */
                if (is_simple_cycle && max_edges != 0 && nedges > max_edges)
                {
                    std::cout << "- ... simplifying" << std::endl; 
                    
                    // Instantiate a Polygon object with the given vertex order
                    std::vector<Point_2> traversed_points;
                    for (auto it = traversal.begin(); it != traversal.end(); ++it)
                    {
                        int nearest_index = vertices_to_points[indices_to_vertices[*it]].first;  
                        traversed_points.push_back(points[nearest_index]);
                    } 
                    Polygon_2 polygon(traversed_points.begin(), traversed_points.end()); 
                    if (!polygon.is_simple())
                        throw std::runtime_error("Polygon is not simple");

                    // Simplify the Polygon object
                    polygon = CGAL::Polyline_simplification_2::simplify(polygon, Cost(), Stop(max_edges));

                    // Collect the vertices and edges of the simplified Polygon object
                    for (auto it = polygon.vertices_begin(); it != polygon.vertices_end(); ++it)
                    {
                        // Find the vertex in the alpha shape
                        Point_2 p = *it;
                        auto p_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [p, points](std::pair<Vertex_handle_2, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((p - points[i]).squared_length()) <= sqnorm); 
                            }
                        );
                        vertex_indices_in_order.push_back(p_it->second.first);
                    }
                    for (auto it = polygon.edges_begin(); it != polygon.edges_end(); ++it)
                    {
                        // Find the source and target vertices in the alpha shape
                        Point_2 source = it->source();
                        Point_2 target = it->target();
                        auto source_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [source, points](std::pair<Vertex_handle_2, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((source - points[i]).squared_length()) <= sqnorm);
                            }
                        );
                        auto target_it = std::find_if(
                            vertices_to_points.begin(), vertices_to_points.end(),
                            [target, points](std::pair<Vertex_handle_2, std::pair<unsigned, double> > v)
                            {
                                unsigned i = v.second.first;
                                double sqnorm = v.second.second;
                                return (CGAL::to_double((target - points[i]).squared_length()) <= sqnorm);
                            }
                        );
                        edge_indices_in_order.push_back(
                            std::make_pair(source_it->second.first, target_it->second.first)
                        );
                    }
                    nvertices = vertex_indices_in_order.size(); 
                    nedges = edge_indices_in_order.size(); 

                    // Compute the area of the simplified Polygon
                    double total_area = abs(CGAL::to_double(polygon.area()));
                    std::cout << "- optimal value of alpha = " << opt_alpha << std::endl;
                    std::cout << "- enclosed area = " << total_area << std::endl;  
                    std::cout << "- number of vertices = " << nvertices << std::endl; 
                    std::cout << "- number of edges = " << nedges << std::endl; 
                    
                    return AlphaShape2DProperties(
                        x, y, vertex_indices_in_order, edge_indices_in_order,  
                        opt_alpha, total_area, is_simple_cycle, max_edges
                    );
                }
                // If the boundary is a simple cycle but was *not* simplified,
                // then accumulate the indices of the boundary vertices in the
                // order in which they were traversed
                else if (is_simple_cycle)
                {
                    auto it = traversal.begin();
                    unsigned curr = vertices_to_points[indices_to_vertices[*it]].first; 
                    vertex_indices_in_order.push_back(curr);
                    ++it;
                    while (it != traversal.end())
                    {
                        unsigned next = vertices_to_points[indices_to_vertices[*it]].first;
                        edge_indices_in_order.emplace_back(std::make_pair(curr, next));
                        curr = next;  
                        vertex_indices_in_order.push_back(curr);
                        ++it;  
                    }
                    edge_indices_in_order.emplace_back(
                        std::make_pair(curr, vertices_to_points[indices_to_vertices[*(traversal.begin())]].first)
                    );
                }
                // Otherwise, accumulate the indices of the boundary vertices 
                // in arbitrary order 
                else
                {
                    for (unsigned i = 0; i < nvertices; ++i) 
                    {
                        Vertex_handle_2 v = indices_to_vertices[i];
                        unsigned curr = vertices_to_points[v].first;  // Index of vertex *among all stored input points*
                        vertex_indices_in_order.push_back(curr); 
                        for (SparseMatrix<int, RowMajor>::InnerIterator row_it(adj, i); row_it; ++row_it)
                        {
                            unsigned j = row_it.col();
                            Vertex_handle_2 w = indices_to_vertices[j];  
                            unsigned target = vertices_to_points[w].first; 
                            if (curr < target)
                                edge_indices_in_order.emplace_back(std::make_pair(curr, target));
                        } 
                    }
                }

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

                return AlphaShape2DProperties(
                    x, y, vertex_indices_in_order, edge_indices_in_order,  
                    opt_alpha, total_area, is_simple_cycle, max_edges
                );
            }
            else if (connected)
            {
                opt_alpha_index = static_cast<unsigned>(shape.find_optimal_alpha(1) - shape.alpha_begin());
                opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
                shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

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
                    for (unsigned i = 0; i < this->n; ++i)
                    {
                        double sqdist = std::pow(this->x[i] - x, 2) + std::pow(this->y[i] - y, 2);
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
            }
            else
            {
                // Begin with the smallest value of alpha 
                auto alpha_it = shape.alpha_begin();
                unsigned i = 0; 

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
                        Vertex_handle_2 v = Tds::Vertex_range::s_iterator_to(*itv);
                        auto type = shape.classify(v);
                        if (type != Alpha_shape::REGULAR && type != Alpha_shape::INTERIOR)
                        {
                            regular = false;
                            break;
                        }
                    }
                    if (regular)
                    {
                        opt_alpha = CGAL::to_double(*it);
                        opt_alpha_index = i;
                        break;
                    }

                    i++; 
                }
                shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

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
                    for (unsigned i = 0; i < this->n; ++i)
                    {
                        double sqdist = std::pow(this->x[i] - x, 2) + std::pow(this->y[i] - y, 2);
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
            }

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
            std::cout << "- enclosed area = " << total_area << std::endl;  
            std::cout << "- number of vertices = " << nvertices << std::endl; 
            std::cout << "- number of edges = " << nedges << std::endl; 

            // Finally collect the boundary vertices and edges in arbitrary order 
            std::vector<std::pair<unsigned, unsigned> > edges;
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                Face_handle_2 f = it->first;
                int i = it->second;
                Vertex_handle_2 s = f->vertex(f->cw(i));
                Vertex_handle_2 t = f->vertex(f->ccw(i));
                edges.emplace_back(std::make_pair(vertices_to_points[s].first, vertices_to_points[t].first));
            }
            std::vector<unsigned> vertices;
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
                vertices.push_back(vertices_to_points[*it].first);

            return AlphaShape2DProperties(
                x, y, vertices, edges, opt_alpha, total_area, false, max_edges
            );
        }
};

#endif
