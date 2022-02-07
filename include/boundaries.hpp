/**
 * An implementation of boundary-finding algorithms in the plane. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     2/7/2022
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

using std::sin;
using std::cos;
using std::acos;
using std::sqrt;
using namespace Eigen;
constexpr double TWO_PI = 2 * std::acos(-1);

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef K::FT                                                       FT;
typedef K::Point_2                                                  Point_2;
typedef K::Vector_2                                                 Vector_2;
typedef CGAL::Aff_transformation_2<K>                               Transformation;
typedef CGAL::Orientation                                           Orientation;
typedef CGAL::Polygon_2<K>                                          Polygon_2;
typedef CGAL::Polyline_simplification_2::Squared_distance_cost      Cost;
typedef CGAL::Polyline_simplification_2::Stop_below_count_threshold Stop;

/**
 * A container that stores a planar point-set along with the indices of the 
 * points lying in an alpha shape of the point-set, together with the
 * corresponding value of alpha and the area of the enclosed region. 
 */
struct AlphaShape2DProperties
{
    private:
        /**
         * Given the indices of three *consecutive* vertices `p`, `q`, `r` in
         * the alpha shape, return the outward normal vector at the middle
         * vertex, `q`.
         *
         * The alpha shape is assumed to contain the edges `(p,q)` and `(q,r)`.
         *
         * @param p Index of first vertex in alpha shape.
         * @param q Index of second vertex in alpha shape. 
         * @param r Index of third vertex in alpha shape. 
         * @return Outward normal vector at `q`. 
         */
        Vector_2 outwardVertexNormal(int p, int q, int r)
        {
            Vector_2 v, w, normal;

            // Get the vectors from q to p and from q to r 
            v = Vector_2(this->x[p] - this->x[q], this->y[p] - this->y[q]);
            w = Vector_2(this->x[r] - this->x[q], this->y[r] - this->y[q]);

            // Get the angle between the two vectors 
            double angle = std::acos(CGAL::scalar_product(v, w) / std::sqrt(v.squared_length() * w.squared_length()));
            Orientation v_to_w = CGAL::orientation(-v, w);

            // Case 1: The boundary is oriented by right turns and -v and w
            // form a right turn
            if (this->orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::RIGHT_TURN)
            {
                // Rotate w by (2*pi - angle) / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, std::sin((TWO_PI - angle) / 2.0), std::cos((TWO_PI - angle) / 2.0)); 
                normal = rotate(w);
            }
            // Case 2: The boundary is oriented by right turns and -v and w
            // form a left turn
            else if (this->orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::LEFT_TURN)
            {
                // Rotate w by angle / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, std::sin(angle / 2.0), std::cos(angle / 2.0));
                normal = rotate(w);
            }
            // Case 3: The boundary is oriented by left turns and -v and w
            // form a right turn
            else if (this->orientation == CGAL::LEFT_TURN && v_to_w == CGAL::RIGHT_TURN)
            {
                // Rotate v by angle / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, std::sin(angle / 2.0), std::cos(angle / 2.0));
                normal = rotate(v);
            }
            // Case 4: The boundary is oriented by left turns and -v and w 
            // form a left turn
            else if (this->orientation == CGAL::LEFT_TURN && v_to_w == CGAL::LEFT_TURN)
            {
                // Rotate v by (2*pi - angle) / 2 counterclockwise
                Transformation rotate(CGAL::ROTATION, std::sin((TWO_PI - angle) / 2.0), std::cos((TWO_PI - angle) / 2.0));
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
        // x- and y-coordinates of the points in the point-set 
        std::vector<double> x;
        std::vector<double> y;

        // Vertices and edges of the alpha shape, indicated by the indices
        // of the points in x and y
        std::vector<int> vertices;
        std::vector<std::pair<int, int> > edges;
         
        unsigned np;                // Number of points in the point-set
        unsigned nv;                // Number of vertices in the alpha shape
        double alpha;               // Value of alpha
        double area;                // Enclosed area 
        bool is_simple_cycle;       // Whether the region is a simple cycle 
        unsigned min;               // Index of point with minimum y-coordinate
        Orientation orientation;    // Orientation of edges

        /**
         * Trivial constructor. 
         */
        AlphaShape2DProperties()
        {
        }

        /**
         * Constructor with a non-empty point-set and alpha shape.
         *
         * Setting `is_simple_cycle = true` and `check_order = true` enforces
         * an explicit check that the boundary is a simple cycle and the
         * vertices and edges are specified "in order," as in `edges[0]` lies
         * between `vertices[0]` and `vertices[1]`, `edges[1]` between
         * `vertices[1]` and `vertices[2]`, and so on.
         *
         * @param x               x-coordinates of input point-set. 
         * @param y               y-coordinates of input point-set. 
         * @param vertices        Indices of vertices lying in input alpha shape.
         * @param edges           Pairs of indices of vertices connected by 
         *                        edges in input alpha shape.
         * @param alpha           Corresponding value of alpha. 
         * @param area            Area of region enclosed by the alpha shape. 
         * @param is_simple_cycle If true, the alpha shape consists of one 
         *                        simple cycle of edges.
         * @param check_order     If true (and `is_simple_cycle` is also true), 
         *                        this constructor checks whether the alpha
         *                        shape indeed consists of one simple cycle of 
         *                        edges, and whether the vertices and edges have
         *                        been specified in order, as described above.
         * @throws std::invalid_argument If `x`, `y`, and `vertices` do not all 
         *                               have the same size, or if `is_simple_cycle`
         *                               is true and yet `vertices` and `edges`
         *                               do not have the same size. 
         * @throws std::runtime_error    If `check_order` and `is_simple_cycle`
         *                               are true, but the vertices and edges 
         *                               do not form a simple cycle or have not 
         *                               been specified in order. 
         */
        AlphaShape2DProperties(std::vector<double> x, std::vector<double> y,
                               std::vector<int> vertices,
                               std::vector<std::pair<int, int> > edges, 
                               double alpha, double area, bool is_simple_cycle,
                               bool check_order = false)
        {
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
                int i = 0;
                // Check the first vertex first 
                bool ordered = (vertices[0] == edges[0].first && vertices[0] == edges[this->nv-1].second);
                while (ordered && i < this->nv - 1)
                {
                    i++;
                    ordered = (vertices[i] == edges[i-1].second && vertices[i] == edges[i].first);
                }
                if (!ordered)
                    throw std::runtime_error(
                        "Vertices and edges were not specified in order in given simple-cycle boundary"
                    );
            }

            // Find the orientation of the edges
            Point_2 p, q, r;
            int nv = this->vertices.size();
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
        
        /**
         * Trivial destructor. 
         */
        ~AlphaShape2DProperties()
        {
        }

        /**
         * Re-direct the edges (i.e., change the edge `(p, q)` to `(q, p)`) so
         * that every edge exhibits the given orientation.
         *
         * @param orientation            Desired orientation of edges.
         * @throws std::invalid_argument If the specified orientation is invalid 
         *                               (is not `CGAL::LEFT_TURN` or
         *                               `CGAL::RIGHT_TURN`).  
         */
        void orient(Orientation orientation)
        {
            if (orientation != CGAL::LEFT_TURN && orientation != CGAL::RIGHT_TURN)
                throw std::invalid_argument("Invalid orientation specified");

            // If the given orientation is the opposite of the current orientation ...
            if (orientation != this->orientation)
            {
                std::vector<int> vertices;
                std::vector<std::pair<int, int> > edges; 

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

        /**
         * Return the outward normal vectors from all vertices along the alpha
         * shape.
         *
         * @returns `std::vector` of outward normal vectors.  
         */
        std::vector<Vector_2> outwardVertexNormals()
        {
            int p, q, r;
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

        /**
         * Write the boundary data to an output file with the given name, as
         * follows: 
         * - The first line contains the value of alpha. 
         * - The second line contains the area of the enclosed region.
         * - The next block of lines contains the coordinates of the points
         *   in the region.
         * - The next block of lines contains the indices of the vertices
         *   in the alpha shape.
         * - The final block of lines contains the indices of the endpoints
         *   of the edges in the alpha shape.
         *
         * @param filename Output file name. 
         */
        void write(std::string filename)
        {
            std::ofstream outfile;
            outfile.open(filename);
            outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);

            // Write the value of alpha and the enclosed area 
            outfile << "ALPHA\t" << this->alpha << std::endl;
            outfile << "AREA\t" << this->area << std::endl;

            // Write each point in the full point-set 
            for (unsigned i = 0; i < this->np; ++i)
                outfile << "POINT\t" << this->x[i] << "\t" << this->y[i] << std::endl;

            // Write each vertex and edge in the alpha shape 
            for (auto&& v : this->vertices)
                outfile << "VERTEX\t" << v << std::endl;
            for (auto&& e : this->edges)
                outfile << "EDGE\t" << e.first << "\t" << e.second << std::endl;

            outfile.close();
        }
};

/**
 * A class that, given a set of points in the plane, computes a suitable
 * "boundary" from its family of alpha shapes.  
 */
class Boundary2D 
{
    private:
        std::vector<double> x;    // x-coordinates
        std::vector<double> y;    // y-coordinates
        int n;                    // Number of stored points

    public:
        /**
         * Trivial constructor. 
         */
        Boundary2D()
        {
        }

        /** 
         * Constructor with a non-empty input point-set.
         *
         * @param x x-coordinates of input point-set. 
         * @param y y-coordinates of input point-set.
         * @throws std::invalid_argument If there are different numbers of x- 
         *                               and y-coordinates specified.  
         */
        Boundary2D(std::vector<double> x, std::vector<double> y)
        {
            // Check that x and y have the same number of coordinates
            this->n = x.size(); 
            if (this->n != y.size())
            {
                throw std::invalid_argument(
                    "Invalid input point-set (different numbers of x- and y-coordinates)"
                );
            }
            this->x = x;
            this->y = y;
        }

        /**
         * Trivial destructor. 
         */
        ~Boundary2D()
        {
        }

        /**
         * Parse a set of points in the given delimited .txt file and either 
         * (1) append the parsed points to the stored point-set (`clear = false`)
         * or (2) overwrite the stored point-set with the parsed points
         * (`clear = true`). 
         *
         * @param filename Input file name. 
         * @param delim    Delimiter in the input file.
         * @param clear    If true, overwrite all previously stored points with
         *                 the parsed points; if false, append the parsed points
         *                 to the previously stored points.  
         */
        void fromFile(std::string filename, char delim = '\t', bool clear = true)
        {
            // Check that a file exists at the given path
            std::ifstream f(filename);
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

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, with no topological 
         * assumption regarding the enclosed region approximated by the point-set.
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the *smallest* value of alpha such that every point in the 
         * the point-set falls within the boundary or its interior.
         *
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getBoundary() 
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            using std::abs;
            using std::pow;
            using std::floor;

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
            // to the corresponding point in the point-set
            std::unordered_map<Vertex_handle_2, std::pair<unsigned, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the least value of alpha such that each point in the
             * point-set lies either along the boundary or within the boundary 
             * ------------------------------------------------------------------ */
            double opt_alpha = 0.0;
            unsigned opt_alpha_index = 0;

            // Begin with the smallest value of alpha 
            auto alpha_it = shape.alpha_begin();
            unsigned i = 0; 

            // For each larger value of alpha, test that every vertex in the 
            // point-set lies along either along the alpha shape or within 
            // its interior  
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

            return AlphaShape2DProperties(x, y, vertices, edges, opt_alpha, total_area, false);
        }

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, assuming that the 
         * enclosed region approximated by the point-set is connected. 
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the *smallest* value of alpha such that the boundary encloses
         * a connected region.
         *
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getConnectedBoundary()
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            using std::abs;
            using std::pow;
            using std::floor;

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
            // to the corresponding point in the point-set 
            std::unordered_map<Vertex_handle_2, std::pair<unsigned, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the least value of alpha such that the boundary encloses
             * a connected region (i.e., the alpha complex consists of a single 
             * connected component) 
             * ------------------------------------------------------------------ */
            unsigned opt_alpha_index = std::distance(shape.alpha_begin(), shape.find_optimal_alpha(1));
            double opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
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
                
                // Find the nearest point in the point-set to this boundary vertex
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

            return AlphaShape2DProperties(x, y, vertices, edges, opt_alpha, total_area, false);
        }

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, assuming that the 
         * enclosed region approximated by the point-set is simply connected.  
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the *smallest* value of alpha such that the boundary forms 
         * a simple cycle (i.e., the enclosed region is simply connected).
         *
         * @param max_edges Maximum number of edges to be contained in the 
         *                  alpha shape; if the number of edges in the alpha 
         *                  shape exceeds this value, the alpha shape is
         *                  simplified. 
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getSimplyConnectedBoundary(unsigned max_edges = 0)
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            using std::abs;
            using std::pow;
            using std::floor;

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
            // to the corresponding point in the point-set 
            std::unordered_map<Vertex_handle_2, std::pair<unsigned, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the optimal value of alpha such that the boundary encloses
             * a simply connected region (i.e., the boundary consists of one 
             * simple cycle of vertices) 
             * ------------------------------------------------------------------ */
            double opt_alpha = 0.0;
            unsigned opt_alpha_index = 0;
            const unsigned INVALID_ALPHA_INDEX = shape.number_of_alphas();

            // Set the lower value of alpha for which the region is connected
            unsigned low = 0; 
            unsigned high; 
            unsigned last_valid = INVALID_ALPHA_INDEX;

            // Find the least value of alpha that is greater than the lower bound 
            // on the inter-point distance found through the minimum and maximum
            // x- and y-coordinate values 
            high = std::distance(shape.alpha_begin(), shape.alpha_lower_bound(maxdist));

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
            std::cout << "- searching between alpha = "
                      << CGAL::to_double(shape.get_nth_alpha(low))
                      << " and "
                      << CGAL::to_double(shape.get_nth_alpha(high)) << std::endl;

            // Also keep track of the vertices and edges in the order in which 
            // they are traversed 
            std::vector<int> traversal;

            // For each larger value of alpha, test that the boundary is 
            // a simple cycle (every vertex has only two incident edges,
            // and traveling along the cycle in one direction gets us 
            // back to the starting point)
            for (unsigned mid = low; mid < high; ++mid)
            {
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
                std::cout << nvertices << " " << this->n << std::endl;
                
                // Run through the points in the point-set and classify them 
                // as lying:
                // - along the boundary (REGULAR),
                // - within the boundary (INTERIOR),
                // - along the boundary without touching a face of the
                //   triangulation (SINGULAR), or
                // - away from all the other points (EXTERIOR)
                unsigned nregular = 0;
                unsigned nsingular = 0;
                unsigned ninterior = 0; 
                unsigned nexterior = 0;  
                for (auto it = shape.finite_vertices_begin(); it != shape.finite_vertices_end(); ++it)
                {
                    Vertex_handle_2 v = Tds::Vertex_range::s_iterator_to(*it);
                    auto type = shape.classify(v); 
                    if (type == Alpha_shape::REGULAR)
                        nregular++; 
                    else if (type == Alpha_shape::SINGULAR)
                        nsingular++; 
                    else if (type == Alpha_shape::INTERIOR)
                        ninterior++; 
                    else 
                        nexterior++; 
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
                // the starting vertex, and does every vertex lie either along
                // or within the simple cycle? 
                int nvisited = visited.sum(); 
                if (nvisited == nvertices && returned && this->n == nregular + ninterior)
                {
                    std::cout << "- ... traversed " << nvisited << "/" << nvertices
                              << " boundary vertices in a simple cycle" << std::endl;
                    last_valid = mid; 
                    break; 
                }
                else
                {
                    std::cout << "- ... traversed " << nvisited << "/" << nvertices 
                              << " boundary vertices; boundary contains "
                              << nedges << " edges" << std::endl; 
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

            // Identify, for each vertex in the alpha shape, the point corresponding to it 
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                // Find the point being pointed to by the boundary vertex 
                Point_2 point = (*it)->point();
                double x = point.x(); 
                double y = point.y(); 
                
                // Find the nearest point in the point-set to this boundary vertex
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
            // indices in the point-set*, as well as the edges in the 
            // order in which they were traversed
            std::vector<unsigned> vertex_indices_in_order; 
            std::vector<std::pair<unsigned, unsigned> > edge_indices_in_order; 

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
                if (!polygon.is_simple())
                    throw std::runtime_error("Simple-cycle boundary does not form a simple polygon"); 
                
                // Simplify the polygon: this polygon should be simple   
                Polygon_2 simplified_polygon = CGAL::Polyline_simplification_2::simplify(polygon, Cost(), Stop(max_edges));

                for (auto it = simplified_polygon.vertices_begin(); it != simplified_polygon.vertices_end(); ++it)
                {
                    double x = it->x(); 
                    double y = it->y(); 

                    // ... and identify the index of the vertex in the 
                    // entire point-set 
                    auto itp = std::find_if(
                        vertices_to_points.begin(), vertices_to_points.end(), 
                        [x, y, &points](std::pair<Vertex_handle_2, std::pair<unsigned, double> > v)
                        {
                            unsigned i = v.second.first; 
                            double sqdist = v.second.second; 
                            double xv = points[i].x(); 
                            double yv = points[i].y();
                            return (std::pow(x - xv, 2) + std::pow(y - yv, 2) <= sqdist); 
                        }
                    );
                    vertex_indices_in_order.push_back(itp->second.first);
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
                double total_area = abs(CGAL::to_double(polygon.area())); 
                std::cout << "- optimal value of alpha = " << opt_alpha << std::endl;
                std::cout << "- enclosed area = " << total_area << std::endl;  
                std::cout << "- number of vertices = " << nvertices << std::endl; 
                std::cout << "- number of edges = " << nedges << std::endl; 
                
                return AlphaShape2DProperties(
                    x, y, vertex_indices_in_order, edge_indices_in_order,  
                    opt_alpha, total_area, is_simple_cycle
                );
            }
            // If the boundary is a simple cycle but was *not* simplified,
            // then accumulate the indices of the boundary vertices in the
            // order in which they were traversed
            else
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
                    opt_alpha, total_area, is_simple_cycle
                );
            }
        }
};

#endif
