#ifndef CONVEX_POLYTOPE_SAMPLE_HPP
#define CONVEX_POLYTOPE_SAMPLE_HPP

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <iomanip>
#include <limits>
#include <tuple>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include "simplex.hpp"

/*
 * Functions for random sampling from convex polytopes.
 *
 * Implements two main strategies (and associated helper functions):
 *
 *   1) MCMC sampling through Smith's hit-and-run random walk
 *
 *   2) Delaunay triangulation followed by sampling of simplex with probability
 *      proportional to its volume, followed by sampling from Dirichlet 
 *      distribution on chosen simplex 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/13/2021
 */
using namespace Eigen;

class Polytope
{
    /* 
     * Simple wrapper class for a Euclidean convex polytope. 
     */
    public:
        MatrixXd constraints;    // Half-plane inequality matrix

        unsigned nrows()   { return constraints.rows(); }
        unsigned nparams() { return constraints.cols() - 1; }

        Polytope(const Ref<const MatrixXd>& constraints)
        {
            /*
             * Trivial constructor. 
             */
            this->constraints = constraints; 
        }

        bool contains(const Ref<const VectorXd>& point) const 
        {
            /*
             * Indicator function for the given point. 
             */
            return ((this->constraints.rightCols(this->constraints.cols() - 1) * point).array() >= -this->constraints.col(0).array()).all();
        }
};

MatrixXd parseConstraintsFile(std::string filename)
{
    /*
     * Parse the given .poly file of linear inequalities (half-planes) and
     * return them in an Eigen::MatrixXd. 
     *
     * Each line in the input file should be a space-delimited vector, 
     *
     * a0 a1 a2 ... aN
     *
     * which encodes the inequality a0 + a1*x1 + ... + an*xN >= 0.
     */
    MatrixXd constraints;
    unsigned nrows = 0;
    unsigned ncols = 0;

    std::ifstream infile(filename);
    std::string line; 
    while (std::getline(infile, line, '\n'))
    {
        // Parse each line in the file ... 
        std::istringstream ss(line);

        // If #columns has not been initialized, count the number of columns
        if (ncols == 0)
        {
            std::string a; 
            while (std::getline(ss, a, ' ')) ncols++;
            ss.str(std::string());
            ss.clear();
            ss.str(line); 
        }

        // Now append the row ...
        nrows++;
        constraints.conservativeResize(nrows, ncols);
        unsigned j = 0;
        std::string a; 
        while (std::getline(ss, a, ' '))
        {
            constraints(nrows - 1, j) = std::stod(a);
            j++;
        }
    }
    
    return constraints; 
}

std::tuple<std::vector<std::vector<double> >, std::vector<Simplex>, std::vector<double> >
    parseTriangulation(std::string triangulation_file)
{
    /*
     * Given a .delv file specifying a convex polytope in terms of its 
     * Delaunay triangulation and its vertices, parse its simplices and
     * return them as a vector. 
     */
    // Vector of vertices
    std::vector<std::vector<double> > vertices; 

    // Vector of Simplex objects
    std::vector<Simplex> simplices;

    // Vector of simplex volumes
    std::vector<double> volumes; 

    // Parse the input triangulation file ...
    std::string line;
    std::ifstream infile(triangulation_file);
    unsigned dim = 0;
    std::regex regex;
    std::string pattern;
    if (!infile.is_open()) 
    {
        std::stringstream ss; 
        ss << "File not found: " << triangulation_file; 
        throw std::invalid_argument(ss.str());
    }
    while (std::getline(infile, line))
    {
        // Each vertex is specified as a space-delimited line of floating-point values 
        if (line.compare(0, 1, "{") != 0)
        {
            std::istringstream iss(line);
            std::vector<double> vertex;
            std::string token;
            while (std::getline(iss, token, ' '))
                vertex.push_back(std::stod(token));
            vertices.push_back(vertex);
            if (dim == 0)
            {
                dim = vertex.size();
                // Define a regular expression for subsequent lines
                // in the file specifying the simplices and their volumes
                pattern = "^\\{";
                for (unsigned i = 0; i < dim; i++) pattern = pattern + "([[:digit:]]+) ";
                pattern = pattern + "([[:digit:]]+)\\} ([[:digit:]]+)(\\/[[:digit:]]+)?$";
                regex.assign(pattern);
            }
        }
        // Each simplex is specified as a space-delimited string of  
        // vertex indices, surrounded by braces, followed by its volume
        // as a rational number 
        else
        {
            if (dim == 0)
            {
                throw std::invalid_argument("Vertices of polytope not specified in input file");
            }
            else
            {
                // Match the contents of each line to the regular expression
                std::smatch matches;
                std::vector<unsigned> vertex_indices;
                if (std::regex_match(line, matches, regex))
                {
                    if (matches.size() == dim + 4)
                    {
                        for (unsigned i = 1; i < matches.size() - 2; i++)
                        {
                            std::ssub_match match = matches[i];
                            std::string match_str = match.str();
                            vertex_indices.push_back(std::stoul(match_str));
                        }

                        // Get the vertex coordinates and instantiate simplex
                        MatrixXd vertex_coords(vertex_indices.size(), dim);
                        for (unsigned i = 0; i < vertex_indices.size(); ++i)
                        {   // Each row is a vertex in the simplex 
                            for (unsigned j = 0; j < dim; ++j)
                            {
                                vertex_coords(i,j) = vertices[vertex_indices[i]][j];
                            }
                        }
                        simplices.emplace_back(Simplex(vertex_coords));

                        // Keep track of pre-computed volumes as well
                        std::string volume_num = matches[matches.size() - 2].str();
                        std::string volume_den = matches[matches.size() - 1].str();
                        double volume = 0.0;
                        // The line matches the regular expression and the volume
                        // was specified as an integer
                        if (volume_den.empty())
                            volume = std::stod(volume_num);
                        // The line matches the regular expression and the volume
                        // was specified as a fraction
                        else
                            volume = std::stod(volume_num) / std::stod(volume_den.erase(0, 1));
                        volumes.push_back(volume);
                    }
                    // The line does not match the regular expression
                    else
                    {
                        std::cerr << "Incorrect number of matches" << std::endl;
                        throw std::exception();
                    }
                }
                else
                {
                    std::cerr << "Does not match regex" << std::endl;
                    throw std::exception();
                }
            }
        }
    }

    return std::make_tuple(vertices, simplices, volumes); 
}

MatrixXd getBoundingBox(const Ref<const MatrixXd>& constraints)
{
    /*
     * Given a matrix specifying a set of linear inequalities, find a bounding
     * box that contains the polytope specified by the inequalities. 
     *
     * Though this problem can be addressed more generally via vertex
     * enumeration, we assume that ``constraints'' contains rows that
     * specify the lower and upper bounds of each parameter (independently
     * of the other parameters), in which case a bounding box is easily 
     * identified. If no such constraints are specified for any of the 
     * parameters, a std::runtime_error is raised. 
     */
    unsigned nrows = constraints.rows();
    unsigned ncols = constraints.cols();
    MatrixXd box_bounds = MatrixXd::Zero(ncols - 1, 2);

    for (unsigned i = 1; i < ncols; ++i)
    {
        // Find the lower bound and upper bound for the i-th parameter 
        double lower = std::numeric_limits<double>::infinity();
        double upper = -std::numeric_limits<double>::infinity();

        for (unsigned j = 0; j < nrows; ++j)
        {
            // Look for any constraints in which all entries other than 
            // the 0-th and i-th entries are zero 
            bool valid = true;  
            for (unsigned k = 1; k < ncols; ++k)
            {
                if (k != i && constraints(j, k) != 0) 
                {
                    valid = false; 
                    break;
                }
            }
            if (valid)
            {
                // Case 1: x + y*param >= 0, or param >= -x / y
                if (constraints(j, i) > 0)
                {
                    if (lower > -(constraints(j, 0) / constraints(j, i)))
                        lower = -(constraints(j, 0) / constraints(j, i));
                }
                // Case 2: x - y*param >= 0, or param <= x / y
                else if (constraints(j, i) < 0)
                {
                    if (upper < constraints(j, 0) / (-constraints(j, i)))
                        upper = constraints(j, 0) / (-constraints(j, i));
                }
            }
        }

        box_bounds(i-1, 0) = lower;
        box_bounds(i-1, 1) = upper; 
    }

    // Check that a finite lower/upper bound exists for each parameter
    for (unsigned i = 0; i < ncols - 1; ++i)
    {
        if (box_bounds(i, 0) == std::numeric_limits<double>::infinity() ||
            box_bounds(i, 1) == -std::numeric_limits<double>::infinity())
        {
            throw std::runtime_error("Lower/upper bound for at least one parameter missing in input constraints"); 
        }
    }

    return box_bounds; 
}

VectorXd sampleUnitVector(unsigned dim, boost::random::mt19937& rng)
{
    /*
     * Sample a unit vector with the given dimension from the uniform 
     * distribution on the unit hypersphere. 
     */
    // Get (dim) samples from the standard normal, then normalize the 
    // vector by its Euclidean norm
    VectorXd sample = VectorXd::Zero(dim);
    boost::random::normal_distribution<> dist(0.0, 1.0);
    for (unsigned i = 0; i < dim; ++i)
        sample(i) = dist(rng);

    return sample / sample.norm();
}

std::pair<double, double> findChordEndpoints(Polytope& polytope,
                                             const Ref<const VectorXd>& point,
                                             const Ref<const VectorXd>& direction, 
                                             double atol = 1e-8)
{
    /*
     * Given a convex polytope, a point, and a unit vector, find the 
     * endpoints of the chord arising as the intersection of the convex
     * polytope with the line passing through the given point parallel to 
     * the given unit vector. 
     */
    std::function<double(double)> findSingleEndpoint = [polytope, point, direction, atol](double increment = 1.0)
    {
        /*
         *
         */
        // Jump in increments away from the given point until we leave the 
        // polytope 
        double beta = 0.0;
        while (polytope.contains(point + beta * direction))
            beta += increment;

        // Now that we are out of the region, run a version of binary search 
        // between the current value of beta (above) and alpha (initialized 
        // to zero), and continually update beta and alpha until their 
        // difference is less than atol
        double alpha = 0.0;
        while (std::abs(alpha - beta) >= atol)
        {
            double avg = (alpha + beta) / 2.0;

            // If the average lies within the polytope, then update alpha
            if (polytope.contains(point + avg * direction))
                alpha = avg; 
            else 
                beta = avg;
        }

        // Return the scalar estimate whose corresponding point along the 
        // given direction lies *within* the polytope (i.e., alpha)
        return alpha; 
    };

    // Now, the two endpoints can be found by advancing in opposite directions
    // along the given unit vector from the given point 
    double alpha1 = findSingleEndpoint(-1.0);
    double alpha2 = findSingleEndpoint(1.0);

    return std::make_pair(alpha1, alpha2);
}

MatrixXd sampleFromConvexPolytopeRandomWalk(Polytope& polytope, unsigned npoints,
                                            boost::random::mt19937& rng, 
                                            unsigned nchains = 5, double atol = 1e-8,
                                            double warmup = 0.5, unsigned ntrials = 50)
{
    /*
     * Sample from a bounded convex polytope within some bounding box via 
     * Smith's hit-and-run random walk.
     *
     * Convergence of the random walk is monitored by running multiple 
     * independent runs of the walk and computing the Gelman-Rubin statistic
     * for each parameter separately. 
     */
    unsigned nrows = polytope.nrows();
    unsigned nparams = polytope.nparams();

    // Get lower and upper bounds for each parameter 
    MatrixXd box_bounds = getBoundingBox(polytope.constraints);

    // Instantiate a Polytope object for the bounding box 
    MatrixXd box_constraints = MatrixXd::Zero(nparams * 2, nparams + 1);
    for (unsigned i = 0; i < nparams; ++i)
    {
        double lower = box_bounds(i, 0);
        double upper = box_bounds(i, 1);
        box_constraints(2*i, 0) = -lower;
        box_constraints(2*i + 1, 0) = upper;
        box_constraints(2*i, i + 1) = 1;
        box_constraints(2*i + 1, i + 1) = -1;
    }
    Polytope box = Polytope(box_constraints);

    // Initialize the random walk through rejection sampling ...
    VectorXd lower_bounds = box_bounds.col(0);
    VectorXd upper_bounds = box_bounds.col(1);

    // ... by randomly sampling a direction ...
    VectorXd z = VectorXd::Zero(nparams);
    boost::random::uniform_01<double> dist;
    for (unsigned i = 0; i < nparams; ++i) z(i) = dist(rng);

    // ... and advancing from the "lowermost" vertex of the bounding box 
    VectorXd init_point = lower_bounds + ((upper_bounds - lower_bounds).array() * z.array()).matrix();
    while (!polytope.contains(init_point))
    {
        for (unsigned i = 0; i < nparams; ++i) z(i) = dist(rng);
        init_point = lower_bounds + ((upper_bounds - lower_bounds).array() * z.array()).matrix();
    }

    // Run the random walk ...
    std::vector<MatrixXd> chains;
    for (unsigned i = 0; i < nchains; ++i)
    {
        MatrixXd init_chain = MatrixXd::Zero(100, nparams);
        init_chain.row(0) = init_point;
        chains.push_back(init_chain);
    }
    unsigned count = 1;    // Number of points that have been sampled 
    bool converged = false; 
    unsigned curr = 0;     // Number of convergence checks (done every 100 points) 
    unsigned nburned = 0;
    unsigned nretained = 0;
    while (!converged && curr < ntrials)
    {
        for (unsigned i = 0; i < nchains; ++i)
        {
            // Get the last point visited by the chain 
            VectorXd curr_point = chains[i].row(count - 1);

            // First, sample a direction (a unit vector) from the current point
            VectorXd direction = sampleUnitVector(nparams, rng);

            // Second, find the endpoints of the chord arising as the 
            // intersection of the line passing through the current point 
            // along the sampled direction with the polytope 
            std::pair<double, double> endpoints = findChordEndpoints(polytope, curr_point, direction, atol);

            // Pick a point from the uniform distribution along this chord 
            double multiplier = endpoints.first + (endpoints.second - endpoints.first) * dist(rng);
            chains[i].row(count) = curr_point + multiplier * direction;
        }
        count++; 

        // Every 100 points, compute the Gelman-Rubin statistic for the 
        // mean and covariance *for each parameter* over the latter
        // (1 - warmup) fraction of each run
        if (count >= chains[0].rows())
        {
            // Burn the first (warmup * count) points in each run, and 
            // ensure that we retain an even number of points per run 
            nburned = static_cast<unsigned>(warmup * count);
            nretained = count - nburned; 
            if (nretained % 2 == 1)
            {
                nburned--;
                nretained++;
            }

            // Treat each of the parameters individually ...
            converged = true;  
            for (unsigned i = 0; i < nparams; ++i)
            {
                // Split each of the retained chains in half and treat them 
                // as separate chains
                MatrixXd points_retained = MatrixXd::Zero(nchains * 2, nretained / 2);
                for (unsigned j = 0; j < nchains; ++j)
                {
                    points_retained.row(2*j) = chains[j].col(i).segment(nburned, nretained / 2);
                    points_retained.row(2*j + 1) = chains[j].col(i).segment(nburned + nretained / 2, nretained / 2);
                }

                // Estimate the potential scale reduction factor 
                unsigned nseqs = nchains * 2;
                unsigned nper = nretained / 2;
                VectorXd means_within = points_retained.rowwise().mean();
                double mean_total = points_retained.mean();
                double var_between = (1.0 * nper / (nseqs - 1)) * (means_within.array() - mean_total).pow(2).matrix().sum();
                double var_within = (1.0 / (nper * (nseqs - 1))) * (points_retained.colwise() - means_within).array().pow(2).matrix().sum();
                double psrf = ((nseqs + 1) / nseqs) * ((nper - 1) * var_within / nper + var_between / nper) / var_within
                    + (nper - 1) / (nper * nseqs);

                // If any of the potential scale reduction factors is >= 1.2,
                // we have not converged
                if (psrf >= 1.2)
                {
                    converged = false;
                    break;
                }
            }

            // Increment convergence check counter 
            curr++;
            std::cout << "[CHECK " << curr << "] PSRF >= 1.2 for at least one parameter, not converged ("
                      << chains[0].rows() << " points sampled per chain)\n";

            // If convergence has not been reached, add 100 more rows to each
            // chain matrix 
            if (!converged)
            {
                for (unsigned i = 0; i < nchains; ++i)
                    chains[i].conservativeResize(chains[i].rows() + 100, chains[i].cols());
            }
        }
    }

    // Has convergence been achieved? 
    if (!converged)
        throw std::runtime_error("Convergence not achieved within given number of trials");

    // If we have not yet sampled the desired number of points ...
    unsigned nfinal = nchains * nretained;
    while (nfinal < npoints)
    {
        for (unsigned i = 0; i < nchains; ++i)
        {
            // Resize the chain matrix 
            chains[i].conservativeResize(chains[i].rows() + 1, chains[i].cols());

            // Get the last point visited by the chain 
            VectorXd curr_point = chains[i].row(chains[i].rows() - 2);

            // First, sample a direction (a unit vector) from the current point
            VectorXd direction = sampleUnitVector(nparams, rng);

            // Second, find the endpoints of the chord arising as the 
            // intersection of the line passing through the current point 
            // along the sampled direction with the polytope 
            std::pair<double, double> endpoints = findChordEndpoints(polytope, curr_point, direction, atol);

            // Pick a point from the uniform distribution along this chord 
            double multiplier = endpoints.first + (endpoints.second - endpoints.first) * dist(rng);
            chains[i].row(chains[i].rows() - 1) = curr_point + multiplier * direction;
        }
        nretained++;
        nfinal = nchains * nretained; 
    }

    // Pool points from all chains into a single matrix 
    MatrixXd points_pooled = MatrixXd::Zero(nfinal, nparams);
    for (unsigned i = 0; i < nchains; ++i)
        points_pooled.block(i * nretained, 0, nretained, nparams) = chains[i].bottomRows(nretained);

    // Permute the rows in this pooled matrix, then return the desired
    // number of points
    std::vector<int> indices;
    for (int i = 0; i < npoints; ++i)
        indices.push_back(i);
    boost::range::random_shuffle(indices);
    VectorXi perm(npoints);
    for (int i = 0; i < npoints; ++i)
        perm(i) = indices[i]; 

    return perm.asPermutation() * points_pooled.topRows(npoints);
}

MatrixXd sampleFromConvexPolytopeTriangulation(std::string triangulation_file,
                                               unsigned npoints,
                                               boost::random::mt19937& rng)
{
    /*
     * Given a .delv file specifying a convex polytope in terms of its 
     * vertices and its Delaunay triangulation, parse the simplices of
     * the triangulation and sample uniformly from the polytope,
     * returning the vertices of the polytope and the sampled points.  
     */
    // Parse given triangulation file
    auto triangulation = parseTriangulation(triangulation_file);
    std::vector<std::vector<double> > vertices = std::get<0>(triangulation); 
    std::vector<Simplex> simplices = std::get<1>(triangulation);
    std::vector<double> volumes = std::get<2>(triangulation);

    // Instantiate a categorical distribution with probabilities 
    // proportional to the simplex volumes 
    double sum_volumes = 0.0;
    for (auto&& v : volumes) sum_volumes += v;
    for (auto&& v : volumes) v /= sum_volumes;
    boost::random::discrete_distribution<> dist(volumes);

    // Maintain an array of points ...
    unsigned dim = vertices[0].size(); 
    MatrixXd sample(npoints, dim);
    for (unsigned i = 0; i < npoints; i++)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Sample a point from the simplex
        sample.row(i) = simplices[j].sample(1, rng);
    }

    return sample;
}

#endif
