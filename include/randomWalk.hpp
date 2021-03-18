#ifndef RANDOM_WALK_CONVEX_POLYTOPE_SAMPLING_HPP
#define RANDOM_WALK_CONVEX_POLYTOPE_SAMPLING_HPP 

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>

/*
 * Functions for sampling from convex polytopes via random walks. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/17/2021
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
     * Parse the given file of linear inequalities (half-planes) and
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

#endif 
