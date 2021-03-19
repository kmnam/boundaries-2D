#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "../../include/sample.hpp"
#include "../../include/boundaryFinder.hpp"

/*
 * Test module for the BoundaryFinder class. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/19/2021
 */
using namespace Eigen;

boost::random::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

// -------------------------------------------------------- //
//                      TEST FUNCTIONS                      //
// -------------------------------------------------------- //
VectorXd project(const Ref<const VectorXd>& x)
{
    /*
     * Project onto the first two coordinates. 
     */
    return x.head(2);
}

VectorXd func1(const Ref<const VectorXd>& x)
{
    /*
     * (x, y) ---> (x^2 + y^2, x^2 - y^2).
     */
    VectorXd y(x.size());
    y << x(0) * x(0) + x(1) * x(1), x(0) * x(0) - x(1) * x(1);
    return y;
}

// -------------------------------------------------------- //
//                     MUTATION FUNCTIONS                   //
// -------------------------------------------------------- //
VectorXd add_delta(const Ref<const VectorXd>& x, boost::random::mt19937& rng)
{
    /*
     * Perturb each coordinate up or down by 0.1. 
     */
    const double delta = 0.1;
    VectorXd y(x.size());
    for (unsigned i = 0; i < x.size(); ++i)
    {
        int toss = coin_toss(rng);
        y(i) = (!toss ? x(i) + delta : x(i) - delta);
    }
    return y;
}

// -------------------------------------------------------- //
//                 BOUNDARY FINDER FUNCTIONS                //
// -------------------------------------------------------- //
void testBoundaryFinder(std::string poly_filename, std::string vert_filename,
                        std::string output_prefix, std::function<VectorXd(const Ref<const VectorXd>&)> func,
                        std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate)
{
    /*
     * Run BoundaryFinder on the projection of the given polytope onto the 
     * first two coordinates. 
     */
    boost::random::mt19937 rng(1234567890);
    const double tol = 1e-5;         // Tolerance for convergence

    BoundaryFinder finder(tol, rng, poly_filename, vert_filename);
    finder.run(
        func, mutate,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering 
        20, 20,    // 20 points within the cube, 20 points from the boundary
        10,        // Minimum of 10 mutation iterations
        100,       // Maximum of 100 mutation iterations
        3,         // Minimum of 3 pulling iterations
        20,        // Maximum of 20 pulling iterations
        50,        // Simplify boundary to 50 points 
        true,      // Verbose output
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-4,      // SQP convergence tolerance
        false,     // Suppress SQP output
        output_prefix 
    );
}

int main(int argc, char** argv)
{
    std::string polytope_dir = argv[1];

    // Function for writing paths to input files 
    std::function<void(std::stringstream&, std::string, std::string)> joinPath =
        [](std::stringstream& ss, std::string dir, std::string filename)
    {
        ss.clear(); 
        ss.str(std::string());
        if (dir.compare(dir.length() - 1, 1, "/") == 0)
            ss << dir << filename;  
        else
            ss << dir << "/" << filename; 
    };

    // Square of length 2 in 2-D
    std::stringstream square_poly, square_vert;
    joinPath(square_poly, polytope_dir, "square-2.poly");
    joinPath(square_vert, polytope_dir, "square-2.vert"); 

    // Cube of length 2 in 4-D
    std::stringstream cube_poly, cube_vert;
    joinPath(cube_poly, polytope_dir, "cube-4.poly");
    joinPath(cube_vert, polytope_dir, "cube-4.vert"); 

    // Output directory 
    std::string output_dir = argv[2];
    
    // Run boundary-finding algorithm
    std::stringstream ss_out;
    joinPath(ss_out, output_dir, "square-2-func1");
    testBoundaryFinder(
        square_poly.str(), square_vert.str(), ss_out.str(), func1, add_delta
    );
    joinPath(ss_out, output_dir, "cube-4-project"); 
    testBoundaryFinder(
        cube_poly.str(), cube_vert.str(), ss_out.str(), project, add_delta
    );

    return 0;
}
