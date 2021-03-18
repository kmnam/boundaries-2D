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
 *     1/11/2020
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

// -------------------------------------------------------- //
//                     MUTATION FUNCTIONS                   //
// -------------------------------------------------------- //
template <double delta = 0.1>
VectorXd add_delta(const Ref<const VectorXd>& x, boost::random::mt19937& rng)
{
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
void testBoundaryFinderProject(std::string poly_filename, std::string vert_filename,
                               std::string output_prefix)
{
    /*
     * Run BoundaryFinder on the projection of the given polytope onto the 
     * first two coordinates. 
     */
    boost::random::mt19937 rng(1234567890);
    const double tol = 1e-5;         // Tolerance for convergence

    BoundaryFinder<double> finder(tol, rng, poly_filename, vert_filename);
    finder.run(
        project, add_delta<0.1>,
        [](const Ref<const VectorXd>&){ return false; },    // No filtering 
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

    // Cube of length 2 in 4-D
    std::stringstream ss1_poly, ss1_vert; 
    if (polytope_dir.compare(polytope_dir.length() - 1, 1, "/") == 0)
    {
        ss1_poly << polytope_dir << "cube-4.poly";
        ss1_vert << polytope_dir << "cube-4.vert";
    }
    else
    {
        ss1_poly << polytope_dir << "/cube-4.poly";
        ss1_vert << polytope_dir << "/cube-4.vert";
    }

    // Output directory 
    std::string output_dir = argv[2];
    std::stringstream ss1_prefix; 
    if (output_dir.compare(output_dir.length() - 1, 1, "/") == 0)
        ss1_prefix << output_dir << "cube-4-project";
    else 
        ss1_prefix << output_dir << "/cube-4-project"; 
    
    // Run boundary-finding algorithm  
    testBoundaryFinderProject(ss1_poly.str(), ss1_vert.str(), ss1_prefix.str());

    return 0;
}
