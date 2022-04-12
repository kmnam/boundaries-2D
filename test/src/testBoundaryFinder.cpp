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
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     4/12/2022
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
template <int Dim>
void testBoundaryFinder(BoundaryFinder<Dim>* finder,
                        const Ref<const MatrixXd>& init_input, 
                        const std::string output_prefix, 
                        std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate)
{
    /*
     * Run the boundary-finding algorithm on the projection of the given
     * polytope onto the first two coordinates. 
     */
    finder->run(
        mutate,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering
        init_input,  
        10,        // Minimum of 10 mutation iterations
        100,       // Maximum of 100 mutation iterations
        3,         // Minimum of 3 pulling iterations
        20,        // Maximum of 20 pulling iterations
        50,        // Simplify boundary to 50 points 
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-5,      // SQP convergence tolerance
        true,      // Verbose output
        false,     // Suppress SQP output
        true,      // Use line-search SQP
        output_prefix
    );
}

int main(int argc, char** argv)
{
    const double tol = 1e-8; 
    boost::random::mt19937 rng(1234567890); 
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
    std::stringstream square_constraints, square_vertices;
    joinPath(square_constraints, polytope_dir, "square-2.poly");
    joinPath(square_vertices, polytope_dir, "square-2.vert"); 

    // Cube of length 2 in 4-D
    std::stringstream cube_constraints, cube_vertices;
    joinPath(cube_constraints, polytope_dir, "cube-4.poly");
    joinPath(cube_vertices, polytope_dir, "cube-4.vert"); 

    // Output directory 
    std::string output_dir = argv[2];
    
    // Run boundary-finding algorithm on the image of func1() on the square 
    std::stringstream ss_out;
    joinPath(ss_out, output_dir, "square-2-func1");
    std::function<VectorXd(const Ref<const VectorXd>&)> obj1 = func1; 
    BoundaryFinder<2>* finder1 = new BoundaryFinder<2>(
        tol, rng, square_constraints.str(), square_vertices.str(), 
        Polytopes::GreaterThanOrEqualTo, obj1
    );
    MatrixXd init_input1 = Polytopes::sampleFromConvexPolytope<100>(square_vertices.str(), 20, 0, rng); 
    testBoundaryFinder(finder1, init_input1, ss_out.str(), add_delta);
   
    // Run boundary-finding algorithm on the image of project() on the cube 
    joinPath(ss_out, output_dir, "cube-4-project");
    std::function<VectorXd(const Ref<const VectorXd>&)> obj2 = project;  
    BoundaryFinder<4>* finder2 = new BoundaryFinder<4>(
        tol, rng, cube_constraints.str(), cube_vertices.str(), 
        Polytopes::GreaterThanOrEqualTo, obj2
    );
    MatrixXd init_input2 = Polytopes::sampleFromConvexPolytope<100>(cube_vertices.str(), 20, 0, rng); 
    testBoundaryFinder(finder2, init_input2, ss_out.str(), add_delta); 

    delete finder1; 
    delete finder2; 
    return 0;
}
