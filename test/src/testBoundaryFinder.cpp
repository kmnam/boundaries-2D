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
    joinPath(ss_out, output_dir, "square-2-mutate-func1");
    std::function<VectorXd(const Ref<const VectorXd>&)> obj1 = func1; 
    BoundaryFinder<2>* finder1 = new BoundaryFinder<2>(
        tol, rng, square_constraints.str(), square_vertices.str(), 
        Polytopes::GreaterThanOrEqualTo, obj1
    );
    MatrixXd init_input1 = Polytopes::sampleFromConvexPolytope<100>(square_vertices.str(), 100, 0, rng); 
    finder1->run(   // Only use <= 10 mutations ... 
        add_delta,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering
        init_input1,  
        0,         // Minimum of 0 mutation iterations
        10,        // Maximum of 10 mutation iterations
        0,         // Minimum of 0 pulling iterations
        0,         // Maximum of 0 pulling iterations
        10,        // Simplify boundary to 10 points 
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-5,      // SQP convergence tolerance
        true,      // Verbose output
        true,      // Verbose SQP output
        true,      // Use line-search SQP
        ss_out.str() 
    );
    joinPath(ss_out, output_dir, "square-2-pull-func1"); 
    finder1->run(   // Only use <= 10 pulling iterations ... 
        add_delta,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering
        init_input1,  
        0,         // Minimum of 0 mutation iterations
        0,         // Maximum of 0 mutation iterations
        0,         // Minimum of 0 pulling iterations
        10,        // Maximum of 10 pulling iterations
        10,        // Simplify boundary to 10 points 
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-5,      // SQP convergence tolerance
        true,      // Verbose output
        true,      // Verbose SQP output 
        true,      // Use line-search SQP
        ss_out.str()
    );
   
    // Run boundary-finding algorithm on the image of project() on the cube
    joinPath(ss_out, output_dir, "cube-4-mutate-project");
    std::function<VectorXd(const Ref<const VectorXd>&)> obj2 = project;  
    BoundaryFinder<4>* finder2 = new BoundaryFinder<4>(
        tol, rng, cube_constraints.str(), cube_vertices.str(), 
        Polytopes::GreaterThanOrEqualTo, obj2
    );
    MatrixXd init_input2 = Polytopes::sampleFromConvexPolytope<100>(cube_vertices.str(), 100, 0, rng);
    finder2->run(   // Only use <= 10 mutations ... 
        add_delta,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering
        init_input2,  
        0,         // Minimum of 0 mutation iterations
        10,        // Maximum of 10 mutation iterations
        0,         // Minimum of 0 pulling iterations
        0,         // Maximum of 0 pulling iterations
        10,        // Simplify boundary to 10 points 
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-5,      // SQP convergence tolerance
        true,      // Verbose output
        true,      // Verbose SQP output
        true,      // Use line-search SQP
        ss_out.str() 
    );
    joinPath(ss_out, output_dir, "cube-4-pull-project");
    finder2->run(   // Only use <= 10 pulling iterations ... 
        add_delta,
        [](const Ref<const VectorXd>& v){ return false; },    // No filtering
        init_input2,  
        0,         // Minimum of 0 mutation iterations
        0,         // Maximum of 0 mutation iterations
        0,         // Minimum of 0 pulling iterations
        10,        // Maximum of 10 pulling iterations
        10,        // Simplify boundary to 10 points 
        10,        // Maximum of 10 quadratic programs per SQP iteration
        1e-5,      // SQP convergence tolerance
        true,      // Verbose output
        true,      // Verbose SQP output 
        true,      // Use line-search SQP
        ss_out.str()
    );

    delete finder1; 
    delete finder2; 
    return 0;
}
