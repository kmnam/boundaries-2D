#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <boost/multiprecision/gmp.hpp>
#include <boost/random.hpp>
#include "../include/boundaries.hpp"
#include "../include/boundaryFinder.hpp"

/*
 * Approximate the boundary of a 2-D projection of a convex polytope. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     5/9/2022
 */
using boost::multiprecision::mpq_rational;

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);
    auto coin_toss = [&fair_bernoulli_dist](boost::random::mt19937& rng)
    {
        return fair_bernoulli_dist(rng);
    };

    // Check and parse input arguments
    std::string infilename, outprefix;
    bool simplify = false;
    int max_edges = 0; 
    if (argc == 2)    // Only acceptable call here is "./findProjection --help"
    {
        std::string option(argv[1]);
        if (option == "-h" || option == "--help")
        {
            std::stringstream ss;
            ss << "\nCommand:\n\t./findProjection [-s NUM_EDGES] INPUT OUTPUT\n\n";
            ss << "\t-s (optional): Simplify alpha shape to contain at most the given number of edges\n";
            ss << "\tINPUT: Input file\n";
            ss << "\tOUTPUT: Output file prefix\n\n";
            std::cerr << ss.str();
            return -1;
        }
        else
        {
            std::stringstream ss;
            ss << "Unrecognized input command\n\n";
            ss << "Help:\n\t./findProjection --help\n\n";
            std::cerr << ss.str();
            return -1;
        }
    }
    else if (argc == 3)    // Only acceptable call here is "./findProjection INPUT OUTPUT"
    {
        infilename = argv[1];
        outprefix = argv[2];
    }
    else if (argc == 4)    // No acceptable call here
    {
        std::stringstream ss;
        ss << "Unrecognized input command\n\n";
        ss << "Help:\n\t./findProjection --help\n\n";
        std::cerr << ss.str();
        return -1;
    }
    else if (argc == 5)    // Only acceptable call here is "./findProjection -s NUM_EDGES INPUT OUTPUT"
    {
        std::string option(argv[1]);
        if (option == "-s")
        {
            simplify = true;
            std::string token(argv[2]); 
            if (std::regex_match(token, std::regex("((\\+|-)?[[:digit:]]+)")))
            {
                max_edges = std::stoi(token);
            }
            else
            {
                std::stringstream ss; 
                ss << "Non-numeric argument to -s flag\n\n";
                ss << "Help:\n\t./findProjection --help\n\n";
                std::cerr << ss.str();
                return -1;  
            }
        }
        else
        {
            std::stringstream ss;
            ss << "Unrecognized flag: " << option << "\n\n";
            ss << "Help:\n\t./findProjection --help\n\n";
            std::cerr << ss.str();
            return -1;
        }
        infilename = argv[3];
        outprefix = argv[4]; 
    }
    else
    {
        std::stringstream ss;
        ss << "Unrecognized input command\n\n";
        ss << "Help:\n\t./findProjection --help\n\n";
        std::cerr << ss.str();
        return -1;
    }

    // Parse the input polytope and instantiate a BoundaryFinder object
    const double area_tol = 1e-8;
    Polytopes::LinearConstraints<mpq_rational>* constraints = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    constraints->parse(infilename);
    std::function<VectorXd(const Ref<const VectorXd>&)> func = [](const Ref<const VectorXd>& x)
    {
        return x(Eigen::seqN(0, 2));
    }; 
    BoundaryFinder* finder = new BoundaryFinder(
        area_tol, rng, infilename,
        Polytopes::InequalityType::LessThanOrEqualTo,
        func
    );

    // Compute boundary with assumption that region is simply connected
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)>
        mutate = [&coin_toss](const Ref<const VectorXd>& x, boost::random::mt19937& rng)
    {
        VectorXd mutated(x); 
        const double delta = 0.1; 
        for (unsigned i = 0; i < mutated.size(); ++i)
        {
            int toss = coin_toss(rng); 
            if (!toss) mutated(i) += delta;
            else       mutated(i) -= delta;
        }
        return mutated; 
    };
    std::function<bool(const Ref<const VectorXd>&)> filter = [](const Ref<const VectorXd>& x)
    {
        return false; 
    };
    const int n_init = 50; 
    MatrixXd init_input = finder->sampleInput(n_init);
    const int min_step_iter = 4;
    const int max_step_iter = 8; 
    const int min_pull_iter = 10;
    const int max_pull_iter = 20;
    const unsigned sqp_max_iter = 100;
    const double sqp_tol = 1e-6;
    const double tau = 0.5; 
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const bool use_only_armijo = false;  
    const bool use_strong_wolfe = false;
    const bool hessian_modify_max_iter = 1000;
    const double c1 = 1e-4; 
    const double c2 = 0.9; 
    const bool verbose = true;
    const bool sqp_verbose = true;
    finder->run(
        mutate, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, tau, delta, beta,
        use_only_armijo, use_strong_wolfe, hessian_modify_max_iter, outprefix,
        c1, c2, verbose, sqp_verbose
    );
    MatrixXd final_input = finder->getInput(); 

    return 0;
}
