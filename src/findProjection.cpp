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
 *     5/12/2022
 */
using boost::multiprecision::mpq_rational;

/**
 * Get the maximum distance between any pair of vertices in the given matrix.
 *
 * @param vertices Matrix of vertex coordinates.
 * @returns        Maximum distance between any pair of vertices. 
 */
template <typename T>
T getMaxDist(const Ref<const Matrix<T, Dynamic, Dynamic> >& vertices)
{
    T maxdist = 0;
    for (int i = 0; i < vertices.rows() - 1; ++i)
    {
        for (int j = i + 1; j < vertices.rows(); ++j)
        {
            T dist = (vertices.row(i) - vertices.row(j)).norm(); 
            if (maxdist < dist)
                maxdist = dist; 
        }
    }
    
    return maxdist; 
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_int_distribution<> dist(1, 10000000); 

    // Check and parse input arguments
    std::string infilename, outprefix;
    int max_edges = 30;        // Maximum number of 30 edges by default
    int max_step_iter = 10;    // Maximum number of 10 mutation iterations by default
    int max_pull_iter = 10;    // Maximum number of 10 pulling iterations by default 
    if (argc == 2)             // Only acceptable call here is "./findProjection --help"
    {
        std::string option(argv[1]);
        if (option == "-h" || option == "--help")
        {
            std::stringstream ss;
            ss << "\nCommand:\n\t./findProjection [-s NUM_EDGES] [-m MAX_MUTATION_ITER] [-p MAX_PULL_ITER] INPUT OUTPUT\n\n";
            ss << "\t-s (optional): Simplify alpha shape to contain at most the given number of edges\n";
            ss << "\t-m (optional): Maximum number of mutation iterations\n";
            ss << "\t-p (optional): Maximum number of pulling iterations\n";
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
    else if (argc == 5 || argc == 7 || argc == 9)    // Optional arguments specified
    {
        std::vector<std::string> options;
        std::vector<std::string> tokens;
        std::vector<int> args;
        for (int i = 1; i < argc - 3; i += 2)
        {
            options.push_back(argv[i]);
            tokens.push_back(argv[i + 1]); 
        } 

        // Disallow re-specification of the same argument
        for (int i = 0; i < options.size() - 1; ++i)
        {
            for (int j = i + 1; j < options.size(); ++j)
            {
                if (options[i] == options[j])
                {
                    std::stringstream ss; 
                    ss << "Optional argument specified multiple times: " << options[i] << "\n\n"; 
                    ss << "Help:\n\t./findProjection --help\n\n";
                    std::cerr << ss.str();
                    return -1;  
                }
            }
        } 

        // Interpret each token 
        for (std::string token : tokens)
        {
            if (std::regex_match(token, std::regex("((\\+|-)?[[:digit:]]+)")))
            {
                int arg = std::stoi(token);
                args.push_back(arg); 
            }
            else 
            {
                std::stringstream ss; 
                ss << "Invalid non-numeric argument specified\n\n"; 
                ss << "Help:\n\t./findProjection --help\n\n";
                std::cerr << ss.str();
                return -1;  
            }
        }
        for (int i = 0; i < options.size(); ++i)
        { 
            if (options[i] == "-s")
            {
                max_edges = args[i];
            }
            else if (options[i] == "-m")
            {
                max_step_iter = args[i];
            }
            else if (options[i] == "-p")
            {
                max_pull_iter = args[i];
            }
            else 
            {
                std::stringstream ss;
                ss << "Unrecognized flag: " << options[i] << "\n\n";
                ss << "Help:\n\t./findProjection --help\n\n";
                std::cerr << ss.str();
                return -1;
            }
        }
        infilename = argv[argc - 2];
        outprefix = argv[argc - 1]; 
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
        Polytopes::InequalityType::LessThanOrEqualTo, func
    );

    // Compute boundary with assumption that region is simply connected
    double mutate_delta = 0.1 * getMaxDist<double>(finder->getVertices());
    std::function<bool(const Ref<const VectorXd>&)> filter = [](const Ref<const VectorXd>& x)
    {
        return false; 
    };
    const int n_init = 50; 
    MatrixXd init_input = finder->sampleInput(n_init);
    const int min_step_iter = 1;
    const int min_pull_iter = 1;
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
    const bool sqp_verbose = false;
    finder->run(
        mutate_delta, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, tau, delta, beta,
        use_only_armijo, use_strong_wolfe, hessian_modify_max_iter, outprefix,
        c1, c2, verbose, sqp_verbose
    );
    
    return 0;
}
