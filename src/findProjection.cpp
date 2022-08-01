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
 *     8/1/2022
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

    // Default parameter values 
    int n_init = 200;
    int max_edges = 30;
    int n_keep_interior = 100; 
    int n_keep_origbound = 100;
    int n_mutate_origbound = 10;
    int n_pull_origbound = 10;
    int max_step_iter = 50;
    int max_pull_iter = 50;

    // Check and parse input arguments
    std::string infilename, outprefix;
    if (argc == 2)             // Only acceptable call here is "./findProjection --help"
    {
        std::string option(argv[1]);
        if (option == "-h" || option == "--help")
        {
            std::stringstream ss;
            ss << "\nCommand:\n\t./findProjection "
               << "[-i N_INIT] "
               << "[-e MAX_EDGES] "
               << "[-n N_KEEP_INTERIOR] "
               << "[-o N_KEEP_ORIGBOUND] "
               << "[-x N_MUTATE/PULL_ORIGBOUND] "
               << "[-m MAX_MUTATION_ITER] "
               << "[-p MAX_PULL_ITER] "
               << "INPUT OUTPUT\n\n";
            ss << "\t-i (optional): Size of initial sample\n";
            ss << "\t-e (optional): Maximum number of edges in each simplified boundary\n";
            ss << "\t-n (optional): Maximum number of points to sample from interior of each boundary\n"; 
            ss << "\t-o (optional): Maximum number of vertices to preserve from complement of simplified "
               << "boundary w.r.t unsimplified boundary\n";
            ss << "\t-x (optional): Maximum number of vertices to mutate/pull from complement of simplified "
               << "boundary w.r.t unsimplified boundary\n";
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
    else if (argc >= 5 && argc <= 17 && argc % 2 == 1)    // Optional arguments specified
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
            // Each token should be an positive integer 
            if (std::regex_match(token, std::regex("((\\+|-)?[[:digit:]]+)")))
            {
                int arg = std::stoi(token);
                if (arg <= 0)
                {
                    std::stringstream ss;
                    ss << "Invalid zero or negative argument specified\n\n";
                    ss << "Help:\n\t./findProjection --help\n\n";
                    std::cerr << ss.str(); 
                    return -1;
                }
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
            if (options[i] == "-i")
            {
                n_init = args[i];
            }
            else if (options[i] == "-e")
            {
                max_edges = args[i];
            }
            else if (options[i] == "-o")
            {
                n_keep_origbound = args[i];
            }
            else if (options[i] == "-x")
            {
                n_mutate_origbound = args[i];
                n_pull_origbound = args[i];
            }
            else if (options[i] == "-n")
            {
                n_keep_interior = args[i];
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
    MatrixXd init_input = finder->sampleInput(n_init);
    const int min_step_iter = 1;
    const int min_pull_iter = 1;
    const int sqp_max_iter = 100;
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
    const bool write_pulled_points = true; 
    finder->run(
        mutate_delta, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, n_keep_interior,
        n_keep_origbound, n_mutate_origbound, n_pull_origbound, tau, delta,
        beta, use_only_armijo, use_strong_wolfe, hessian_modify_max_iter,
        outprefix, RegularizationMethod::NOREG, 0, c1, c2, verbose, sqp_verbose,
        write_pulled_points
    );
    
    return 0;
}
