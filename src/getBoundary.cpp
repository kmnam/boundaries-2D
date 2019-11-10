#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "../include/boundaries.hpp"

/*
 * Compute the boundary of a 2-D point cloud. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/10/2019
 */

int main(int argc, char** argv)
{
    // Check and parse input arguments
    std::string infile, outfile;
    bool simplify = false;
    if (argc == 2)
    {
        std::string option(argv[1]);
        if (option == "-h" || option == "--help")
        {
            std::stringstream ss;
            ss << "\nCommand:\n\t./getBoundary [--simplify] INPUT OUTPUT\n\n";
            ss << "\t--simplify (optional): Output simplified alpha shape\n";
            ss << "\tINPUT: Input file\n";
            ss << "\tOUTPUT: Output file\n\n";
            std::cerr << ss.str();
            return -1;
        }
        else
        {
            std::stringstream ss;
            ss << "Unrecognized input command\n\n";
            ss << "Help:\n\t./getBoundary --help\n\n";
            std::cerr << ss.str();
            return -1;
        }
    }
    else if (argc == 3)
    {
        infile = argv[1];
        outfile = argv[2];
    }
    else if (argc == 4)
    {
        std::string option(argv[1]);
        if (option == "--simplify")
            simplify = true;
        else
        {
            std::stringstream ss;
            ss << "Unrecognized flag: " << option << "\n\n";
            ss << "Help:\n\t./getBoundary --help\n\n";
            std::cerr << ss.str();
            return -1;
        }
        infile = argv[2];
        outfile = argv[3]; 
    }
    else
    {
        std::stringstream ss;
        ss << "Unrecognized input command\n\n";
        ss << "Help:\n\t./getBoundary --help\n\n";
        std::cerr << ss.str();
        return -1;
    }

    // Parse input points and store in vectors 
    std::vector<double> x;
    std::vector<double> y;
    std::ifstream f;
    f.open(infile);
    if (!f.good())
    {
        std::stringstream ss;
        ss << "Unrecognized input file: " << infile << "\n\n";
        ss << "Help:\n\t./getBoundary --help\n\n";
        std::cerr << ss.str();
        return -1;
    }
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream iss(line);
        double a, b;
        if (!(iss >> a >> b)) break;
        x.push_back(a);
        y.push_back(b);
    }
    f.close();

    // Compute boundary with assumption that region is simply connected
    Boundary2D boundary(x, y);
    AlphaShape2DProperties bound_data = boundary.getBoundary(true, true, simplify);
    bound_data.write(outfile);

    return 0;
}
