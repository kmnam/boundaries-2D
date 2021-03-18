#ifndef SAMPLE_HPP
#define SAMPLE_HPP

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
#include "simplex.hpp"

/*
 * Functions for random sampling.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/17/2021
 */
using namespace Eigen;

std::tuple<std::vector<std::vector<double> >, std::vector<Simplex>, std::vector<double> >
    parseTriangulation(std::string triangulation_file)
{
    /*
     * Given a .delv file specifying a convex polytope in terms of its 
     * vertices and its Delaunay triangulation, parse its simplices 
     * and return them as a vector. 
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
