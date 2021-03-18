#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

/*
 * Helper class for representing n-dimensional simplices. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     3/17/2021
 */
using namespace Eigen;

int factorial(int n)
{
    /*
     * Returns n!. Should only be used for small n. 
     */
    if (n == 0)
        return 1;
    else
        return n * factorial(n - 1);
}

class Simplex
{
    /*
     * A simple wrapper class for a simplex.
     */
    public:    // No private variables
        unsigned d;        // Dimension of ambient space
        unsigned k;        // Dimension of simplex
        MatrixXd points;

        Simplex()
        {
            /*
             * Initialize to the zero-simplex at the origin.
             */
            this->d = 1;
            this->k = 0;
            this->points = MatrixXd::Zero(1, 1);
        }

        Simplex(const Ref<const MatrixXd>& points)
        {
            /*
             * Initialize to the given simplex.
             *
             * Each row in the given matrix is a vertex in the simplex. 
             */
            this->d = points.cols();
            this->k = points.rows() - 1;
            if (this->d < this->k)
            {
                throw std::invalid_argument("d is less than k");
            }
            this->points = points;
        }

        ~Simplex()
        {
            /*
             * Trivial destructor.
             */
        }

        void setPoints(const Ref<const MatrixXd>& points)
        {
            /*
             * Set the points in the simplex to the given coordinates.
             * Identical to constructor. 
             */
            this->d = points.cols();
            this->k = points.rows() - 1;
             if (this->d < this->k)
            {
                throw std::invalid_argument("d is less than k");
            }
            this->points = points;
        }

        double faceVolume(std::vector<unsigned> idx)
        {
            /*
             * Compute the volume of the simplex face obtained by removing
             * the given vertices.  
             */
            // Pick out the vertices not to be excluded 
            MatrixXd face_points(this->k + 1 - idx.size(), this->d);
            unsigned j = 0;
            for (unsigned i = 0; i < this->k + 1; ++i)
            {
                if (std::find(idx.begin(), idx.end(), i) == idx.end())
                {
                    face_points.row(j) = points.row(i);
                    j++;
                }
            }

            // Compute the Cayley-Menger matrix
            unsigned size = face_points.rows() + 1;
            MatrixXd cayley_menger = MatrixXd::Zero(size, size);
            for (unsigned i = 0; i < face_points.rows() - 1; ++i)
            {
                for (unsigned j = i + 1; j < face_points.rows(); ++j)
                {
                    double dij = (face_points.row(i) - face_points.row(j)).squaredNorm();
                    cayley_menger(i,j) = dij;
                    cayley_menger(j,i) = dij;
                }
            }
            for (unsigned i = 0; i < size - 1; ++i)
            {
                cayley_menger(i,size-1) = 1.0;
                cayley_menger(size-1,i) = 1.0;
            }

            return std::sqrt(std::abs(cayley_menger.determinant()) / (std::pow(factorial(size - 2), 2) * std::pow(2, size - 2))); 
        }

        MatrixXd sample(unsigned npoints, boost::random::mt19937& rng)
        {
            /*
             * Randomly sample the given number of points from the 
             * uniform density (i.e., flat Dirichlet) on the simplex.
             */
            // Sample the desired number of points from the flat Dirichlet 
            // distribution on the standard simplex of appropriate dimension
            MatrixXd barycentric(npoints, this->points.rows());
            boost::random::gamma_distribution<> gamma_dist(1.0);
            for (unsigned i = 0; i < npoints; ++i)
            {
                // Sample independent Gamma-distributed variables with
                // alpha = 1, and normalize by their sum
                for (unsigned j = 0; j < this->points.rows(); ++j)
                    barycentric(i,j) = gamma_dist(rng);
                barycentric.row(i) = barycentric.row(i) / barycentric.row(i).sum();
            }
       
            // Convert from barycentric coordinates to Cartesian coordinates
            MatrixXd sample = barycentric * this->points;

            return sample;
        }
};

MatrixXd sampleFromSimplices(std::vector<Simplex> simplices, unsigned npoints,
                             boost::random::mt19937& rng)
{
    /*
     * Given a vector of simplices, sample with probability proportional 
     * to each simplex's volume. The simplices all must have the same
     * dimension.
     */
    // Check that the simplices all have the same dimension and are 
    // embedded in the same space
    if (simplices.size() == 0)
        throw std::invalid_argument("Specified no input simplices");
    unsigned d = simplices[0].d;
    unsigned k = simplices[0].k;
    for (auto&& simplex : simplices)
    {
        if (simplex.d != d || simplex.k != k)
            throw std::invalid_argument("Invalid input vector of simplices");
    }

    // Compute the volume of each simplex
    std::vector<double> volumes;
    std::vector<unsigned> idx;
    for (auto&& simplex : simplices)
        volumes.push_back(simplex.faceVolume(idx));

    // Instantiate a categorical distribution with probabilities 
    // proportional to the simplex volumes 
    double sum_volumes = 0.0;
    for (auto&& v : volumes) sum_volumes += v;
    for (auto&& v : volumes) v /= sum_volumes;
    boost::random::discrete_distribution<> dist(volumes);

    // Maintain an array of points ...
    MatrixXd sample(npoints, d);
    for (unsigned i = 0; i < npoints; i++)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Sample a point from the selected simplex
        sample.row(i) = simplices[j].sample(1, rng);
    }

    return sample;
} 

#endif 
