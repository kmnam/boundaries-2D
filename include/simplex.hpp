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
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     1/11/2020
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
};

#endif 
