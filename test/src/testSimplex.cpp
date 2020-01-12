#define BOOST_TEST_MODULE testSimplex
#define BOOST_TEST_DYN_LINK
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/simplex.hpp"

/*
 * Test module for the Simplex class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     1/11/2020
 */
using namespace Eigen;

BOOST_AUTO_TEST_CASE(testVolumeTriangle)
{
    MatrixXd points(3, 2);
    points << 1.0, 0.0,
              1.0, 1.0,
              0.0, 0.0;
    Simplex simplex(points);
    std::vector<unsigned> indices;
    BOOST_TEST(simplex.faceVolume(indices) == 0.5);
    indices.push_back(1);
    BOOST_TEST(simplex.faceVolume(indices) == 1.0);
}
