#define BOOST_TEST_MODULE testLinearConstraints
#define BOOST_TEST_DYN_LINK
#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include "../../include/linearConstraints.hpp"

/*
 * Test module for the LinearConstraints class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/13/2019
 */
using namespace Eigen;

BOOST_AUTO_TEST_CASE(testParse)
{
    LinearConstraints constraints;
    constraints.parse("/n/groups/gunawardena/chris_nam/boundaries/test/src/test-1.poly");
    MatrixXd A = constraints.getA();
    VectorXd b = constraints.getb();

    // Test that each coordinate of A and b were correctly specified
    MatrixXd A2(16, 6);
    VectorXd b2(16);
    A2 << -1,  0,  0,  0,  0,  0,
           0, -1,  0,  0,  0,  0,
           0,  0, -1,  0,  0,  0,
           0,  0,  0, -1,  0,  0,
           0,  0,  0,  0, -1,  0,
           0,  0,  0,  0,  0, -1,
           1,  0,  0,  0,  0,  0,
           0,  1,  0,  0,  0,  0,
           0,  0,  1,  0,  0,  0,
           0,  0,  0,  1,  0,  0,
           0,  0,  0,  0,  1,  0,
           0,  0,  0,  0,  0,  1,
           1, -1,  0,  0,  0,  0,
           0,  0, -1,  1,  0,  0,
           1,  0, -1,  0,  0,  0,
           0, -1,  0,  1,  0,  0;
    b2 << -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0;
    BOOST_TEST(A == A2);
    BOOST_TEST(b == b2);
}
