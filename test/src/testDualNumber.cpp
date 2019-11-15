#define BOOST_TEST_MODULE testDualNumber
#define BOOST_TEST_DYN_LINK
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/duals/duals.hpp"
#include "../../include/duals/eigen.hpp"

/*
 * Test module for the Duals::DualNumber class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/14/2019
 */
using namespace Eigen;
using Duals::DualNumber;

DualNumber rosenbrock(const Ref<const VectorXDual>& x)
{
    /*
     * 2-D Rosenbrock function.
     */
    using Duals::pow;
    return pow(1.0 - x(0), 2.0) + 100.0 * (x(1) - pow(x(0), 2.0), 2.0);
}

BOOST_AUTO_TEST_CASE(testInitialize)
{
    DualNumber v;
    BOOST_TEST(v.x() == 0.0);
    BOOST_TEST(v.d() == 0.0);
}

BOOST_AUTO_TEST_CASE(testInitializeFromDouble)
{
    DualNumber v(10.0);
    BOOST_TEST(v.x() == 10.0);
    BOOST_TEST(v.d() == 0.0);

    DualNumber w = 10.0;
    BOOST_TEST(v.x() == 10.0);
    BOOST_TEST(v.d() == 0.0);
}

BOOST_AUTO_TEST_CASE(testMatrixFromDoubleInput)
{
    MatrixXDual A(3, 3);
    A << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    BOOST_TEST((A(0,0).x() == 1.0 && A(0,0).d() == 0.0));
    BOOST_TEST((A(0,1).x() == 2.0 && A(0,1).d() == 0.0));
    BOOST_TEST((A(0,2).x() == 3.0 && A(0,2).d() == 0.0));
    BOOST_TEST((A(1,0).x() == 4.0 && A(1,0).d() == 0.0));
    BOOST_TEST((A(1,1).x() == 5.0 && A(1,1).d() == 0.0));
    BOOST_TEST((A(1,2).x() == 6.0 && A(1,2).d() == 0.0));
    BOOST_TEST((A(2,0).x() == 7.0 && A(2,0).d() == 0.0));
    BOOST_TEST((A(2,1).x() == 8.0 && A(2,1).d() == 0.0));
    BOOST_TEST((A(2,2).x() == 9.0 && A(2,2).d() == 0.0));
}

BOOST_AUTO_TEST_CASE(testMatrixFromDualNumberInput)
{
    MatrixXDual A(3, 3);
    A << DualNumber(1.0, 1.0), DualNumber(2.0, 2.0), DualNumber(3.0, 3.0),
         DualNumber(4.0, 4.0), DualNumber(5.0, 5.0), DualNumber(6.0, 6.0),
         DualNumber(7.0, 7.0), DualNumber(8.0, 8.0), DualNumber(9.0, 9.0);
    BOOST_TEST((A(0,0).x() == 1.0 && A(0,0).d() == 1.0));
    BOOST_TEST((A(0,1).x() == 2.0 && A(0,1).d() == 2.0));
    BOOST_TEST((A(0,2).x() == 3.0 && A(0,2).d() == 3.0));
    BOOST_TEST((A(1,0).x() == 4.0 && A(1,0).d() == 4.0));
    BOOST_TEST((A(1,1).x() == 5.0 && A(1,1).d() == 5.0));
    BOOST_TEST((A(1,2).x() == 6.0 && A(1,2).d() == 6.0));
    BOOST_TEST((A(2,0).x() == 7.0 && A(2,0).d() == 7.0));
    BOOST_TEST((A(2,1).x() == 8.0 && A(2,1).d() == 8.0));
    BOOST_TEST((A(2,2).x() == 9.0 && A(2,2).d() == 9.0));
}

BOOST_AUTO_TEST_CASE(testRosenbrockGradient)
{
    VectorXDual v(2);
    v << 1.0, 1.0;
    DualNumber u;
    VectorXd grad = Duals::gradient(rosenbrock, v, u);
    BOOST_TEST(grad(0) == 0.0);
    BOOST_TEST(grad(1) == 0.0);
}
