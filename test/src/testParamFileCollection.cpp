#define BOOST_TEST_MODULE testParamFileCollection
#define BOOST_TEST_DYN_LINK
#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include "../../include/boundaryFinder.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/11/2019
 */

using namespace Eigen;

BOOST_AUTO_TEST_CASE(testSmallExample)
{
    /*
     * Test that ParamFileCollection methods run correctly on a small example.
     */ 
    MatrixXd params(12, 6);
    params.block(0, 0, 3, 6) = MatrixXd::Constant(3, 6, 1.0);
    params.block(3, 0, 3, 6) = MatrixXd::Constant(3, 6, 2.0);
    params.block(6, 0, 3, 6) = MatrixXd::Constant(3, 6, 3.0);
    params.block(9, 0, 3, 6) = MatrixXd::Constant(3, 6, 4.0);

    // Dump each block into a file
    ParamFileCollection collection("test");
    for (unsigned i = 0; i < 4; ++i)
    {
        MatrixXd block = params.block(i*3, 0, 3, 6);
        collection.dump(block);
    }

    // Read each block and check that each was stored correctly
    for (unsigned i = 0; i < 4; ++i)
    {
        MatrixXd block = collection.read(i);
        BOOST_TEST(block.rows() == 3);
        BOOST_TEST(block.cols() == 6);
        for (unsigned j = 0; j < 3; ++j)
        {
            for (unsigned k = 0; k < 6; ++k)
            {
                BOOST_TEST(block(j,k) == static_cast<double>(i) + 1.0);
            }
        }
    }
}
