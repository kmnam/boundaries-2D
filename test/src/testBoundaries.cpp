#define BOOST_TEST_MODULE testBoundaries
#define BOOST_TEST_DYN_LINK
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <boost/test/included/unit_test.hpp>
#include "../../include/boundaries.hpp"

/*
 * Test module for the AlphaShape2DProperties and Boundary2D classes.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/9/2019
 */
using namespace Eigen;

AlphaShape2DProperties getTwentyPoints()
{
    /*
     * Return an AlphaShape2DProperties instance with 20 points and 
     * their convex hull. 
     */
    MatrixXd points(20, 2);
    points << 0.61879477, 0.59162363,
              0.88868359, 0.89165480,
              0.45756748, 0.77818808,
              0.26706377, 0.99610621,
              0.54009489, 0.53752161,
              0.40099938, 0.70540579,
              0.40518559, 0.94999075,
              0.03075388, 0.13602495,
              0.08297726, 0.42352224,
              0.23449723, 0.74743526,
              0.65177865, 0.68998682,
              0.16413419, 0.87642114,
              0.44733314, 0.57871104,
              0.52377835, 0.62689056,
              0.34869427, 0.26209748,
              0.07498055, 0.17940570,
              0.82999425, 0.98759822,
              0.11326099, 0.63846415,
              0.73056694, 0.88321124,
              0.52721004, 0.66487673;
    std::vector<double> x, y;
    for (unsigned i = 0; i < 20; ++i)
    {
        x.push_back(points(i, 0));
        y.push_back(points(i, 1));
    }

    // Vertices are specified in counterclockwise (left turns) order
    std::vector<unsigned> vertices = {7, 14, 1, 16, 3, 11, 17};
    std::vector<std::pair<unsigned, unsigned> > edges = { {7, 14}, {14, 1}, {1, 16}, {16, 3}, {3, 11}, {11, 17}, {17, 7} };
    
    return AlphaShape2DProperties(
        x, y, vertices, edges,
        100.0,                // alpha set to some arbitrary large value
        2.757265508219424,    // area of convex hull
        true, true, false,    // connected, simply connected, not simplified
        true                  // check ordering of vertices 
    );
}

BOOST_AUTO_TEST_CASE(testAlphaShape2DPropertiesOrient)
{
    /*
     * Test that the orientation of 20-point-set convex hull is correctly
     * initialized and successfully reversed. 
     */
    AlphaShape2DProperties shape = getTwentyPoints();
    BOOST_TEST(shape.orientation == CGAL::LEFT_TURN);

    // Reverse the orientation and check that the edges are specified 
    // in the correct order
    shape.orient(CGAL::RIGHT_TURN);
    BOOST_TEST(shape.orientation == CGAL::RIGHT_TURN);
    BOOST_TEST((shape.vertices[0] == 7  && shape.edges[0].first == 7  && shape.edges[0].second == 17));
    BOOST_TEST((shape.vertices[1] == 17 && shape.edges[1].first == 17 && shape.edges[1].second == 11));
    BOOST_TEST((shape.vertices[2] == 11 && shape.edges[2].first == 11 && shape.edges[2].second == 3));
    BOOST_TEST((shape.vertices[3] == 3  && shape.edges[3].first == 3  && shape.edges[3].second == 16));
    BOOST_TEST((shape.vertices[4] == 16 && shape.edges[4].first == 16 && shape.edges[4].second == 1));
    BOOST_TEST((shape.vertices[5] == 1  && shape.edges[5].first == 1  && shape.edges[5].second == 14));
    BOOST_TEST((shape.vertices[6] == 14 && shape.edges[6].first == 14 && shape.edges[6].second == 7));
}
