#include "../../include/boundaryPlot.hpp"

/*
 * Test module for plotting boundaries with pybind11.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/24/2019
 */
int main()
{
    /*
     * Plot the simplified boundary given in the examples directory. 
     */
    plotBoundary(
        "/Users/kmnam/Dropbox/gene-regulation/projects/boundaries/example/boundary_AN3_simplified.txt",
        "/Users/kmnam/Dropbox/gene-regulation/projects/boundaries/example/boundary_AN3_simplified.pdf",
        "Position", "Steepness"
    );
    return 0;
}

