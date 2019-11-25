#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/24/2019
 */
namespace py = pybind11;

void plotBoundary(std::string boundary_filename, std::string plot_filename,
                  std::string xlabel, std::string ylabel)
{
    /*
     * Plot the boundary written in the given input file, and save it 
     * to a PDF file at the given path. 
     */
    // From matplotlib.pyplot import figure
    py::object figure = py::module::import("matplotlib.pyplot").attr("figure");

    // From boundaries import Boundary2D
    py::object Boundary2D = py::module::import("boundaries").attr("Boundary2D");

    // Call the class method from_file()
    py::object b = Boundary2D.attr("from_file")(py::cast(boundary_filename));

    // Instantiate a figure and grab its axes
    py::object fig = figure();
    py::object ax = fig.attr("gca")();
    std::cout << "instantiated figure\n";

    // Plot the boundary onto the instantiated axes
    b.attr("plot")(ax);

    // Label the axes with the given labels
    ax.attr("set_xlabel")(py::cast(xlabel));
    ax.attr("set_ylabel")(py::cast(ylabel));

    // Save the figure to file
    py::object savefig = py::module::import("matplotlib.pyplot").attr("savefig");
    savefig(py::cast(plot_filename));
}
