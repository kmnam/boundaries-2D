"""
A simple class for plotting a point cloud and its boundary by parsing 
text files in the format output by the functions `Grid2DProperties::write()`
and `AlphaShape2DProperties::write()`.

Example usage:

    ```
    > from boundaries import Boundary2D
    > import matplotlib.pyplot as plt
    > b = Boundary2D.from_file(filename)
    > fig, ax = plt.subplots()
    > b.plot(ax)
    > plt.show()
    ```

Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School

Last updated:
    1/8/2022
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import seaborn as sns

######################################################################
class Boundary2D(object):
    """
    A simple wrapper class that stores boundary information.
    """
    def __init__(self):
        """
        Empty constructor.
        """
        self.points = np.array([], dtype=np.float64)
        self.vertices = []
        self.edges = []
        self.input = np.array([], dtype=np.float64)
        self.alpha = None
        self.area = None

    ##################################################################
    @classmethod
    def from_file(cls, path):
        """
        Parse a Boundary2D object from the given file.

        Parameters
        ----------
        path : str
            Path to input file.

        Returns
        -------
        None
        """
        b = Boundary2D()
        points = []
        inputs = []
        with open(path) as f:
            for line in f:
                if line.startswith('ALPHA'):      # Line specifies alpha value 
                    b.alpha = float(line.strip().split('\t')[1])
                elif line.startswith('AREA'):     # Line specifies enclosed area 
                    b.area = float(line.strip().split('\t')[1])
                elif line.startswith('POINT'):    # Line specifies a point
                    xs, ys = line.replace('POINT\t', '').strip().split('\t')
                    points.append([float(xs), float(ys)])
                elif line.startswith('VERTEX'):   # Line specifies a vertex
                    b.vertices.append(int(line.replace('VERTEX\t', '').strip()))
                elif line.startswith('EDGE'):     # Line specifies an edge
                    ss, ts = line.replace('EDGE\t', '').strip().split('\t')
                    b.edges.append([int(ss), int(ts)])
                elif line.startswith('INPUT'):    # Line specifies an input point
                    inputs.append([
                        float(x) for x in line.replace('INPUT\t', '').strip().split('\t')
                    ])
        b.points = np.array(points, dtype=np.float64)
        b.input = np.array(inputs, dtype=np.float64)

        # If there are input points present in the file but their number
        # doesn't match the number of output points in the file, then raise 
        # an exception
        if b.input.shape[0] > 0 and b.points.shape[0] != b.input.shape[0]:
            raise RuntimeError('Input and output dimensions do not match')

        return b

    ##################################################################
    def get_boundary_points(self):
        """
        Return the coordinates of the vertices in the stored boundary.

        Parameters
        ----------
        None

        Returns
        -------
        `numpy.ndarray`
            Array of coordinates of the vertices in the stored boundary,  
            with each row corresponding to a vertex. 
        """
        return self.points[self.vertices, :]

    ##################################################################
    def get_boundary_inputs(self):
        """
        Return the input points whose images correspond to the vertices 
        in the stored boundary. 

        Parameters
        ----------
        None

        Returns
        -------
        `numpy.ndarray`
            Array of coordinates of the input points whose images
            correspond to the vertices in the stored boundary, with
            each row corresponding to an input point. Returns an empty
            array if the input points are not stored.
        """
        if self.input.shape[0] == 0:
            return self.input
        else:
            return self.input[self.vertices, :]

    ##################################################################
    def plot(self, ax, color=None, linewidth=None, scatter=False,
             scatter_color=None, pointsize=None, shade_interior=False,
             interior_color=None, scatter_alpha=None, shade_alpha=None,
             autoscale=True, boundary_kws={}, scatter_kws={}):
        """
        Plot the stored points, with the boundary points emphasized and 
        connected by their edges.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes onto which the points are to be plotted.
        color : RGB tuple or string
            Color of the polygonal edges along the boundary.
        linewidth : float
            Width of the polygonal edges along the boundary. 
        scatter : bool
            Whether to plot the polygonal vertices along the boundary.
        scatter_color : RGB tuple or string
            Color of the polygonal vertices along the boundary, if the 
            vertices are to be plotted.
        pointsize : float
            Size of the polygonal vertices along the boundary, if the 
            vertices are to be plotted. 
        shade_interior : bool
            Whether to shade in the interior.
        interior_color : RGB tuple or string
            Color for the interior shading, if the interior is to be 
            shaded. 
        scatter_alpha : float
            Alpha transparency value for boundary scatter-plotted points.
        shade_alpha : float
            Alpha transparency value for interior shading.
        autoscale : bool
            If True, auto-scale the axes limits. 
        boundary_kws : dict
            Dict of keywords to be passed to `matplotlib.patches.Polygon()`
            when instantiating the boundary polygon.
        scatter_kws : dict
            Dict of keywords to be passed to `matplotlib.pyplot.scatter()` 
            when plotting the boundary vertices. 
        
        Returns
        -------
        None
        """
        # Instantiate the Polygon object to be plotted 
        polygon = patches.Polygon(
            self.points[self.vertices, :], closed=True,
            facecolor=(None if not shade_interior else interior_color),
            edgecolor=color, alpha=(None if not shade_interior else shade_alpha),
            linewidth=linewidth, **boundary_kws
        )
        ax.add_patch(polygon)

        # Plot the individual vertices along the boundary if desired 
        if scatter:
            ax.scatter(
                self.points[self.vertices, 0],
                self.points[self.vertices, 1],
                c=scatter_color, s=pointsize, alpha=scatter_alpha,
                **scatter_kws
            )

        # Auto-scale the axes limits if desired.
        if autoscale:
            ax.autoscale_view()

