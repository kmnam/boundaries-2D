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
    4/9/2022
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import seaborn as sns

######################################################################
def line_segments_intersect(line1_x1, line1_y1, line1_x2, line1_y2,
                            line2_x1, line2_y1, line2_x2, line2_y2): 
    """
    Given the endpoints of two line segments, determine whether the 
    line segments intersect. 

    Parameters
    ----------
    line1_x1 : float
        x-coordinate of endpoint 1 of line 1. 
    line1_y1 : float
        y-coordinate of endpoint 1 of line 1. 
    line1_x2 : float
        x-coordinate of endpoint 2 of line 1. 
    line1_y2 : float
        y-coordinate of endpoint 2 of line 1. 
    line2_x1 : float
        x-coordinate of endpoint 1 of line 2.
    line2_y1 : float
        y-coordinate of endpoint 1 of line 2.
    line2_x2 : float
        x-coordinate of endpoint 2 of line 2. 
    line2_y2 : float
        y-coordinate of endpoint 2 of line 2.

    Returns
    -------
    bool
        True if the line segments intersect, False otherwise. 
    """
    # The two line segments intersect if, and only if, the linear system 
    # 
    # [ line1_x2 - line1_x1  -(line2_x2 - line2_x1) ] [ s ] = [ line2_x1 - line1_x1 ]
    # [ line1_y2 - line1_y1  -(line2_y2 - line2_y1) ] [ t ] = [ line2_y1 - line1_y1 ]
    # 
    # has a solution with 0 <= s,t <= 1
    # 
    # Compute this solution via Cramer's rule for 2x2 systems 
    A = np.array([
        [line1_x2 - line1_x1, -(line2_x2 - line2_x1)], 
        [line1_y2 - line1_y1, -(line2_y2 - line2_y1)]
    ])
    b = np.array([line2_x1 - line1_x1, line2_y1 - line1_y1])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    s = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    t = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

    return (s > 0 and s < 1 and t > 0 and t < 1)

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
        self.bbox_coords = [None, None, None, None] 

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

        # Compute the bounding-box coordinates of the boundary
        b.bbox_coords = b.get_bounding_box()

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
    def get_bounding_box(self):
        """
        Return the values of the least x-coordinate, least y-coordinate, 
        greatest x-coordinate, and greatest y-coordinate among the points
        along the boundary. 

        These values form the vertices of a bounding rectangle for the 
        polygon formed by the boundary. 

        Parameters
        ----------
        None

        Returns
        -------
        list of four floats
            Least x-coordinate, least y-coordinate, greatest x-coordinate, 
            and greatest y-coordinate among the boundary vertices. 
        """
        vertices = self.points[self.vertices, :]
        return [
            vertices[:, 0].min(), vertices[:, 1].min(),
            vertices[:, 0].max(), vertices[:, 1].max()
        ]

    ##################################################################
    def _contains_ray_casting(self, x, y, rng=None): 
        """
        Given a (2-D) point, determine whether the point lies within the 
        boundary with the ray casting algorithm.

        Parameters
        ----------
        x : float
            Input x-coordinate. 
        y : float
            Input y-coordinate.
        rng : `numpy.random.Generator`
            Random number generator for sampling the initial point outside 
            the boundary. 

        Returns
        -------
        bool
            True if the point lies within the boundary, False otherwise. 
        """
        if rng is None:
            rng = np.random.default_rng(1234567890)

        # Find a point outside the boundary by perturbing a corner of the 
        # bounding box by random increments 
        bbox_length = self.bbox_coords[2] - self.bbox_coords[0]
        epsilon = bbox_length / 100.
        x_init = self.bbox_coords[2] + epsilon * (1 + rng.random())
        y_init = self.bbox_coords[3] + epsilon * (1 + rng.random())

        # Define a quick function that tests whether a point lies within
        # (a slightly larger version of) the bounding box
        bbox_contains = lambda x0, y0: (
            x0 > self.bbox_coords[0] - epsilon and
            x0 < self.bbox_coords[2] + epsilon and
            y0 > self.bbox_coords[1] - epsilon and
            y0 < self.bbox_coords[3] + epsilon 
        )

        # Return False if the given point is not within the bounding box 
        if not bbox_contains(x, y): 
            return False

        # Define a quick function that advances along the ray from the 
        # initial point to the given point
        advance = lambda t: (x_init + t * (x - x_init), y_init + t * (y - y_init))

        # Draw a ray from the initial point toward the given point, *starting
        # from the given point* ...
        x_ray_endpoint = x
        y_ray_endpoint = y
        delta = 0.1
        t = 1.0
        while bbox_contains(x_ray_endpoint, y_ray_endpoint):
            # Get a new point along the ray 
            x_ray_endpoint, y_ray_endpoint = advance(t)

            # Increment t (to advance along the ray)
            t += delta

        # Check if the line segment starting at the given point and ending at
        # the ray endpoint (which should now lie outside the boundary's
        # bounding box) intersects with any of the edges along the boundary
        num_intersect = 0
        for i, j in self.edges:
            edge_x1, edge_y1 = self.points[i, 0], self.points[i, 1]
            edge_x2, edge_y2 = self.points[j, 1], self.points[j, 1]
            num_intersect += line_segments_intersect(
                edge_x1, edge_y1, edge_x2, edge_y2, 
                x, y, x_ray_endpoint, y_ray_endpoint
            )

        # The given point lies inside the boundary if the line segment
        # intersects an odd number of edges, and not otherwise
        return num_intersect % 2

    ##################################################################
    def contains(self, x, y):
        """
        Given a (2-D) point, determine whether the point lies within the 
        boundary.

        Parameters
        ----------
        x : float
            Input x-coordinate. 
        y : float
            Input y-coordinate. 

        Returns
        -------
        bool
            True if the point lies within the boundary, False otherwise. 
        """
        return self._contains_ray_casting(x, y) 

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
            facecolor=('none' if not shade_interior else interior_color),
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

