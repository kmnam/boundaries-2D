"""
A simple class for plotting a point cloud and its boundary by parsing 
text files in the format output by the functions Grid2DProperties::write() 
and AlphaShape2DProperties::write().

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
    3/19/2021

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import seaborn as sns

class Boundary2D(object):
    """
    A simple wrapper class that stores boundary information.
    """
    def __init__(self):
        """
        Empty constructor.
        """
        self.points = []
        self.vertices = []
        self.edges = []
        self.alpha = None
        self.area = None

    #######################################################
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
        with open(path) as f:
            for line in f:
                if line.startswith('ALPHA'):      # Line specifies alpha value 
                    b.alpha = float(line.strip().split('\t')[1])
                elif line.startswith('AREA'):     # Line specifies enclosed area 
                    b.area = float(line.strip().split('\t')[1])
                elif line.startswith('POINT'):    # Line specifies a point
                    xs, ys = line.replace('POINT', '').strip().split('\t')
                    b.points.append([float(xs), float(ys)])
                elif line.startswith('VERTEX'):   # Line specifies a vertex
                    b.vertices.append(int(line.replace('VERTEX', '').strip()))
                elif line.startswith('EDGE'):     # Line specifies an edge
                    ss, ts = line.replace('EDGE', '').strip().split('\t')
                    b.edges.append([int(ss), int(ts)])
        b.points = np.array(b.points)

        return b

    #######################################################
    def plot(self, ax, plot_interior=False, shade_interior=False,
             interior_color=sns.xkcd_rgb['denim blue'],
             boundary_color=sns.xkcd_rgb['pale red'], interior_size=20,
             boundary_size=30, interior_alpha=0.1, rasterized=True):
        """
        Plot the stored points, with the boundary points emphasized and 
        connected by their edges.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes onto which the points are to be plotted.
        
        Returns
        -------
        None
        """
        # Shade in the interior if desired
        if shade_interior:
            polygon = patches.Polygon(
                self.points[self.vertices,:], closed=True, facecolor=boundary_color,
                edgecolor=boundary_color, alpha=0.3
            )
            ax.add_patch(polygon)

        # Plot the interior points if desired
        if plot_interior:
            ax.scatter(
                self.points[:,0], self.points[:,1], c=[interior_color], s=interior_size,
                alpha=interior_alpha, zorder=0, rasterized=rasterized
            )
        
        # Plot the boundary points
        ax.scatter(
            self.points[self.vertices,0], self.points[self.vertices,1],
            c=[boundary_color], s=boundary_size, zorder=2
        )

        # Plot each boundary edge
        for edge in self.edges:
            ax.plot(self.points[edge,0], self.points[edge,1], c=boundary_color, zorder=1)

