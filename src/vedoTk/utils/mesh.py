from typing import Union, Optional
from numpy import ndarray, array, concatenate, unique, setdiff1d, arange, dot, argwhere, in1d
from vedo import Points, Mesh, TetMesh, Plotter
from vtk import vtkGeometryFilter


class SurfaceSelection:

    def __init__(self,
                 mesh: Union[Mesh, TetMesh],
                 selection_idx: Optional[ndarray] = None,
                 keep_colors: bool = False):
        """
        Create a 3D objects collection to easily select vertices or cells on a surface mesh.

        :param mesh: The mesh to select, either a vedo.Mesh or a vedo.TetMesh object.
        :param selection_idx: Initial list of selected vertices / cells.
        :param keep_colors:
        """

        # Create the surface mesh
        if isinstance(mesh, Mesh):
            self.__surface: Mesh = mesh.copy()
            self.__idx_tetra = None
        else:
            # In tetra mesh case, create a surface mesh from the tetra then get conversion between indices
            gf = vtkGeometryFilter()
            gf.SetInputData(mesh.dataset)
            gf.Update(0)
            self.__surface: Mesh = Mesh(gf.GetOutput())
            pts = Points(mesh.vertices)
            self.__idx_tetra = array([pts.closest_point(v, return_point_id=True) for v in self.__surface.vertices]).astype(int)
        self.__surface.compute_normals().lw(1).c('grey')

        # Create the selection point cloud
        self.__surface_points = Points(self.__surface.vertices)
        self.__surface_points.pickable(True).point_size(8).c('lightgreen')
        self.__selection = Points()

        # Selection
        if selection_idx is None:
            self.__idx = array([]).astype(int)
        elif self.__idx_tetra is None:
            self.__idx = selection_idx.astype(int)
        else:
            self.__idx = self.__idx_tetra[selection_idx]

    @property
    def selected_idx(self) -> ndarray:

        # Tetra mesh: convert vertices index
        if self.__idx_tetra is not None:
            return self.__idx_tetra[self.__idx]

        # Surface mesh: simply return selection
        return self.__idx.copy()

    @selected_idx.setter
    def selected_idx(self, idx):

        self.__idx = array(idx).astype(int)

    @property
    def selected_points_coord(self) -> ndarray:

        if len(self.__idx) == 0:
            return array([])
        return self.__surface.vertices[self.__idx]

    def init(self, plt: Plotter):

        plt.add(self.__surface, self.__surface_points, self.__selection)

    def add(self, idx: ndarray):

        idx = concatenate((self.__idx, idx))
        self.__idx = idx[sorted(unique(idx, return_index=True)[1])]

    def remove(self, idx: ndarray):

        self.__idx = self.__idx[~in1d(self.__idx, idx)]

    def clear(self):

        self.__idx = array([], dtype=int)

    def all(self):

        self.__idx = arange(self.__surface.nvertices)

    def invert(self):

        self.__idx = setdiff1d(arange(self.__surface.nvertices), self.__idx)

    def update_selection(self, plt: Plotter, cursor_idx: Optional[ndarray] = None, point_size: float = 8.) -> None:

        # Remove the visual selection from plotter
        plt.remove(self.__selection)

        # Create a new visual selection
        if cursor_idx is None or len(cursor_idx) == 0:
            self.__selection = Points() if len(self.__idx) == 0 else Points(self.__surface.vertices[self.__idx])
        else:
            idx = concatenate((self.__idx, cursor_idx))
            idx = idx[sorted(unique(idx, return_index=True)[1])]
            self.__selection = Points(self.__surface.vertices[idx])

        # Add the new visual selection to the plotter
        self.__selection.pickable(False).point_size(point_size).color('tomato').alpha(0.9)
        plt.add(self.__selection)

    def get_closest_points(self, pos: ndarray, radius: float = 0.) -> ndarray:

        # Get the closest points from pos within the defined radius
        idx = self.__surface_points.closest_point(pos, radius=radius, return_point_id=True)
        idx = array([idx]) if isinstance(idx, int) else array(idx)

        # Return empty selection
        if len(idx) == 0:
            return array([])

        # Filter the points with normals
        pos_normal = self.__surface.point_normals[self.__surface.closest_point(pos, return_point_id=True)]
        pos_dots = array([dot(pos_normal, vertex_normal) for vertex_normal in self.__surface.vertex_normals[idx]])
        return idx[argwhere(pos_dots > 0).flatten()]
