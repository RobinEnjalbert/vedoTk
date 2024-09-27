from typing import Optional, TypeVar, List, Tuple, Union

from numpy import ndarray, array
from numpy import round as np_round
from numpy.linalg import norm
from threading import Thread
from vedo import Mesh, show, visible_points

TVisiblePoints = TypeVar('TVisiblePoints', bound='VisiblePoints')


class VisiblePoints:

    def __init__(self,
                 mesh: Union[Mesh, List[ndarray]]):
        """
        Extract the visible nodes of a mesh depending on the camera parameters.
        If using in a plotter instance, use vedo.visible_points instead.

        :param mesh: Mesh to compute visible surface extraction. Either a vedo.Mesh instance or a list containing the
                     positions array and the cells array sucha as [positions_array, cells_array].
        """

        # Mesh information
        self._mesh: Mesh = mesh if type(mesh) == Mesh else Mesh(mesh)
        self._visible_points: ndarray = array([])
        self._indices_visible_points: List[int] = []

        # Camera information
        self._camera_position: List[float] = [0., 0., 1.]
        self._focal_point: List[float] = [0., 0., 0.]
        self._view_angle: float = 30.
        self._distance_threshold = float('Inf')

    def set_camera(self,
                   camera_position: Optional[List[float]] = None,
                   focal_point: Optional[List[float]] = None,
                   view_angle: Optional[float] = None,
                   distance_threshold: Optional[float] = None) -> TVisiblePoints:
        """
        Define the camera parameters.

        :param camera_position: Current position of the camera.
        :param focal_point: Current focal point of the camera.
        :param view_angle: Current view angle of the camera.
        :param distance_threshold: Max distance to the camera for a point to be detected.
        """

        self._camera_position = self._camera_position if camera_position is None else camera_position
        self._focal_point = self._focal_point if focal_point is None else focal_point
        self._view_angle = self._view_angle if view_angle is None else view_angle
        self._distance_threshold = self._distance_threshold if distance_threshold is None else distance_threshold
        return self

    def extract(self,
                positions: Optional[ndarray] = None) -> Tuple[ndarray, List[int]]:
        """
        Return the positions and the indices of the current visible nodes.

        :param positions: Current positions of the mesh.
        """

        # Launch the extractor in a new thread
        t = Thread(target=self._extract, args=(positions,))
        t.start()
        t.join()
        return self._visible_points, self._indices_visible_points

    def _extract(self, positions):

        # Create the mesh
        mesh = self._mesh.clone() if positions is None else Mesh([positions, self._mesh.cells()])
        mesh.compute_normals()

        # Format the camera with position and focal point, extract visible nodes
        cam = dict(pos=self._camera_position, focalPoint=self._focal_point, viewAngle=self._view_angle)
        plt = show(mesh, new=True, camera=cam, offscreen=True)
        visible_pcd = visible_points(mesh)
        plt.close()

        # Build visible node indices and position list filtered with distance to camera
        filtered_visible_points = []
        list_positions = np_round(mesh.points().copy(), 6).tolist()
        indices = []
        for point in visible_pcd.points():
            if norm(point - self._camera_position) < self._distance_threshold:
                filtered_visible_points.append(point)
                position = np_round(point, 6).tolist()
                try:
                    indices.append(list_positions.index(position))
                except ValueError:
                    pass
        self._visible_points = array(filtered_visible_points)
        self._indices_visible_points = indices
