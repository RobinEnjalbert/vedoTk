from typing import Optional, List
from os import listdir, remove
from os.path import join, abspath, exists, sep, dirname
from numpy import array, ndarray, load, save, unique, dot, argwhere, mean
from vedo import Plotter, Mesh, Points, Text2D, settings
from vedo.colors import get_color


class MeshPointsSelection(Plotter):

    def __init__(self, mesh_file: str, selection_file: Optional[str] = None):
        """
        Implementation of a vedo.Plotter that allows selecting the points of a mesh.

        :param mesh_file: Path to the mesh file.
        :param selection_file: Path to an existing selection file.
        """

        Plotter.__init__(self)
        settings.use_parallel_projection = True

        self.__mesh_file = mesh_file
        self.__selection_file = selection_file

        # Create the mesh and the point cloud
        self.__mesh = Mesh(self.__mesh_file).compute_normals()
        self.__mesh.lw(1).c('grey')
        self.__points = Points(self.__mesh.vertices).point_size(8)

        # Default selection of the point cloud
        color = array(list(get_color('lightgreen')) + [1.]) * 255
        self.__default_color = array([color for _ in range(self.__points.nvertices)])
        self.__selection = set([]) if self.__selection_file is None else set(load(self.__selection_file).tolist())
        self.__info = Text2D('Nb selected point: 0', pos='bottom-left', s=0.7)
        self.__cursor = set([])
        self.__undo = set([])
        self.__draw_mode = False

        # Mouse and keyboard callbacks
        self.add_callback('MouseMove', self.__callback_mouse_move)
        self.add_callback('LeftButtonPress', self.__callback_left_click)
        self.add_callback('RightButtonPress', self.__callback_right_click)
        self.add_callback('KeyPress', self.__callback_key_press)

        # Selection radius slider
        self.__radius = 0.
        self.add_slider(sliderfunc=self.__callback_slider, xmin=0, xmax=self.__mesh.diagonal_size() * 0.1)

    @property
    def selected_points_id(self) -> ndarray:
        """
        Get the ids of the selected points.
        """

        return array(list(self.__selection))

    @property
    def selected_points_coord(self) -> ndarray:
        """
        Get the coordinates of the selected points.
        """

        if len(self.selected_points_id) > 0:
            return self.__points.vertices[self.selected_points_id]
        return array([])

    def launch(self, **kwargs) -> None:
        """
        Launch the Plotter. Specify Plotter.show() arguments in kwargs.
        """

        # Plotter legend
        self.render()
        instructions = "MOUSE CONTROL\n" \
                       "  Left click: add a point to selection\n" \
                       "  Right click: remove a point from selection\n\n" \
                       "KEYBOARD CONTROL\n" \
                       "  'z': remove the last selected point\n" \
                       "  'c': clear the selection\n" \
                       "  'd': switch between 'click' and 'draw' modes\n"
        self.add(Text2D(txt=instructions, pos='top-left', s=0.6, bg='grey', c='white', alpha=0.9))
        self.add(self.__info)

        # Add the data to the Plotter and color existing selection
        self.add(self.__mesh, self.__points)
        self.__update()

        # Launch the Plotter
        self.show(**kwargs).close()

    def save(self, selection_file: Optional[str] = None, overwrite: bool = False) -> None:
        """
        Save the current selection.
        """

        # Get the file name
        if selection_file is None and self.__selection_file is None:
            filename = 'selection.npy'
        elif selection_file is None:
            filename = self.__selection_file.split(sep)[-1]
        else:
            filename = selection_file.split(sep)[-1]
        filename = filename.split('.')[0]

        # Get the file dir
        if selection_file is None and self.__selection_file is None:
            filedir = abspath(dirname(self.__mesh_file))
        elif selection_file is None:
            filedir = abspath(dirname(self.__selection_file))
        else:
            filedir = sep.join(selection_file.split(sep)[:-1])

        # Indexing file
        file = join(filedir, filename) if len(filedir) > 0 else filename
        if exists(file):
            if overwrite:
                remove(file)
            else:
                nb_file = len([f for f in listdir(filedir) if f[:len(filename)] == filename])
                file = join(filedir, f'{filename}_{nb_file}')

        # Save selection
        save(f'{file}.npy', array(list(self.__selection), dtype=int))
        print(f'Saved selection at {file}.npy')

    def __update(self, color_cursor: bool = True) -> None:
        """
        Update the point cloud colors.
        """

        # Add the cursor to the selection if a point is flown over
        ids = self.__selection.copy()
        if len(self.__cursor) > 0 and color_cursor:
            ids = ids.union(self.__cursor)

        # Update color array
        color = list(get_color('tomato')) + [1.]
        colors = self.__default_color.copy()
        colors[list(ids)] = array(color) * 255

        # Update
        self.__points.pointcolors = colors
        self.__info.text(f'Nb selected points: {len(self.__selection)}')
        self.render()

    def __get_closest_points(self, picked: ndarray) -> List[int]:
        """

        """

        # Get the closest points from the mouse cursor within the defined radius
        ids = self.__points.closest_point(picked, return_point_id=True, radius=self.__radius)
        ids = array([ids]) if isinstance(ids, int) else array(ids)

        # Return empty selection
        if len(ids) == 0:
            return []

        # Filter the points with normals
        picked_normal = self.__mesh.point_normals[self.__points.closest_point(picked, return_point_id=True)]
        dots = array([dot(picked_normal, vertex_normal) for vertex_normal in self.__mesh.vertex_normals[ids]])
        return ids[argwhere(dots > 0).flatten()].tolist()

    def __callback_slider(self, widget, event) -> None:
        """
        Slider callback.
        """

        self.__radius = widget.value

    def __callback_mouse_move(self, event) -> None:
        """
        MouseMoveEvent callback.
        """

        # Cursor on the point cloud: color the hovered point
        if event.actor:
            self.__cursor = set(self.__get_closest_points(event.picked3d))
            if self.__draw_mode and len(self.__cursor) > 0:
                self.__selection = self.__selection.union(self.__cursor)

        # Cursor out of the point cloud: uncolor the hovered point
        else:
            self.__cursor = set([])

        # Update colors
        self.__update()

    def __callback_left_click(self, event) -> None:
        """
        LeftButtonPressEvent callback.
        """

        # Cursor on the point cloud: add the point to selection
        if event.actor:
            self.__cursor = set(self.__get_closest_points(event.picked3d))
            if len(self.__cursor) > 0:
                self.__undo = self.__cursor.difference(self.__selection)
                self.__selection = self.__selection.union(self.__cursor)
                self.__update()

    def __callback_right_click(self, event):
        """
        RightButtonPressEvent callback.
        """

        # Cursor on the point cloud: remove the point from selection
        if event.actor:
            self.__get_closest_points(event.picked3d)
            self.__selection = self.__selection.difference(self.__cursor)
            self.__update(color_cursor=False)

    def __callback_key_press(self, event):
        """
        KeyPressEvent callback.
        """

        # 'z' pressed: remove the last selected cell
        if event.keypress == 'z' and len(self.__selection) > 0:
            self.__selection = self.__selection.difference(self.__undo)
            self.__update()

        # 'c' pressed: clear the selection
        elif event.keypress == 'c':
            self.__selection = set([])
            self.__update()

        # 'd' pressed: switch draw flag
        if event.keypress == 'd':
            self.__draw_mode = not self.__draw_mode


class MeshCellsSelection(Plotter):

    def __init__(self, mesh_file: str, selection_file: Optional[str] = None):
        """
        Implementation of a vedo.Plotter that allows selecting the cells of a mesh.

        :param mesh_file: Path to the mesh file.
        :param selection_file: Path to an existing selection file.
        """

        Plotter.__init__(self)
        settings.use_parallel_projection = True

        self.__mesh_file = mesh_file
        self.__selection_file = selection_file

        # Create the mesh
        self.__mesh = Mesh(self.__mesh_file).compute_normals()
        self.__mesh.lw(1).c('grey')

        # Default selection of the point cloud
        color = array(list(get_color('lightgreen')) + [1.]) * 255
        self.__default_color = array([color for _ in range(self.__mesh.ncells)])
        self.__selection = set([]) if self.__selection_file is None else set(load(self.__selection_file).tolist())
        self.__info = Text2D('Nb selected point: 0', pos='bottom-left', s=0.7)
        self.__cursor = set([])
        self.__undo = set([])
        self.__draw_mode = False

        # Mouse and keyboard callbacks
        self.add_callback('MouseMove', self.__callback_mouse_move)
        self.add_callback('LeftButtonPress', self.__callback_left_click)
        self.add_callback('RightButtonPress', self.__callback_right_click)
        self.add_callback('KeyPress', self.__callback_key_press)

        # Selection radius slider
        self.__radius = 0.
        self.add_slider(sliderfunc=self.__callback_slider, xmin=0, xmax=int(self.__mesh.diagonal_size() * 0.1))

        # Define the cells centers point cloud
        self.__points = Points(mean(self.__mesh.vertices[self.__mesh.cells], axis=1))

    @property
    def selected_cells_id(self) -> ndarray:
        """
        Get the ids of the selected cells.
        """

        return array(list(self.__selection))

    @property
    def selected_cells_values(self) -> ndarray:
        """
        Get the values of the selected cells.
        """

        if len(self.selected_cells_id) > 0:
            return array(self.__mesh.cells)[self.selected_cells_id]
        return array([])

    @property
    def selected_points_id(self) -> ndarray:
        """
        Get the ids of the points in the selected cells.
        """

        return unique(self.selected_cells_values.flatten())

    @property
    def selected_points_coord(self) -> ndarray:
        """
        Get the coordinates of the points in the selected cells.
        """

        if len(self.selected_points_id) > 0:
            return array(self.__mesh.vertices)[self.selected_points_id]
        return array([])

    def launch(self, **kwargs) -> None:
        """
        Launch the Plotter. Specify Plotter.show() arguments in kwargs.
        """

        # Plotter legend
        self.render()
        instructions = "MOUSE CONTROL\n" \
                       "  Left click: add a point to selection\n" \
                       "  Right click: remove a point from selection\n\n" \
                       "KEYBOARD CONTROL\n" \
                       "  'z': remove the last selected point\n" \
                       "  'c': clear the selection\n" \
                       "  'd': switch between 'click' and 'draw' modes\n"
        self.add(Text2D(txt=instructions, pos='top-left', s=0.6, bg='grey', c='white', alpha=0.9))
        self.add(self.__info)

        # Add the data to the Plotter and color existing selection
        self.add(self.__mesh)
        self.__update()

        # Launch the Plotter
        self.show(**kwargs).close()

    def save(self, selection_file: Optional[str] = None, overwrite: bool = False) -> None:
        """
        Save the current selection.
        """

        # Get the file name
        if selection_file is None and self.__selection_file is None:
            filename = 'selection.npy'
        elif selection_file is None:
            filename = self.__selection_file.split(sep)[-1]
        else:
            filename = selection_file.split(sep)[-1]
        filename = filename.split('.')[0]

        # Get the file dir
        if selection_file is None and self.__selection_file is None:
            filedir = abspath(dirname(self.__mesh_file))
        elif selection_file is None:
            filedir = abspath(dirname(self.__selection_file))
        else:
            filedir = sep.join(selection_file.split(sep)[:-1])

        # Indexing file
        file = join(filedir, filename) if len(filedir) > 0 else filename
        if exists(file):
            if overwrite:
                remove(file)
            else:
                nb_file = len([f for f in listdir(filedir) if f[:len(filename)] == filename])
                file = join(filedir, f'{filename}_{nb_file}')

        # Save selection
        save(f'{file}.npy', array(list(self.__selection), dtype=int))
        print(f'Saved selection at {file}.npy')

    def __update(self, color_cursor: bool = True) -> None:
        """
        Update the point cloud colors.
        """

        # Add the cursor to the selection if a cell is flown over
        ids = self.__selection.copy()
        if len(self.__cursor) > 0 and color_cursor:
            ids = ids.union(self.__cursor)

        # Update color array
        color = list(get_color('tomato')) + [1.]
        colors = self.__default_color.copy()
        colors[list(ids)] = array(color) * 255

        # Update
        self.__mesh.cellcolors = colors
        self.__info.text(f'Nb selected cells: {len(self.__selection)}')
        self.render()

    def __get_closest_cells(self, picked: ndarray) -> List[int]:
        """

        """

        # Get the closest points from the mouse cursor within the defined radius
        ids = self.__points.closest_point(picked, return_point_id=True, radius=self.__radius)
        ids = array([ids]) if isinstance(ids, int) else array(ids)

        # Return empty selection
        if len(ids) == 0:
            return []

        # Filter the points with normals
        picked_normal = self.__mesh.cell_normals[self.__mesh.closest_point(picked, return_cell_id=True)]
        dots = array([dot(picked_normal, cell_normal) for cell_normal in self.__mesh.cell_normals[ids]])
        return ids[argwhere(dots > 0).flatten()].tolist()

    def __callback_slider(self, widget, event) -> None:
        """
        Slider callback.
        """

        self.__radius = widget.value

    def __callback_mouse_move(self, event) -> None:
        """
        MouseMoveEvent callback.
        """

        # Cursor on the point cloud: color the hovered point
        if event.actor:
            self.__cursor = set(self.__get_closest_cells(event.picked3d))
            if self.__draw_mode and len(self.__cursor) > 0:
                self.__selection = self.__selection.union(self.__cursor)

        # Cursor out of the point cloud: uncolor the hovered point
        else:
            self.__cursor = set([])

        # Update colors
        self.__update()

    def __callback_left_click(self, event) -> None:
        """
        LeftButtonPressEvent callback.
        """

        # Cursor on the point cloud: add the point to selection
        if event.actor:
            self.__cursor = set(self.__get_closest_cells(event.picked3d))
            if len(self.__cursor) > 0:
                self.__undo = self.__cursor.difference(self.__selection)
                self.__selection = self.__selection.union(self.__cursor)
                self.__update()

    def __callback_right_click(self, event):
        """
        RightButtonPressEvent callback.
        """

        # Cursor on the point cloud: remove the point from selection
        if event.actor:
            self.__get_closest_cells(event.picked3d)
            self.__selection = self.__selection.difference(self.__cursor)
            self.__update(color_cursor=False)

    def __callback_key_press(self, event):
        """
        KeyPressEvent callback.
        """

        # 'z' pressed: remove the last selected cell
        if event.keypress == 'z' and len(self.__selection) > 0:
            self.__selection = self.__selection.difference(self.__undo)
            self.__update()

        # 'c' pressed: clear the selection
        elif event.keypress == 'c':
            self.__selection = set([])
            self.__update()

        # 'd' pressed: switch draw flag
        if event.keypress == 'd':
            self.__draw_mode = not self.__draw_mode
