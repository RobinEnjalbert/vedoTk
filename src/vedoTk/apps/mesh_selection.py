from typing import Optional, Union
from os import listdir, remove
from os.path import join, abspath, exists, sep, dirname
from numpy import array, ndarray, load, save
from vedo import Plotter, Points, Mesh, TetMesh, Text2D, settings

from vedoTk.utils.mesh import SurfaceSelection


class MeshSelection(Plotter):

    def __init__(self, mesh: Union[str, Points, Mesh, TetMesh], selection_file: Optional[str] = None):
        """
        Plotter to select manually the vertices of a surface or tetra mesh.

        :param mesh: Path to the mesh file.
        :param selection_file: Path to an existing selection file.
        """

        super().__init__(title='Selection')
        settings.use_parallel_projection = True

        if isinstance(mesh, str):
            self.__mesh_file = mesh
            try:
                mesh = Mesh(mesh)
            except TypeError:
                mesh = TetMesh(mesh)
        else:
            self.__mesh_file = None

        self.__selection_file = selection_file
        self.__mesh = SurfaceSelection(mesh=mesh,
                                       selection_idx=None if selection_file is None else load(selection_file))

        self.__info = Text2D(f'Nb selected vertices: 0', pos='bottom-left', s=0.7)
        self.__cursor = array([])
        self.__history = []
        self.__history_idx = 0
        self.__draw_mode = False

        # Mouse and keyboard callbacks
        self.add_callback('MouseMove', self.__callback_mouse_move)
        self.add_callback('LeftButtonPress', self.__callback_left_click)
        self.add_callback('RightButtonPress', self.__callback_right_click)
        self.add_callback('KeyPress', self.__callback_key_press)

        # Selection radius slider
        self.__slider_area, self.__slider_radius = 0, 8
        self.add_slider(sliderfunc=self.__callback_slider_area, show_value=False, pos='bottom-right',
                        xmin=0, xmax=mesh.diagonal_size() * 0.1, title='Selection Area', title_size=0.8)
        self.add_slider(sliderfunc=self.__callback_slider_radius, show_value=False, pos='bottom-left',
                        xmin=8, xmax=30, title='Selected Points Radius', title_size=0.8)

    @property
    def selected_points_id(self) -> ndarray:
        """
        Get the ids of the selected points.
        """

        return self.__mesh.selected_idx

    @property
    def selected_points_coord(self) -> ndarray:
        """
        Get the coordinates of the selected points.
        """

        return self.__mesh.selected_points_coord

    def launch(self, **kwargs) -> None:
        """
        Launch the Plotter. Specify Plotter.show() arguments in kwargs.
        """

        close_viewer = kwargs.pop('close_viewer', True)

        # Plotter legend
        self.render()
        instructions = " MOUSE CONTROL\n" \
                       "   Left click: add a point to selection\n" \
                       "   Right click: remove a point from selection\n\n" \
                       " KEYBOARD CONTROL\n" \
                       "   'Ctrl+d': switch between 'click' and 'draw' modes \n" \
                       "   'Ctrl+z': undo the selection\n" \
                       "   'Ctrl+y': redo the selection\n" \
                       "   'Ctrl+c': clear the selection\n" \
                       "   'Ctrl+i ': invert the selection\n" \
                       "   'Ctrl+a': select all\n"
        self.add(Text2D(txt=instructions, pos='top-left', s=0.6, bg='grey', c='white', alpha=0.9))
        self.add(self.__info)

        # Add the data to the Plotter and color existing selection
        self.__mesh.init(plt=self)
        self.__update()

        # Launch the Plotter
        self.show(**kwargs)

        if close_viewer:
            self.close()

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
        save(f'{file}.npy', self.__mesh.selected_idx)
        print(f'Saved selection at {file}.npy')

    def __update(self, color_cursor: bool = True, history: bool = True) -> None:
        """
        Update the point cloud colors.
        """

        # Update the visual selection
        self.__mesh.update_selection(plt=self, cursor_idx=self.__cursor if color_cursor else array([]),
                                     point_size=self.__slider_radius)

        # Keep selection in history
        if history:
            self.__history = self.__history[:self.__history_idx + 1] + [self.__mesh.selected_idx]
            self.__history_idx = len(self.__history) - 1

        # Update the rendering
        self.__info.text(f'Nb selected vertices: {len(self.__mesh.selected_idx)}')
        self.render()

    def __callback_slider_area(self, widget, event) -> None:
        """
        Slider callback.
        """

        self.__slider_area = widget.value

    def __callback_slider_radius(self, widget, event) -> None:
        """
        Slider callback.
        """

        self.__slider_radius = widget.value
        self.__update(history=False)

    def __callback_mouse_move(self, event) -> None:
        """
        MouseMoveEvent callback.
        """

        # Cursor on the surface: color the hovered point(s)
        if event.actor:
            self.__cursor = self.__mesh.get_closest_points(pos=event.picked3d, radius=self.__slider_area)
            if self.__draw_mode and len(self.__cursor) > 0:
                self.__mesh.add(self.__cursor)

        # Cursor out of the point cloud: uncolor the hovered point
        else:
            self.__cursor = array([])

        # Update colors
        self.__update(history=False)

    def __callback_left_click(self, event) -> None:
        """
        LeftButtonPressEvent callback.
        """

        # Cursor on the surface: add the point(s) to selection
        if event.actor:
            self.__cursor = self.__mesh.get_closest_points(pos=event.picked3d, radius=self.__slider_area)
            if len(self.__cursor) > 0:
                self.__mesh.add(idx=self.__cursor)
                self.__update(history=not self.__draw_mode)

    def __callback_right_click(self, event):
        """
        RightButtonPressEvent callback.
        """

        # Cursor on the point cloud: remove the point from selection
        if event.actor:
            self.__cursor = self.__mesh.get_closest_points(pos=event.picked3d, radius=self.__slider_area)
            if len(self.__cursor) > 0:
                self.__mesh.remove(idx=self.__cursor)
                self.__update(color_cursor=False)

    def __callback_key_press(self, event):
        """
        KeyPressEvent callback.
        """

        # 'ctrl+z' pressed: undo
        if event.keypress == 'Ctrl+z':
            if self.__history_idx > 0:
                self.__history_idx -= 1
                self.__mesh.selected_idx = self.__history[self.__history_idx]
                self.__update(history=False)

        # 'ctrl+y' pressed: redo
        elif event.keypress == 'Ctrl+y':
            if self.__history_idx < len(self.__history) - 1:
                self.__history_idx += 1
                self.__mesh.selected_idx = self.__history[self.__history_idx]
                self.__update(history=False)

        # 'ctrl+c' pressed: clear the selection
        elif event.keypress == 'Ctrl+c':
            self.__mesh.clear()
            self.__update()

        # 'ctrl+d' pressed: switch draw flag
        elif event.keypress == 'Ctrl+d':
            self.__draw_mode = not self.__draw_mode
            if not self.__draw_mode:
                self.__history = self.__history[:self.__history_idx + 1] + [self.__mesh.selected_idx]
                self.__history_idx = len(self.__history) - 1

        # 'ctrl+a' pressed: select all the vertices
        elif event.keypress == 'Ctrl+a':
            self.__mesh.all()
            self.__update()

        # 'ctrl+i' pressed: invert the selection
        elif event.keypress == 'Ctrl+i':
            self.__mesh.invert()
            self.__update()
